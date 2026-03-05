"""統一股票分析服務 — 6 階段管線

取代原有兩條獨立路線（Pipeline + Market Scan），整合為單一入口。

6 階段:
1. 數據收集 (並行)
2. 特徵萃取 (20 因子 + HMM + ML + LLM 情緒)
3. 多因子評分
4. LLM 敘事生成
5. 風險控制 + 部位建議
6. 儲存 + 警報
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import AsyncGenerator

import numpy as np
import pandas as pd

from src.utils.constants import STOCK_LIST
from src.utils.config import settings
from src.data.twse_scanner import TWSEScanner
from src.data.stock_fetcher import StockFetcher
from src.db.database import (
    get_stock_prices, upsert_stock_prices,
    save_market_scan, save_pipeline_result_record,
    save_factor_ic_records, get_sentiment,
)
from src.analysis.technical import TechnicalAnalyzer
from api.services.market_service import (
    FactorResult, BASE_WEIGHTS, REGIME_MULTIPLIERS, STOCK_SECTOR,
    score_stock, _compute_weights, _build_reasoning,
    _fetch_global_market_data, _fetch_macro_data,
    _compute_sector_aggregates,
)
from api.services.alert_service import generate_alerts_from_scan

logger = logging.getLogger(__name__)

MODEL_DIR = settings.PROJECT_ROOT / "models"


# ═══════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════


@dataclass
class StockData:
    """Phase 1 收集的原始數據"""
    stock_id: str
    stock_name: str
    df: pd.DataFrame                     # OHLCV + institutional flows
    df_tech: pd.DataFrame                # Technical indicators
    signals: dict                        # Technical signals summary
    trust_info: dict                     # {cumulative, consecutive_days} per fund type
    revenue_df: pd.DataFrame | None      # Monthly revenue
    global_data: dict | None             # SOX/TSM/EWT
    macro_data: dict | None              # VIX/FX/TNX/XLI
    sentiment_df: pd.DataFrame | None    # News/PTT sentiment records
    sentiment_scores: dict               # {stock_id: avg_score}
    fundamental_data: dict | None = None # yfinance info + quarterly_income_stmt
    per_pbr_df: pd.DataFrame | None = None  # FinMind 每日 P/E, P/B
    current_price: float = 0.0
    price_change_pct: float = 0.0


@dataclass
class ScoreResult:
    """Phase 3 評分結果"""
    total_score: float
    signal: str                # strong_buy/buy/hold/sell/strong_sell
    confidence: float
    confidence_breakdown: dict
    factor_details: dict       # 20 factors full transparency
    factors: list              # FactorResult list (internal)
    weights: dict
    reasoning: list[str]
    regime: str


@dataclass
class NarrativeResult:
    """Phase 4 LLM 敘事結果"""
    outlook: str = ""
    outlook_horizon: str = ""
    key_drivers: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    catalysts: list[str] = field(default_factory=list)
    key_levels: dict = field(default_factory=dict)
    position_suggestion: str = ""
    source: str = "algorithm"  # "llm" or "algorithm"


@dataclass
class RiskDecision:
    """Phase 5 風控結果"""
    action: str                # buy/sell/hold
    position_size: float       # 0.0-0.20
    approved: bool = True
    risk_notes: list[str] = field(default_factory=list)
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class AnalysisResult:
    """Phase 6 最終完整結果"""
    stock_id: str
    stock_name: str
    current_price: float
    price_change_pct: float
    # Scoring
    total_score: float
    signal: str
    confidence: float
    confidence_breakdown: dict
    factor_details: dict
    regime: str
    reasoning: str
    # Narrative
    narrative: dict
    # Risk
    risk_decision: dict
    # Metadata
    analysis_date: str
    pipeline_version: str = "3.0"


# ═══════════════════════════════════════════════════════
# SSE helper
# ═══════════════════════════════════════════════════════


def _sse_event(phase: str, status: str, progress: int,
               message: str = "", data: dict | None = None) -> str:
    """Format SSE event for streaming"""
    payload = {
        "phase": phase,
        "status": status,
        "progress": progress,
        "message": message,
    }
    if data is not None:
        payload["data"] = data
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


# ═══════════════════════════════════════════════════════
# StockAnalysisService
# ═══════════════════════════════════════════════════════


class StockAnalysisService:
    """統一股票分析服務 — 6 階段管線

    單一入口: analyze_stock(stock_id) → SSE stream
    """

    def __init__(self):
        self._risk_manager = None

    def _get_risk_manager(self):
        if self._risk_manager is None:
            from src.risk.manager import RiskManager
            self._risk_manager = RiskManager()
        return self._risk_manager

    # ─── Main entry point ────────────────────────────────

    async def analyze_stock(
        self,
        stock_id: str,
        stock_name: str = "",
    ) -> AsyncGenerator[str, None]:
        """單一入口：對指定股票執行完整 6 階段分析，透過 SSE 串流回報進度。"""
        import time as _time
        stock_name = stock_name or STOCK_LIST.get(stock_id, stock_name)
        t_total = _time.perf_counter()

        logger.info("═══ 開始分析 %s %s ═══", stock_id, stock_name)

        # Phase 1: 數據收集
        logger.info("[Phase 1/6] 數據收集 — 開始 (%s)", stock_id)
        t0 = _time.perf_counter()
        yield _sse_event("data_collection", "running", 5, f"收集 {stock_id} {stock_name} 數據...")
        try:
            data = await self._collect_data(stock_id, stock_name)
        except Exception as e:
            logger.error("[Phase 1/6] 數據收集 — 失敗: %s", e, exc_info=True)
            yield _sse_event("data_collection", "error", 5, f"數據收集失敗: {e}")
            yield _sse_event("complete", "error", 100, "分析失敗", {
                "stock_id": stock_id, "error": str(e),
            })
            return

        if data.df.empty or len(data.df) < 20:
            logger.warning("[Phase 1/6] 數據收集 — 資料不足 (rows=%d)", len(data.df))
            yield _sse_event("complete", "error", 100, "價格資料不足", {
                "stock_id": stock_id, "error": "價格資料不足，無法分析",
            })
            return

        logger.info(
            "[Phase 1/6] 數據收集 — 完成 (%.1fs) | 價格=%d筆 技術=%d筆 現價=%.1f 漲跌=%.2f%% 法人=%s 營收=%s 情緒=%s",
            _time.perf_counter() - t0,
            len(data.df), len(data.df_tech),
            data.current_price, data.price_change_pct,
            "有" if data.trust_info else "無",
            "有" if data.revenue_df is not None else "無",
            "有" if data.sentiment_df is not None else "無",
        )
        yield _sse_event("data_collection", "done", 15, "數據收集完成")

        # Phase 2: 特徵萃取
        logger.info("[Phase 2/6] 特徵萃取 — 開始 (HMM + ML + LLM 情緒)")
        t0 = _time.perf_counter()
        yield _sse_event("feature_extraction", "running", 20, "計算 20 因子 + HMM + ML...")
        regime, ml_scores = await self._extract_features(data)
        logger.info(
            "[Phase 2/6] 特徵萃取 — 完成 (%.1fs) | regime=%s ml_scores=%s",
            _time.perf_counter() - t0, regime, ml_scores or "無模型",
        )
        yield _sse_event("feature_extraction", "done", 40, "特徵萃取完成")

        # Phase 3: 多因子評分
        logger.info("[Phase 3/6] 多因子評分 — 開始")
        t0 = _time.perf_counter()
        yield _sse_event("scoring", "running", 45, "多因子加權評分...")
        score_result = self._score(data, regime, ml_scores)
        avail_count = sum(1 for f in score_result.factors if f.available)
        logger.info(
            "[Phase 3/6] 多因子評分 — 完成 (%.1fs) | 總分=%.3f 訊號=%s 信心=%.2f 可用因子=%d/20 regime=%s",
            _time.perf_counter() - t0,
            score_result.total_score, score_result.signal, score_result.confidence,
            avail_count, regime,
        )
        yield _sse_event("scoring", "done", 55, f"評分完成: {score_result.total_score:.2f} ({score_result.signal})")

        # Phase 4: LLM 敘事生成
        logger.info("[Phase 4/6] LLM 敘事生成 — 開始")
        t0 = _time.perf_counter()
        yield _sse_event("narrative", "running", 60, "生成分析報告...")
        narrative = await self._generate_narrative(
            stock_id, stock_name, data, score_result, regime, ml_scores
        )
        logger.info(
            "[Phase 4/6] LLM 敘事生成 — 完成 (%.1fs) | source=%s outlook=%s",
            _time.perf_counter() - t0, narrative.source,
            narrative.outlook[:40] + "..." if len(narrative.outlook) > 40 else narrative.outlook,
        )
        yield _sse_event("narrative", "done", 75,
                         f"敘事生成完成 (source={narrative.source})")

        # Phase 5: 風險控制
        logger.info("[Phase 5/6] 風險控制 — 開始")
        t0 = _time.perf_counter()
        yield _sse_event("risk_control", "running", 80, "風險檢查...")
        risk_decision = self._apply_risk_controls(
            stock_id, data, score_result, regime
        )
        logger.info(
            "[Phase 5/6] 風險控制 — 完成 (%.1fs) | action=%s position=%.2f%% approved=%s stop=%.1f profit=%.1f notes=%s",
            _time.perf_counter() - t0,
            risk_decision.action, risk_decision.position_size * 100,
            risk_decision.approved,
            risk_decision.stop_loss or 0, risk_decision.take_profit or 0,
            risk_decision.risk_notes or "無",
        )
        yield _sse_event("risk_control", "done", 88,
                         f"風控完成: {risk_decision.action} (approved={risk_decision.approved})")

        # Phase 6: 儲存 + 警報
        logger.info("[Phase 6/6] 儲存結果 — 開始")
        t0 = _time.perf_counter()
        yield _sse_event("finalize", "running", 90, "儲存結果...")
        result = self._build_result(
            data, score_result, narrative, risk_decision, regime
        )
        await self._save_and_alert(data, score_result, result)
        logger.info("[Phase 6/6] 儲存結果 — 完成 (%.1fs)", _time.perf_counter() - t0)
        yield _sse_event("finalize", "done", 95, "儲存完成")

        # Complete
        elapsed = _time.perf_counter() - t_total
        logger.info(
            "═══ 分析完成 %s %s ═══ 總耗時 %.1fs | 總分=%.3f 訊號=%s 信心=%.2f regime=%s action=%s approved=%s",
            stock_id, stock_name, elapsed,
            result.total_score, result.signal, result.confidence,
            result.regime, risk_decision.action, risk_decision.approved,
        )
        yield _sse_event("complete", "done", 100, "分析完成", asdict(result))

    # ─── Phase 1: Data Collection ────────────────────────

    async def _collect_data(self, stock_id: str, stock_name: str) -> StockData:
        """並行收集所有數據源"""
        logger.info("  ├─ 並行抓取: 價格 / 法人 / 營收 / 全球 / 宏觀 / 情緒 / 基本面 / P/E ...")

        # Parallel fetches
        (df, trust_info, revenue_df, global_data, macro_data, sentiment_result, fundamental_data, per_pbr_df) = (
            await asyncio.gather(
                self._fetch_price_data(stock_id),
                self._fetch_trust_info(stock_id),
                self._fetch_revenue(stock_id),
                asyncio.to_thread(_fetch_global_market_data),
                asyncio.to_thread(_fetch_macro_data),
                self._fetch_sentiment(stock_id),
                self._fetch_fundamental(stock_id),
                self._fetch_per_pbr(stock_id),
                return_exceptions=True,
            )
        )

        # Handle exceptions from gather
        fetch_status = []
        if isinstance(df, Exception):
            logger.error("  │  ✗ 價格抓取失敗: %s", df)
            df = pd.DataFrame()
            fetch_status.append("價格:✗")
        else:
            fetch_status.append(f"價格:✓({len(df)}筆)")
        if isinstance(trust_info, Exception):
            logger.warning("  │  ✗ 法人資料失敗: %s", trust_info)
            trust_info = {}
            fetch_status.append("法人:✗")
        else:
            fetch_status.append(f"法人:{'✓' if trust_info else '空'}")
        if isinstance(revenue_df, Exception):
            logger.warning("  │  ✗ 營收資料失敗: %s", revenue_df)
            revenue_df = None
            fetch_status.append("營收:✗")
        else:
            fetch_status.append(f"營收:{'✓' if revenue_df is not None else '空'}")
        if isinstance(global_data, Exception):
            logger.warning("  │  ✗ 全球市場失敗: %s", global_data)
            global_data = {}
            fetch_status.append("全球:✗")
        else:
            fetch_status.append(f"全球:{'✓' if global_data else '空'}")
        if isinstance(macro_data, Exception):
            logger.warning("  │  ✗ 宏觀資料失敗: %s", macro_data)
            macro_data = {}
            fetch_status.append("宏觀:✗")
        else:
            fetch_status.append(f"宏觀:{'✓' if macro_data else '空'}")
        if isinstance(sentiment_result, Exception):
            logger.warning("  │  ✗ 情緒資料失敗: %s", sentiment_result)
            sentiment_result = (None, {})
            fetch_status.append("情緒:✗")
        else:
            fetch_status.append("情緒:✓")
        if isinstance(fundamental_data, Exception):
            logger.warning("  │  ✗ 基本面資料失敗: %s", fundamental_data)
            fundamental_data = None
            fetch_status.append("基本面:✗")
        else:
            fetch_status.append(f"基本面:{'✓' if fundamental_data else '空'}")
        if isinstance(per_pbr_df, Exception):
            logger.warning("  │  ✗ P/E P/B 失敗: %s", per_pbr_df)
            per_pbr_df = None
            fetch_status.append("P/E:✗")
        else:
            fetch_status.append(f"P/E:{'✓' if per_pbr_df is not None and not per_pbr_df.empty else '空'}")

        logger.info("  ├─ 抓取結果: %s", " | ".join(fetch_status))

        sentiment_df, sentiment_scores = sentiment_result

        # Technical analysis
        df_tech = pd.DataFrame()
        signals = {}
        if not df.empty and len(df) >= 20:
            try:
                analyzer = TechnicalAnalyzer()
                df_tech = analyzer.compute_all(df)
                signals = analyzer.get_signals(df_tech)
                summary = signals.get("summary", {})
                logger.info("  ├─ 技術分析完成: signal=%s score=%s/%s",
                            summary.get("signal", "?"),
                            summary.get("raw_score", "?"),
                            summary.get("max_score", "?"))
            except Exception as e:
                logger.warning("  ├─ 技術分析失敗: %s", e)

        # Current price
        current_price = 0.0
        price_change_pct = 0.0
        if not df.empty:
            current_price = float(df.iloc[-1]["close"]) if pd.notna(df.iloc[-1]["close"]) else 0.0
            if len(df) >= 2:
                prev_close = float(df.iloc[-2]["close"]) if pd.notna(df.iloc[-2]["close"]) else current_price
                if prev_close > 0:
                    price_change_pct = round((current_price - prev_close) / prev_close * 100, 2)

        return StockData(
            stock_id=stock_id,
            stock_name=stock_name,
            df=df,
            df_tech=df_tech,
            signals=signals,
            trust_info=trust_info,
            revenue_df=revenue_df,
            global_data=global_data,
            macro_data=macro_data,
            sentiment_df=sentiment_df,
            sentiment_scores=sentiment_scores,
            fundamental_data=fundamental_data,
            per_pbr_df=per_pbr_df,
            current_price=current_price,
            price_change_pct=price_change_pct,
        )

    async def _fetch_price_data(self, stock_id: str) -> pd.DataFrame:
        """Fetch 120 days of price data"""
        df = await asyncio.to_thread(get_stock_prices, stock_id)
        if df.empty:
            # Try fetching from external source
            try:
                fetcher = StockFetcher()
                end_date = date.today()
                start_date = end_date - timedelta(days=180)
                df = await asyncio.to_thread(
                    fetcher.fetch, stock_id,
                    start_date.isoformat(), end_date.isoformat()
                )
                if not df.empty:
                    await asyncio.to_thread(upsert_stock_prices, df, stock_id)
            except Exception as e:
                logger.warning("External fetch failed for %s: %s", stock_id, e)
        return df

    async def _fetch_trust_info(self, stock_id: str) -> dict:
        """Fetch institutional (T86) trust info"""
        try:
            scanner = TWSEScanner()
            result = await asyncio.to_thread(scanner.get_trust_info, stock_id, days=5)
            return result or {}
        except Exception:
            # Fallback: compute from price data
            return {}

    async def _fetch_revenue(self, stock_id: str) -> pd.DataFrame | None:
        """Fetch monthly revenue from FinMind"""
        try:
            from api.services.market_service import _fetch_revenue_batch
            result = await _fetch_revenue_batch([stock_id])
            return result.get(stock_id)
        except Exception:
            return None

    async def _fetch_sentiment(
        self, stock_id: str
    ) -> tuple[pd.DataFrame | None, dict]:
        """Fetch sentiment data from DB, live-crawl if DB has no recent data"""
        try:
            today = date.today()
            sent_df = await asyncio.to_thread(
                get_sentiment, stock_id, today - timedelta(days=14), today
            )

            # If DB has no recent sentiment, live-crawl from PTT/鉅亨/Google/Yahoo
            if sent_df is None or sent_df.empty:
                sent_df = await self._crawl_sentiment_live(stock_id)

            sentiment_scores = {}
            if sent_df is not None and not sent_df.empty:
                scores = sent_df["sentiment_score"].dropna()
                if not scores.empty:
                    avg = float(scores.mean())
                    sentiment_scores[stock_id] = (avg + 1) / 2  # Normalize to 0-1
            return sent_df, sentiment_scores
        except Exception as e:
            logger.warning("Sentiment fetch failed: %s", e)
            return None, {}

    async def _crawl_sentiment_live(self, stock_id: str) -> pd.DataFrame | None:
        """Live-crawl sentiment from PTT, 鉅亨, Google News, Yahoo TW"""
        try:
            from src.data.sentiment_crawler import SentimentCrawler
            from src.db.database import insert_sentiment

            crawler = SentimentCrawler()
            articles = await asyncio.to_thread(crawler.crawl_all, stock_id)
            if not articles:
                logger.info("  │  情緒爬蟲: 0 篇文章 (stock=%s)", stock_id)
                return None

            logger.info("  │  情緒爬蟲: %d 篇文章 (stock=%s)", len(articles), stock_id)

            # LLM batch-score titles that only have "neutral" label (from RSS/API sources)
            neutral_titles = [
                art.get("title", "") for art in articles
                if art.get("sentiment_label") == "neutral" and art.get("title")
            ]
            llm_labels = {}
            if neutral_titles:
                llm_labels = await self._batch_score_titles(stock_id, neutral_titles)

            # Assign sentiment_score from label
            label_to_score = {"bullish": 0.8, "bearish": -0.8, "neutral": 0.0}
            records = []
            for art in articles:
                title = art.get("title", "")
                # Override neutral labels with LLM results
                if art.get("sentiment_label") == "neutral" and title in llm_labels:
                    art["sentiment_label"] = llm_labels[title]
                art.setdefault("sentiment_label", "neutral")
                art.setdefault("sentiment_score",
                               label_to_score.get(art.get("sentiment_label", "neutral"), 0.0))
                art.setdefault("engagement", 0)
                records.append(art)

            # Persist to DB for future queries
            try:
                await asyncio.to_thread(insert_sentiment, records)
            except Exception as e:
                logger.warning("  │  情緒寫入 DB 失敗: %s", e)

            return pd.DataFrame(records)
        except Exception as e:
            logger.warning("  │  情緒爬蟲失敗: %s", e)
            return None

    async def _batch_score_titles(
        self, stock_id: str, titles: list[str]
    ) -> dict[str, str]:
        """Use LLM to batch-classify news titles as bullish/bearish/neutral"""
        try:
            from src.utils.llm_client import call_claude, parse_json_response
            from src.utils.constants import STOCK_LIST

            stock_name = STOCK_LIST.get(stock_id, stock_id)
            numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles[:30]))
            prompt = f"""對以下與 {stock_id} ({stock_name}) 相關的新聞標題進行情緒分類。

{numbered}

回傳 JSON array，每個元素對應一則標題：
[{{"idx": 1, "sentiment": "bullish"}}, {{"idx": 2, "sentiment": "bearish"}}, ...]

sentiment 只能是: bullish / bearish / neutral
規則：利多消息(營收成長/獲利/漲價/需求增加)=bullish，利空消息(裁員/虧損/下修/賣超)=bearish，中性或無法判斷=neutral
只回傳 JSON。"""

            text = await call_claude(prompt, model="claude-haiku-4-5-20251001", timeout=60)
            results = parse_json_response(text)
            if isinstance(results, list):
                mapping = {}
                for item in results:
                    idx = item.get("idx", 0) - 1
                    if 0 <= idx < len(titles):
                        mapping[titles[idx]] = item.get("sentiment", "neutral")
                logger.info("  │  LLM 標題情緒分類: %d/%d 完成", len(mapping), len(titles))
                return mapping
        except Exception as e:
            logger.warning("  │  LLM 標題分類失敗: %s", e)
        return {}

    async def _fetch_fundamental(self, stock_id: str) -> dict | None:
        """Fetch yfinance fundamental data (info + quarterly_income_stmt) for Phase 1 prefetch"""
        def _do_fetch():
            try:
                import yfinance as yf
                ticker = yf.Ticker(f"{stock_id}.TW")
                result = {}
                try:
                    info = ticker.info
                    if info:
                        result["info"] = info
                except Exception as e:
                    logger.warning("yfinance %s.TW info fetch failed: %s", stock_id, e)
                try:
                    qis = ticker.quarterly_income_stmt
                    if qis is not None and not qis.empty:
                        result["quarterly_income_stmt"] = qis
                except Exception as e:
                    logger.warning("yfinance %s.TW quarterly_income_stmt fetch failed: %s", stock_id, e)
                return result if result else None
            except Exception as e:
                logger.warning("yfinance %s.TW fundamental fetch failed: %s", stock_id, e)
                return None

        return await asyncio.to_thread(_do_fetch)

    async def _fetch_per_pbr(self, stock_id: str) -> pd.DataFrame | None:
        """Fetch FinMind daily P/E, P/B, dividend_yield"""
        try:
            fetcher = StockFetcher()
            end_date = date.today()
            start_date = end_date - timedelta(days=90)
            df = await asyncio.to_thread(
                fetcher.fetch_per_pbr, stock_id,
                start_date.isoformat(), end_date.isoformat()
            )
            return df if df is not None and not df.empty else None
        except Exception as e:
            logger.warning("FinMind P/E P/B fetch failed for %s: %s", stock_id, e)
            return None

    # ─── Phase 2: Feature Extraction ─────────────────────

    async def _extract_features(
        self, data: StockData
    ) -> tuple[str, dict]:
        """並行提取: HMM 體制 + ML 預測 + LLM 情緒增強

        20 因子在 Phase 3 計算 (同步，由 score_stock 處理)。
        Returns: (regime, ml_scores)
        """
        regime_task = asyncio.to_thread(self._detect_regime, data.df)
        ml_task = asyncio.to_thread(self._predict_ml, data.stock_id, data.df)
        sentiment_task = self._enhance_sentiment(data)

        regime, ml_scores, enhanced_sentiment = await asyncio.gather(
            regime_task, ml_task, sentiment_task,
            return_exceptions=True,
        )

        # Handle exceptions
        if isinstance(regime, Exception):
            logger.warning("HMM regime failed: %s", regime)
            regime = "sideways"
        if isinstance(ml_scores, Exception):
            logger.warning("ML predict failed: %s", ml_scores)
            ml_scores = {}
        if isinstance(enhanced_sentiment, Exception):
            logger.warning("LLM sentiment failed: %s", enhanced_sentiment)
            enhanced_sentiment = None

        # Merge enhanced sentiment into sentiment_scores
        if enhanced_sentiment and isinstance(enhanced_sentiment, dict):
            llm_score = enhanced_sentiment.get("sentiment_score")
            if llm_score is not None:
                data.sentiment_scores[data.stock_id] = llm_score

        return regime, ml_scores

    def _detect_regime(self, df: pd.DataFrame) -> str:
        """HMM 3-state regime detection"""
        try:
            from src.models.ensemble import HMMStateDetector
            if df.empty or len(df) < 60:
                return "sideways"
            close = df["close"].dropna()
            if len(close) < 60:
                return "sideways"
            returns = close.pct_change().dropna().values
            hmm = HMMStateDetector(n_states=3)
            hmm.fit(returns)
            state = hmm.predict_state(returns)
            return state.state_name
        except Exception as e:
            logger.warning("HMM detection failed: %s", e)
            return "sideways"

    def _predict_ml(self, stock_id: str, df: pd.DataFrame) -> dict:
        """ML model prediction (LSTM + XGBoost)"""
        ml_scores = {}
        try:
            lstm_path = MODEL_DIR / f"{stock_id}_lstm.pt"
            xgb_path = MODEL_DIR / f"{stock_id}_xgb.json"
            if not (lstm_path.exists() and xgb_path.exists()):
                return ml_scores

            from src.models.trainer import ModelTrainer
            trainer = ModelTrainer(stock_id)
            trainer.load_models()
            today = date.today()
            result = trainer.predict(
                start_date=(today - timedelta(days=200)).isoformat(),
                end_date=today.isoformat(),
            )
            if result is not None:
                signal_score = {
                    "strong_buy": 0.9, "buy": 0.75,
                    "hold": 0.5,
                    "sell": 0.25, "strong_sell": 0.1,
                }
                ml_scores[stock_id] = signal_score.get(result.signal, 0.5)
        except Exception as e:
            logger.warning("ML predict failed for %s: %s", stock_id, e)
        return ml_scores

    async def _enhance_sentiment(self, data: StockData) -> dict | None:
        """LLM 情緒特徵萃取 (Haiku, 1 call)

        Phase 2 的 LLM 呼叫 #1: 從新聞/PTT/法人動向萃取結構化情緒。
        """
        try:
            from src.agents.narrative_agent import extract_sentiment
            return await extract_sentiment(
                stock_id=data.stock_id,
                sentiment_df=data.sentiment_df,
                trust_info=data.trust_info,
                global_data=data.global_data,
            )
        except Exception as e:
            logger.warning("LLM sentiment extraction failed: %s", e)
            return None

    # ─── Phase 3: Multi-Factor Scoring ───────────────────

    def _score(
        self, data: StockData, regime: str, ml_scores: dict
    ) -> ScoreResult:
        """20 因子加權評分 — 直接重用 market_service.score_stock()"""

        # Prepare sector data for sector_rotation factor
        sector_data = None
        try:
            stock_dfs = {data.stock_id: data.df}
            trust_lookup = {}
            if data.trust_info:
                trust_lookup[data.stock_id] = data.trust_info
            sector_data = _compute_sector_aggregates(stock_dfs, trust_lookup)
        except Exception:
            pass

        stock_data = {
            "stock_id": data.stock_id,
            "stock_name": data.stock_name,
            "current_price": data.current_price,
            "price_change_pct": data.price_change_pct,
        }

        result = score_stock(
            stock_data=stock_data,
            df=data.df,
            df_tech=data.df_tech,
            signals=data.signals,
            trust_info=data.trust_info,
            sentiment_scores=data.sentiment_scores,
            sentiment_df=data.sentiment_df,
            ml_scores=ml_scores,
            regime=regime,
            revenue_df=data.revenue_df,
            global_data=data.global_data,
            macro_data=data.macro_data,
            sector_data=sector_data,
            fundamental_data=data.fundamental_data,
            per_pbr_df=data.per_pbr_df,
        )

        # Extract factors for later use
        factors = result.pop("_factors", [])

        # Build ScoreResult
        conf_breakdown = {
            "confidence_agreement": result.get("confidence_agreement", 0),
            "confidence_strength": result.get("confidence_strength", 0),
            "confidence_coverage": result.get("confidence_coverage", 0),
            "confidence_freshness": result.get("confidence_freshness", 0),
            "risk_discount": result.get("risk_discount", 1),
        }

        weights = _compute_weights(factors, regime) if factors else {}
        reasoning = _build_reasoning(
            factors, data.trust_info, data.signals, ml_scores, data.stock_id
        )

        return ScoreResult(
            total_score=result["total_score"],
            signal=result["signal"],
            confidence=result["confidence"],
            confidence_breakdown=conf_breakdown,
            factor_details=result.get("factor_details", {}),
            factors=factors,
            weights=weights,
            reasoning=reasoning,
            regime=regime,
        )

    # ─── Phase 4: Narrative Generation ───────────────────

    async def _generate_narrative(
        self,
        stock_id: str,
        stock_name: str,
        data: StockData,
        score_result: ScoreResult,
        regime: str,
        ml_scores: dict,
    ) -> NarrativeResult:
        """LLM 敘事生成 (Sonnet, 1 call) — 或演算法 fallback"""
        try:
            from src.agents.narrative_agent import generate_narrative
            result = await generate_narrative(
                stock_id=stock_id,
                stock_name=stock_name,
                factor_details=score_result.factor_details,
                total_score=score_result.total_score,
                signal=score_result.signal,
                confidence=score_result.confidence,
                regime=regime,
                ml_scores=ml_scores,
                trust_info=data.trust_info,
                current_price=data.current_price,
                reasoning=score_result.reasoning,
            )
            return NarrativeResult(
                outlook=result.get("outlook", ""),
                outlook_horizon=result.get("outlook_horizon", ""),
                key_drivers=result.get("key_drivers", []),
                risks=result.get("risks", []),
                catalysts=result.get("catalysts", []),
                key_levels=result.get("key_levels", {}),
                position_suggestion=result.get("position_suggestion", ""),
                source="llm",
            )
        except Exception as e:
            logger.warning("LLM narrative failed, using algorithm fallback: %s", e)
            return self._algorithm_narrative(data, score_result)

    def _algorithm_narrative(
        self, data: StockData, score_result: ScoreResult
    ) -> NarrativeResult:
        """演算法 fallback 敘事"""
        # Build outlook from score
        score = score_result.total_score
        if score > 0.7:
            outlook = "短期強烈看多"
        elif score > 0.6:
            outlook = "短期偏多"
        elif score < 0.3:
            outlook = "短期強烈看空"
        elif score < 0.4:
            outlook = "短期偏空"
        else:
            outlook = "短期中性觀望"

        # Key drivers from top 3 weighted factors
        drivers = []
        sorted_factors = sorted(
            [(name, d) for name, d in score_result.factor_details.items()
             if d.get("available")],
            key=lambda x: x[1].get("weight", 0),
            reverse=True,
        )
        for name, detail in sorted_factors[:3]:
            s = detail.get("score", 0.5)
            direction = "偏多" if s > 0.55 else ("偏空" if s < 0.45 else "中性")
            drivers.append(f"{name}: {s:.2f} ({direction})")

        # Risks from low-scoring factors
        risks = []
        for name, detail in score_result.factor_details.items():
            if detail.get("available") and detail.get("score", 0.5) < 0.35:
                risks.append(f"{name} 偏低 ({detail['score']:.2f})")
        if not risks:
            risks = ["目前無明顯風險因子"]

        return NarrativeResult(
            outlook=outlook,
            outlook_horizon="1-2 週",
            key_drivers=drivers,
            risks=risks[:3],
            catalysts=[],
            key_levels={},
            position_suggestion="",
            source="algorithm",
        )

    # ─── Phase 5: Risk Control ───────────────────────────

    def _apply_risk_controls(
        self,
        stock_id: str,
        data: StockData,
        score_result: ScoreResult,
        regime: str,
    ) -> RiskDecision:
        """硬性風控 + 部位建議"""
        rm = self._get_risk_manager()

        signal = score_result.signal
        confidence = score_result.confidence

        # Map signal → action
        if signal in ("strong_buy", "buy"):
            action = "buy"
        elif signal in ("strong_sell", "sell"):
            action = "sell"
        else:
            action = "hold"

        # Position sizing: confidence → 0-20%
        if action == "hold":
            position_size = 0.0
        else:
            position_size = min(confidence * 0.20, 0.20)

        # Meta-label calibration (if available)
        try:
            from src.models.meta_label import MetaLabeler
            meta_path = MODEL_DIR / "meta_labeler.joblib"
            if meta_path.exists():
                meta = MetaLabeler()
                meta.load(str(meta_path))
                proba = meta.predict_proba(
                    pd.DataFrame([{"confidence": confidence, "signal": signal}])
                )
                if proba is not None and len(proba) > 0:
                    position_size *= float(proba[0])
        except Exception:
            pass

        risk_notes = []

        # Circuit breaker check
        if rm.is_circuit_breaker_active():
            if action == "buy":
                action = "hold"
                position_size = 0.0
                risk_notes.append("迴路斷路器啟動，禁止買入")

        # Position limit
        if position_size > rm.max_position_pct:
            position_size = rm.max_position_pct
            risk_notes.append(f"倉位上限 {rm.max_position_pct:.0%}")

        # Regime transition risk
        if regime == "bear" and action == "buy":
            position_size *= 0.5
            risk_notes.append("熊市減碼 50%")
        elif regime == "bear" and action == "hold":
            risk_notes.append("熊市觀望")

        # ATR-based stop loss/take profit
        stop_loss = None
        take_profit = None
        if action == "buy" and not data.df.empty and len(data.df) >= 20:
            try:
                close = data.df["close"].dropna()
                high = data.df["high"].dropna() if "high" in data.df.columns else close
                low = data.df["low"].dropna() if "low" in data.df.columns else close
                # Simple ATR proxy
                tr = pd.concat([
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs(),
                ], axis=1).max(axis=1)
                atr = float(tr.tail(14).mean())
                if atr > 0:
                    stop_loss = round(data.current_price - 2.0 * atr, 2)
                    take_profit = round(data.current_price + 4.0 * atr, 2)
            except Exception:
                pass

        approved = action != "hold" or len(risk_notes) == 0

        return RiskDecision(
            action=action,
            position_size=round(position_size, 4),
            approved=approved,
            risk_notes=risk_notes,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    # ─── Phase 6: Finalize + Save ────────────────────────

    def _build_result(
        self,
        data: StockData,
        score_result: ScoreResult,
        narrative: NarrativeResult,
        risk_decision: RiskDecision,
        regime: str,
    ) -> AnalysisResult:
        """Build the final AnalysisResult"""
        return AnalysisResult(
            stock_id=data.stock_id,
            stock_name=data.stock_name,
            current_price=data.current_price,
            price_change_pct=data.price_change_pct,
            total_score=score_result.total_score,
            signal=risk_decision.action if not risk_decision.approved else score_result.signal,
            confidence=score_result.confidence,
            confidence_breakdown=score_result.confidence_breakdown,
            factor_details=score_result.factor_details,
            regime=regime,
            reasoning="；".join(score_result.reasoning) if score_result.reasoning else "",
            narrative=asdict(narrative),
            risk_decision=asdict(risk_decision),
            analysis_date=date.today().isoformat(),
        )

    async def _save_and_alert(
        self,
        data: StockData,
        score_result: ScoreResult,
        result: AnalysisResult,
    ):
        """Save to DB + generate alerts + record factor IC"""
        logger.info("  ├─ 儲存: MarketScanResult + PipelineResult + FactorIC + Alerts")

        # 1. Save as MarketScanResult (reuse existing DB schema)
        scan_record = {
            "stock_id": data.stock_id,
            "stock_name": data.stock_name,
            "scan_date": date.today(),
            "current_price": data.current_price,
            "price_change_pct": data.price_change_pct,
            "total_score": result.total_score,
            "signal": result.signal,
            "confidence": result.confidence,
            "market_regime": result.regime,
            "factor_details": result.factor_details,
            "reasoning": result.reasoning,
            "score_coverage": {f.name: f.available for f in score_result.factors},
            "effective_coverage": round(sum(
                BASE_WEIGHTS.get(f.name, 0) for f in score_result.factors if f.available
            ), 2),
            **result.confidence_breakdown,
        }

        # Add backward-compatible score fields
        def _fs(name: str) -> float:
            d = result.factor_details.get(name, {})
            return round(d.get("score", 0.5), 4) if d else 0.5

        scan_record.update({
            "technical_score": _fs("technical_signal"),
            "fundamental_score": _fs("institutional_sync"),
            "sentiment_score": _fs("news_sentiment"),
            "ml_score": _fs("ml_ensemble"),
            "momentum_score": _fs("short_momentum"),
            "institutional_flow_score": _fs("foreign_flow"),
            "margin_retail_score": _fs("margin_sentiment"),
            "volatility_score": _fs("volatility_regime"),
            "liquidity_score": _fs("liquidity_quality"),
            "value_quality_score": _fs("fundamental_value"),
        })

        # Add institutional flows
        if data.trust_info:
            scan_record["foreign_net_5d"] = data.trust_info.get("foreign_cumulative", 0)
            scan_record["trust_net_5d"] = data.trust_info.get("trust_cumulative", 0)
            scan_record["dealer_net_5d"] = data.trust_info.get("dealer_cumulative", 0)

        try:
            await asyncio.to_thread(save_market_scan, [scan_record])
            logger.info("  │  ✓ MarketScanResult 已儲存")
        except Exception as e:
            logger.error("  │  ✗ MarketScanResult 儲存失敗: %s", e)

        # 2. Save as PipelineResult (for backward compat with existing pipeline queries)
        pipeline_record = {
            "stock_id": data.stock_id,
            "analysis_date": date.today(),
            "signal": result.signal,
            "confidence": result.confidence,
            "predicted_price": None,
            "reasoning": result.reasoning,
            "agent_scores": {
                "unified_pipeline": {
                    "signal": result.signal,
                    "confidence": round(result.confidence, 2),
                }
            },
            "sentiment_summary": json.dumps(result.narrative, ensure_ascii=False),
            "news_summary": None,
            "technical_data": {k: v.get("score") for k, v in result.factor_details.items()},
            "institutional_data": {
                "foreign_net_5d": data.trust_info.get("foreign_cumulative", 0),
                "trust_net_5d": data.trust_info.get("trust_cumulative", 0),
            } if data.trust_info else None,
            "risk_approved": 1 if asdict(RiskDecision(**result.risk_decision)).get("approved", True) else 0,
            "pipeline_version": "3.0",
        }

        try:
            await asyncio.to_thread(save_pipeline_result_record, pipeline_record)
            logger.info("  │  ✓ PipelineResult 已儲存")
        except Exception as e:
            logger.error("  │  ✗ PipelineResult 儲存失敗: %s", e)

        # 3. Factor IC records
        try:
            ic_records = []
            for f in score_result.factors:
                if f.available and f.raw_value is not None:
                    ic_records.append({
                        "record_date": date.today(),
                        "stock_id": data.stock_id,
                        "factor_name": f.name,
                        "factor_score": f.score,
                    })
            if ic_records:
                await asyncio.to_thread(save_factor_ic_records, ic_records)
                logger.info("  │  ✓ FactorIC 已儲存 (%d 筆)", len(ic_records))
            else:
                logger.info("  │  - FactorIC 無可儲存記錄")
        except Exception as e:
            logger.error("  │  ✗ FactorIC 儲存失敗: %s", e)

        # 4. Alert generation
        try:
            generate_alerts_from_scan([scan_record])
            logger.info("  │  ✓ 警報檢查完成")
        except Exception as e:
            logger.error("Failed to generate alerts: %s", e)


# ═══════════════════════════════════════════════════════
# Module-level convenience function
# ═══════════════════════════════════════════════════════

_service = StockAnalysisService()


async def analyze_stock(
    stock_id: str,
    stock_name: str = "",
) -> AsyncGenerator[str, None]:
    """Module-level convenience wrapper"""
    async for event in _service.analyze_stock(stock_id, stock_name):
        yield event
