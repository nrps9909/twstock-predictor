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
import threading
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from io import StringIO
from typing import AsyncGenerator

import numpy as np
import pandas as pd

from src.utils.constants import STOCK_LIST
from src.utils.config import settings
from src.data.twse_scanner import TWSEScanner
from src.data.stock_fetcher import StockFetcher
from src.db.database import (
    get_stock_prices,
    upsert_stock_prices,
    save_market_scan,
    save_pipeline_result_record,
    save_pipeline_result,
    save_factor_ic_records,
    get_sentiment,
    upsert_data_cache,
    get_data_cache,
    get_data_cache_latest,
)
from src.analysis.technical import TechnicalAnalyzer
from api.services.market_service import (
    BASE_WEIGHTS,
    score_stock,
    _compute_weights,
    _build_reasoning,
    _fetch_global_market_data,
    _fetch_macro_data,
    _compute_sector_aggregates,
)
from api.services.alert_service import generate_alerts_from_scan

logger = logging.getLogger(__name__)

MODEL_DIR = settings.PROJECT_ROOT / "models"

# Per-stock training locks to prevent concurrent training for the same stock
_training_locks: dict[str, threading.Lock] = {}
_training_locks_guard = threading.Lock()

# ── Per-stock daily caches (avoid redundant API calls) ────────────
# Use cachetools TTLCache to prevent unbounded memory growth in long-running servers
try:
    from cachetools import TTLCache

    _trust_cache: dict[str, dict] = TTLCache(maxsize=200, ttl=86400)
    _revenue_cache: dict[str, dict] = TTLCache(maxsize=200, ttl=86400)
    _fundamental_cache: dict[str, dict] = TTLCache(maxsize=200, ttl=86400)
    _per_pbr_cache: dict[str, dict] = TTLCache(maxsize=200, ttl=86400)
except ImportError:
    _trust_cache: dict[str, dict] = {}
    _revenue_cache: dict[str, dict] = {}
    _fundamental_cache: dict[str, dict] = {}
    _per_pbr_cache: dict[str, dict] = {}


# ═══════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════


@dataclass
class StockData:
    """Phase 1 收集的原始數據"""

    stock_id: str
    stock_name: str
    df: pd.DataFrame  # OHLCV + institutional flows
    df_tech: pd.DataFrame  # Technical indicators
    signals: dict  # Technical signals summary
    trust_info: dict  # {cumulative, consecutive_days} per fund type
    revenue_df: pd.DataFrame | None  # Monthly revenue
    global_data: dict | None  # SOX/TSM/EWT
    macro_data: dict | None  # VIX/FX/TNX/XLI
    sentiment_df: pd.DataFrame | None  # News/PTT sentiment records
    sentiment_scores: dict  # {stock_id: avg_score}
    fundamental_data: dict | None = None  # yfinance info + quarterly_income_stmt
    per_pbr_df: pd.DataFrame | None = None  # FinMind 每日 P/E, P/B
    current_price: float = 0.0
    price_change_pct: float = 0.0


@dataclass
class ScoreResult:
    """Phase 3 評分結果"""

    total_score: float
    signal: str  # strong_buy/buy/hold/sell/strong_sell
    confidence: float
    confidence_breakdown: dict
    factor_details: dict  # 20 factors full transparency
    factors: list  # FactorResult list (internal)
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
    # Opus verdict (Phase 5.5)
    verdict: str = ""
    verdict_short: str = ""
    risk_warning: str = ""
    confidence_comment: str = ""


@dataclass
class RiskDecision:
    """Phase 5 風控結果"""

    action: str  # buy/sell/hold
    position_size: float  # 0.0-0.20
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


def _sse_event(
    phase: str, status: str, progress: int, message: str = "", data: dict | None = None
) -> str:
    """Format SSE event for streaming"""
    payload = {
        "phase": phase,
        "status": status,
        "progress": progress,
        "message": message,
    }
    if data is not None:
        payload["data"] = data
    return f"data: {json.dumps(payload, ensure_ascii=False, default=_nan_safe_default)}\n\n"


def _nan_safe_default(obj):
    """JSON default handler that converts NaN/Inf to None, other types to str."""
    import math as _math

    if isinstance(obj, float) and (_math.isnan(obj) or _math.isinf(obj)):
        return None
    return str(obj)


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
        yield _sse_event(
            "data_collection", "running", 5, f"收集 {stock_id} {stock_name} 數據..."
        )
        try:
            data = await self._collect_data(stock_id, stock_name)
        except Exception as e:
            logger.error("[Phase 1/6] 數據收集 — 失敗: %s", e, exc_info=True)
            yield _sse_event("data_collection", "error", 5, f"數據收集失敗: {e}")
            yield _sse_event(
                "complete",
                "error",
                100,
                "分析失敗",
                {
                    "stock_id": stock_id,
                    "error": str(e),
                },
            )
            return

        if data.df.empty or len(data.df) < 20:
            logger.warning("[Phase 1/6] 數據收集 — 資料不足 (rows=%d)", len(data.df))
            yield _sse_event(
                "complete",
                "error",
                100,
                "價格資料不足",
                {
                    "stock_id": stock_id,
                    "error": "價格資料不足，無法分析",
                },
            )
            return

        logger.info(
            "[Phase 1/6] 數據收集 — 完成 (%.1fs) | 價格=%d筆 技術=%d筆 現價=%.1f 漲跌=%.2f%% 法人=%s 營收=%s 情緒=%s",
            _time.perf_counter() - t0,
            len(data.df),
            len(data.df_tech),
            data.current_price,
            data.price_change_pct,
            "有" if data.trust_info else "無",
            "有" if data.revenue_df is not None else "無",
            "有" if data.sentiment_df is not None else "無",
        )
        # Build sub_steps for Phase 1 detail
        p1_steps = []
        p1_steps.append(
            f"OHLCV 日線: {len(data.df)} 筆 ({len(data.df_tech)} 含技術指標)"
        )
        if data.trust_info:
            foreign_cum = data.trust_info.get("foreign_cumulative", 0)
            trust_cum = data.trust_info.get("trust_cumulative", 0)
            p1_steps.append(
                f"三大法人 5 日: 外資 {foreign_cum:+,.0f} / 投信 {trust_cum:+,.0f}"
            )
        else:
            p1_steps.append("三大法人: 無資料")
        if data.revenue_df is not None:
            p1_steps.append(f"月營收: {len(data.revenue_df)} 筆紀錄")
        if data.sentiment_df is not None and not data.sentiment_df.empty:
            sent_count = len(data.sentiment_df)
            sources = (
                data.sentiment_df["source"].value_counts().to_dict()
                if "source" in data.sentiment_df.columns
                else {}
            )
            src_parts = [f"{k} {v}" for k, v in sources.items()]
            p1_steps.append(
                f"情緒文章: {sent_count} 篇 ({', '.join(src_parts) if src_parts else '混合來源'})"
            )
        if data.global_data:
            sox = data.global_data.get("sox_return", 0)
            p1_steps.append(f"全球: SOX {sox:+.1%}" if sox else "全球市場已取得")
        if data.macro_data:
            vix = data.macro_data.get("vix", 0)
            p1_steps.append(f"宏觀: VIX {vix:.1f}" if vix else "宏觀指標已取得")
        if data.fundamental_data:
            p1_steps.append("yfinance 基本面 + 季報已取得")
        if data.per_pbr_df is not None and not data.per_pbr_df.empty:
            p1_steps.append(f"P/E P/B: {len(data.per_pbr_df)} 筆日線")
        tech_summary = data.signals.get("summary", {})
        if tech_summary:
            p1_steps.append(
                f"技術訊號: {tech_summary.get('signal', '?')} ({tech_summary.get('raw_score', '?')}/{tech_summary.get('max_score', '?')})"
            )

        yield _sse_event(
            "data_collection",
            "done",
            15,
            "數據收集完成",
            {
                "price_rows": len(data.df),
                "current_price": round(data.current_price, 2),
                "price_change_pct": round(data.price_change_pct, 2),
                "has_trust": bool(data.trust_info),
                "has_revenue": data.revenue_df is not None,
                "has_sentiment": data.sentiment_df is not None,
                "has_global": bool(data.global_data),
                "has_macro": bool(data.macro_data),
                "has_fundamental": data.fundamental_data is not None,
                "stock_name": data.stock_name,
                "sub_steps": p1_steps,
            },
        )

        # Phase 2: 特徵萃取
        logger.info("[Phase 2/6] 特徵萃取 — 開始 (HMM + ML + LLM 情緒)")
        t0 = _time.perf_counter()
        yield _sse_event(
            "feature_extraction", "running", 20, "計算 20 因子 + HMM + ML..."
        )
        regime, ml_scores = await self._extract_features(data)
        logger.info(
            "[Phase 2/6] 特徵萃取 — 完成 (%.1fs) | regime=%s ml_scores=%s",
            _time.perf_counter() - t0,
            regime,
            ml_scores or "無模型",
        )
        p2_steps = [f"HMM 3-state 體制辨識: {regime}"]
        if ml_scores:
            lstm_path = MODEL_DIR / f"{data.stock_id}_lstm.pt"
            ml_mode = "已載入模型" if lstm_path.exists() else "即時訓練"
            for sid, sc in ml_scores.items():
                p2_steps.append(f"ML 集成 ({sid}): 分數 {sc:.2f} ({ml_mode})")
        else:
            p2_steps.append("ML 模型: 資料不足，跳過")
        p2_steps.append("LLM 情緒增強: Haiku 分析新聞標題+內文 → 結構化情緒")
        yield _sse_event(
            "feature_extraction",
            "done",
            40,
            "特徵萃取完成",
            {
                "regime": regime,
                "has_ml": ml_scores is not None,
                "sub_steps": p2_steps,
            },
        )

        # Phase 3: 多因子評分
        logger.info("[Phase 3/6] 多因子評分 — 開始")
        t0 = _time.perf_counter()
        yield _sse_event("scoring", "running", 45, "多因子加權評分...")
        score_result = self._score(data, regime, ml_scores)
        avail_count = sum(1 for f in score_result.factors if f.available)
        logger.info(
            "[Phase 3/6] 多因子評分 — 完成 (%.1fs) | 總分=%.3f 訊號=%s 信心=%.2f 可用因子=%d/%d regime=%s",
            _time.perf_counter() - t0,
            score_result.total_score,
            score_result.signal,
            score_result.confidence,
            avail_count,
            len(score_result.factors),
            regime,
        )
        # Build top/bottom factors for sub_steps
        p3_steps = [f"可用因子: {avail_count}/{len(score_result.factors)} (regime={regime})"]
        sorted_factors = sorted(
            [f for f in score_result.factors if f.available],
            key=lambda f: f.score,
            reverse=True,
        )
        if sorted_factors:
            top3 = sorted_factors[:3]
            bot3 = sorted_factors[-3:]
            p3_steps.append(
                "最強因子: " + " / ".join(f"{f.name} {f.score:.2f}" for f in top3)
            )
            p3_steps.append(
                "最弱因子: " + " / ".join(f"{f.name} {f.score:.2f}" for f in bot3)
            )
        p3_steps.append(
            f"加權總分: {score_result.total_score:.3f} → 訊號 {score_result.signal}"
        )
        p3_steps.append(
            f"信心度: {score_result.confidence:.1%} (agreement/strength/coverage/freshness × risk_discount)"
        )
        yield _sse_event(
            "scoring",
            "done",
            55,
            f"評分完成: {score_result.total_score:.2f} ({score_result.signal})",
            {
                "total_score": round(score_result.total_score, 3),
                "signal": score_result.signal,
                "confidence": round(score_result.confidence, 3),
                "available_factors": avail_count,
                "regime": regime,
                "sub_steps": p3_steps,
            },
        )

        # Phase 4: 風險控制 (MOVED BEFORE narrative so risk results inform LLM)
        logger.info("[Phase 4/6] 風險控制 — 開始")
        t0 = _time.perf_counter()
        yield _sse_event("risk_control", "running", 60, "風險檢查...")
        risk_decision = self._apply_risk_controls(stock_id, data, score_result, regime)
        logger.info(
            "[Phase 4/6] 風險控制 — 完成 (%.1fs) | action=%s position=%.2f%% approved=%s stop=%.1f profit=%.1f notes=%s",
            _time.perf_counter() - t0,
            risk_decision.action,
            risk_decision.position_size * 100,
            risk_decision.approved,
            risk_decision.stop_loss or 0,
            risk_decision.take_profit or 0,
            risk_decision.risk_notes or "無",
        )
        p4_steps = [f"訊號 {score_result.signal} → 動作 {risk_decision.action}"]
        p4_steps.append(
            f"部位大小: {risk_decision.position_size * 100:.1f}% (信心 × 20% 上限)"
        )
        if risk_decision.stop_loss:
            p4_steps.append(f"ATR 停損: ${risk_decision.stop_loss:.1f}")
        if risk_decision.take_profit:
            p4_steps.append(f"ATR 停利: ${risk_decision.take_profit:.1f}")
        for note in risk_decision.risk_notes[:3]:
            p4_steps.append(f"風控: {note}")
        if not risk_decision.risk_notes:
            p4_steps.append("風控檢查: 無異常")
        p4_steps.append(f"最終核准: {'通過' if risk_decision.approved else '拒絕'}")
        yield _sse_event(
            "risk_control",
            "done",
            68,
            f"風控完成: {risk_decision.action} (approved={risk_decision.approved})",
            {
                "action": risk_decision.action,
                "approved": risk_decision.approved,
                "position_size": round(risk_decision.position_size * 100, 1),
                "stop_loss": round(risk_decision.stop_loss, 2)
                if risk_decision.stop_loss
                else None,
                "take_profit": round(risk_decision.take_profit, 2)
                if risk_decision.take_profit
                else None,
                "risk_notes": risk_decision.risk_notes[:3]
                if risk_decision.risk_notes
                else [],
                "sub_steps": p4_steps,
            },
        )

        # Phase 5: Opus 分析報告 + 投資結論 (AFTER risk so LLM sees risk-adjusted position)
        logger.info("[Phase 5/6] Opus 分析報告 — 開始")
        t0 = _time.perf_counter()
        yield _sse_event("narrative", "running", 72, "Opus 生成分析報告與投資結論...")
        narrative = await self._generate_narrative(
            stock_id,
            stock_name,
            data,
            score_result,
            regime,
            ml_scores,
            risk_decision=risk_decision,
        )
        logger.info(
            "[Phase 5/6] Opus 分析報告 — 完成 (%.1fs) | source=%s verdict=%s",
            _time.perf_counter() - t0,
            narrative.source,
            narrative.verdict_short[:40] if narrative.verdict_short else "(no verdict)",
        )
        p5_steps = []
        if narrative.source == "llm":
            p5_steps.append("Opus 4.6 生成完整報告 (展望/驅動/風險/催化劑/投資結論)")
        else:
            p5_steps.append("演算法 fallback (LLM 不可用)")
        if narrative.verdict_short:
            p5_steps.append(f"結論: {narrative.verdict_short}")
        if narrative.outlook:
            p5_steps.append(f"展望: {narrative.outlook[:80]}")
        if narrative.key_drivers:
            p5_steps.append(
                f"驅動因素 ({len(narrative.key_drivers)}): "
                + " / ".join(narrative.key_drivers[:3])
            )
        if narrative.risks:
            p5_steps.append(
                f"風險因素 ({len(narrative.risks)}): " + " / ".join(narrative.risks[:3])
            )
        if narrative.confidence_comment:
            p5_steps.append(f"信心解讀: {narrative.confidence_comment}")
        yield _sse_event(
            "narrative",
            "done",
            88,
            f"分析完成 (source={narrative.source})",
            {
                "source": narrative.source,
                "outlook": narrative.outlook[:60] if narrative.outlook else "",
                "verdict_short": narrative.verdict_short,
                "has_verdict": bool(narrative.verdict),
                "key_drivers_count": len(narrative.key_drivers),
                "risks_count": len(narrative.risks),
                "sub_steps": p5_steps,
            },
        )

        # Phase 6: 儲存 + 警報
        logger.info("[Phase 6/6] 儲存結果 — 開始")
        t0 = _time.perf_counter()
        yield _sse_event("finalize", "running", 90, "儲存結果...")
        result = self._build_result(
            data, score_result, narrative, risk_decision, regime
        )
        await self._save_and_alert(data, score_result, result)
        logger.info("[Phase 6/6] 儲存結果 — 完成 (%.1fs)", _time.perf_counter() - t0)
        yield _sse_event(
            "finalize",
            "done",
            95,
            "儲存完成",
            {
                "saved": True,
                "sub_steps": [
                    "MarketScanResult 已寫入 (含 15 因子詳情)",
                    "PipelineResult 已寫入 (向後相容)",
                    "Prediction + TradeJournal 已寫入",
                    f"FactorIC 記錄: {avail_count} 筆因子分數",
                    "警報系統: 已檢查訊號變化/強訊號/法人異動",
                ],
            },
        )

        # Complete
        elapsed = _time.perf_counter() - t_total
        logger.info(
            "═══ 分析完成 %s %s ═══ 總耗時 %.1fs | 總分=%.3f 訊號=%s 信心=%.2f regime=%s action=%s approved=%s",
            stock_id,
            stock_name,
            elapsed,
            result.total_score,
            result.signal,
            result.confidence,
            result.regime,
            risk_decision.action,
            risk_decision.approved,
        )
        yield _sse_event("complete", "done", 100, "分析完成", asdict(result))

    # ─── Phase 1: Data Collection ────────────────────────

    async def _collect_data(self, stock_id: str, stock_name: str) -> StockData:
        """並行收集所有數據源"""
        logger.info(
            "  ├─ 並行抓取: 價格 / 法人 / 營收 / 全球 / 宏觀 / 情緒 / 基本面 / P/E ..."
        )

        # Parallel fetches
        (
            df,
            trust_info,
            revenue_df,
            global_data,
            macro_data,
            sentiment_result,
            fundamental_data,
            per_pbr_df,
        ) = await asyncio.gather(
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
            fetch_status.append(
                f"P/E:{'✓' if per_pbr_df is not None and not per_pbr_df.empty else '空'}"
            )

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
                logger.info(
                    "  ├─ 技術分析完成: signal=%s score=%s/%s",
                    summary.get("signal", "?"),
                    summary.get("raw_score", "?"),
                    summary.get("max_score", "?"),
                )
            except Exception as e:
                logger.warning("  ├─ 技術分析失敗: %s", e)

        # Current price
        current_price = 0.0
        price_change_pct = 0.0
        if not df.empty:
            current_price = (
                float(df.iloc[-1]["close"]) if pd.notna(df.iloc[-1]["close"]) else 0.0
            )
            if len(df) >= 2:
                prev_close = (
                    float(df.iloc[-2]["close"])
                    if pd.notna(df.iloc[-2]["close"])
                    else current_price
                )
                if prev_close > 0:
                    price_change_pct = round(
                        (current_price - prev_close) / prev_close * 100, 2
                    )

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
        """Fetch price data: DB-first, incremental API fetch for missing days only."""
        df = await asyncio.to_thread(get_stock_prices, stock_id)
        today = date.today()

        if df.empty or len(df) < 60:
            # No data or too little → full fetch (5 years)
            logger.info("DB 無資料或不足 (%d筆), 全量抓取 %s", len(df), stock_id)
            try:
                fetcher = StockFetcher()
                start_date = today - timedelta(days=3000)
                new_df = await asyncio.to_thread(
                    fetcher.fetch_all,
                    stock_id,
                    start_date.isoformat(),
                    today.isoformat(),
                )
                if not new_df.empty:
                    await asyncio.to_thread(upsert_stock_prices, new_df, stock_id)
                    df = await asyncio.to_thread(get_stock_prices, stock_id)
            except Exception as e:
                logger.warning("External fetch failed for %s: %s", stock_id, e)
        else:
            # DB has data — check freshness, only fetch the gap
            last_date = df["date"].max()
            # Allow 3-day gap for weekends/holidays before fetching
            if (today - last_date).days > 3:
                logger.info(
                    "DB 最新日期=%s, 增量抓取 %s → %s", last_date, stock_id, today
                )
                try:
                    fetcher = StockFetcher()
                    fetch_start = last_date - timedelta(
                        days=1
                    )  # overlap 1 day for safety
                    new_df = await asyncio.to_thread(
                        fetcher.fetch_all,
                        stock_id,
                        fetch_start.isoformat(),
                        today.isoformat(),
                    )
                    if not new_df.empty:
                        await asyncio.to_thread(upsert_stock_prices, new_df, stock_id)
                        df = await asyncio.to_thread(get_stock_prices, stock_id)
                except Exception as e:
                    logger.warning("Incremental fetch failed for %s: %s", stock_id, e)
            else:
                logger.info("DB 已是最新 (%s, %d筆), 跳過 API 抓取", last_date, len(df))
        return df

    async def _fetch_trust_info(self, stock_id: str) -> dict:
        """Fetch institutional trust info — DB-first, TWSE T86 as fallback.

        DB StockPrice already has foreign_buy_sell/trust_buy_sell/dealer_buy_sell
        from FinMind fetch. Compute trust_info from DB to avoid TWSE API calls.
        """
        today = date.today()
        cached = _trust_cache.get(stock_id)
        if cached and cached["date"] == today:
            logger.debug("Trust info cache hit: %s", stock_id)
            return cached["data"]

        # Try computing from DB first
        data = await asyncio.to_thread(self._compute_trust_info_from_db, stock_id)
        if data:
            logger.info(
                "Trust info from DB for %s: foreign=%+.0f trust=%+.0f",
                stock_id,
                data.get("foreign_cumulative", 0),
                data.get("trust_cumulative", 0),
            )
            _trust_cache[stock_id] = {"date": today, "data": data}
            return data

        # Fallback: TWSE T86 API
        try:
            logger.info("Trust info DB 不足，fallback TWSE T86: %s", stock_id)
            scanner = TWSEScanner()
            result = await asyncio.to_thread(scanner.get_trust_info, stock_id, days=5)
            data = result or {}
            _trust_cache[stock_id] = {"date": today, "data": data}
            return data
        except Exception:
            return {}

    @staticmethod
    def _compute_trust_info_from_db(stock_id: str, days: int = 5) -> dict:
        """Compute trust_info from StockPrice DB records (no API call)."""
        start = date.today() - timedelta(days=days * 2 + 5)
        df = get_stock_prices(stock_id, start_date=start)
        if df.empty:
            return {}

        # Need recent data with institutional columns
        inst_cols = ["foreign_buy_sell", "trust_buy_sell", "dealer_buy_sell"]
        for col in inst_cols:
            if col not in df.columns:
                return {}

        recent = df.dropna(subset=["foreign_buy_sell"]).tail(days)
        if len(recent) < 3:  # need at least 3 days of data
            return {}

        # FinMind stores in shares; convert to lots (張) to match TWSE T86 path
        foreign_cum = float(recent["foreign_buy_sell"].sum()) / 1000
        trust_cum = float(recent["trust_buy_sell"].sum()) / 1000
        dealer_cum = float(recent["dealer_buy_sell"].sum()) / 1000

        # Consecutive days (from latest backward): positive=buy streak, negative=sell streak
        def _consecutive_signed(series: pd.Series) -> int:
            count = 0
            for val in series.iloc[::-1]:
                if pd.notna(val) and val > 0:
                    if count < 0:
                        break
                    count += 1
                elif pd.notna(val) and val < 0:
                    if count > 0:
                        break
                    count -= 1
                else:
                    break
            return count

        return {
            "stock_id": stock_id,
            "foreign_cumulative": foreign_cum,
            "trust_cumulative": trust_cum,
            "dealer_cumulative": dealer_cum,
            "total_cumulative": foreign_cum + trust_cum + dealer_cum,
            "trade_days": len(recent),
            "foreign_consecutive_days": _consecutive_signed(
                recent["foreign_buy_sell"].dropna()
            ),
            "trust_consecutive_days": _consecutive_signed(
                recent["trust_buy_sell"].dropna()
            ),
            "sync_buy": foreign_cum > 0 and trust_cum > 0,
        }

    async def _fetch_revenue(self, stock_id: str) -> pd.DataFrame | None:
        """Fetch monthly revenue from FinMind — DB-first + daily cached"""
        today = date.today()
        cached = _revenue_cache.get(stock_id)
        if cached and cached["date"] == today:
            logger.debug("Revenue cache hit: %s", stock_id)
            return cached["data"]

        # DB-first
        cache_key = f"revenue:{stock_id}"
        cached_json = get_data_cache(cache_key, today)
        if cached_json:
            try:
                df = pd.read_json(StringIO(cached_json), orient="records")
                if not df.empty:
                    _revenue_cache[stock_id] = {"date": today, "data": df}
                    logger.debug("Revenue DB cache hit: %s", stock_id)
                    return df
            except Exception:
                pass

        try:
            from api.services.market_service import _fetch_revenue_batch

            result = await _fetch_revenue_batch([stock_id])
            data = result.get(stock_id)
            _revenue_cache[stock_id] = {"date": today, "data": data}
            return data
        except Exception:
            return None

    async def _fetch_sentiment(self, stock_id: str) -> tuple[pd.DataFrame | None, dict]:
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
                    # 可信度加權平均（如有 credibility 欄位）
                    if "credibility" in sent_df.columns:
                        weights = sent_df.loc[scores.index, "credibility"].fillna(0.5)
                        weight_sum = weights.sum()
                        if weight_sum > 0:
                            avg = float((scores * weights).sum() / weight_sum)
                        else:
                            avg = float(scores.mean())
                    else:
                        avg = float(scores.mean())
                    sentiment_scores[stock_id] = (avg + 1) / 2  # Normalize to 0-1
            return sent_df, sentiment_scores
        except Exception as e:
            logger.warning("Sentiment fetch failed: %s", e)
            return None, {}

    async def _crawl_sentiment_live(self, stock_id: str) -> pd.DataFrame | None:
        """Live-crawl sentiment from PTT, 鉅亨, Google News, Yahoo TW

        流程: 爬取 → 內文充實 → LLM 分析(標題+內文) → 情緒+可信度 → 加權分數
        """
        try:
            from src.data.sentiment_crawler import SentimentCrawler
            from src.db.database import insert_sentiment

            crawler = SentimentCrawler()
            articles = await asyncio.to_thread(crawler.crawl_all, stock_id)
            if not articles:
                logger.info("  │  情緒爬蟲: 0 篇文章 (stock=%s)", stock_id)
                return None

            has_content = sum(1 for a in articles if a.get("content_summary"))
            logger.info(
                "  │  情緒爬蟲: %d 篇文章 (%d 有內文) (stock=%s)",
                len(articles),
                has_content,
                stock_id,
            )

            # LLM 分析: 標題 + 內文 → 情緒 + 可信度
            to_score = [
                art
                for art in articles
                if art.get("sentiment_label") == "neutral" and art.get("title")
            ]
            llm_results = {}
            if to_score:
                llm_results = await self._batch_score_articles(stock_id, to_score)

            # Assign sentiment_score with credibility weighting
            label_to_score = {"bullish": 0.8, "bearish": -0.8, "neutral": 0.0}
            records = []
            for art in articles:
                title = art.get("title", "")
                # Override neutral labels with LLM results
                if title in llm_results:
                    result = llm_results[title]
                    art["sentiment_label"] = result["sentiment"]
                    art["credibility"] = result["credibility"]

                art.setdefault("sentiment_label", "neutral")
                art.setdefault("credibility", 0.5)
                base_score = label_to_score.get(art["sentiment_label"], 0.0)
                # 可信度加權: 農場文 credibility 低 → 情緒分數被壓低
                art["sentiment_score"] = base_score * art["credibility"]
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

    async def _batch_score_articles(
        self, stock_id: str, articles: list[dict]
    ) -> dict[str, dict]:
        """LLM 分析新聞: 標題 + 內文 → 情緒 + 可信度（過濾農場標題）"""
        try:
            from src.utils.llm_client import call_claude, parse_json_response
            from src.utils.constants import STOCK_LIST

            stock_name = STOCK_LIST.get(stock_id, stock_id)

            # 格式化: 標題 + 內文摘要
            lines = []
            scored_articles = articles[:25]
            for i, art in enumerate(scored_articles):
                title = art.get("title", "")
                content = art.get("content_summary", "")
                if content:
                    lines.append(f"{i + 1}. [標題] {title}\n   [內文] {content[:300]}")
                else:
                    lines.append(f"{i + 1}. [標題] {title}\n   [內文] (無)")

            numbered = "\n\n".join(lines)
            prompt = f"""分析以下與 {stock_id} ({stock_name}) 相關的新聞。每則有標題和內文摘要。

{numbered}

回傳 JSON array:
[{{"idx": 1, "sentiment": "bullish", "credibility": 0.8}}, ...]

sentiment: bullish（利多）/ bearish（利空）/ neutral（中性）
credibility: 0.0~1.0 可信度:
  - 農場標題、標題黨、業配文、與該股票無直接關聯 → 0.1~0.3
  - 標題聳動但內文空泛或無實質內容 → 0.3~0.5
  - 一般新聞報導、有基本事實 → 0.5~0.7
  - 專業財經分析、有具體數據/財報/法人動向 → 0.7~1.0

重要規則:
- 有內文時，sentiment 必須基於內文判斷，不要只看標題
- 標題與內文情緒不符 → 以內文為準，credibility 降低
- 無內文的新聞 → credibility 上限 0.5
- 利多(營收成長/獲利/漲價/需求增加)=bullish
- 利空(裁員/虧損/下修/賣超)=bearish
只回傳 JSON。"""

            text = await call_claude(
                prompt, model="claude-haiku-4-5-20251001", timeout=90
            )
            results = parse_json_response(text)
            if isinstance(results, list):
                mapping = {}
                for item in results:
                    idx = item.get("idx", 0) - 1
                    if 0 <= idx < len(scored_articles):
                        title = scored_articles[idx].get("title", "")
                        cred = item.get("credibility", 0.5)
                        mapping[title] = {
                            "sentiment": item.get("sentiment", "neutral"),
                            "credibility": min(1.0, max(0.0, float(cred))),
                        }
                logger.info(
                    "  │  LLM 文章分析: %d/%d 完成 (含內文+可信度)",
                    len(mapping),
                    len(scored_articles),
                )
                return mapping
        except Exception as e:
            logger.warning("  │  LLM 文章分析失敗: %s", e)
        return {}

    async def _fetch_fundamental(self, stock_id: str) -> dict | None:
        """Fetch yfinance fundamental data — DB-first + daily cached

        Note: quarterly_income_stmt (DataFrame) cannot be JSON-serialized,
        so only the 'info' dict is persisted to DB cache. The full object
        (with DataFrame) is kept in the in-memory cache for the current day.
        """
        today = date.today()
        cached = _fundamental_cache.get(stock_id)
        if cached and cached["date"] == today:
            logger.debug("Fundamental cache hit: %s", stock_id)
            return cached["data"]

        # DB-first: restore 'info' dict from persistent cache
        cache_key = f"fundamental:{stock_id}"
        cached_json = get_data_cache(cache_key, today)
        if cached_json:
            try:
                import json as _json

                info_data = _json.loads(cached_json)
                if info_data:
                    result = {"info": info_data}
                    _fundamental_cache[stock_id] = {"date": today, "data": result}
                    logger.debug("Fundamental DB cache hit: %s", stock_id)
                    return result
            except Exception:
                pass

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
                    logger.warning(
                        "yfinance %s.TW quarterly_income_stmt fetch failed: %s",
                        stock_id,
                        e,
                    )
                return result if result else None
            except Exception as e:
                logger.warning(
                    "yfinance %s.TW fundamental fetch failed: %s", stock_id, e
                )
                return None

        data = await asyncio.to_thread(_do_fetch)
        _fundamental_cache[stock_id] = {"date": today, "data": data}

        # Persist 'info' to DB (DataFrame not serializable)
        if data and "info" in data:
            try:
                import json as _json

                # Filter out non-serializable values
                safe_info = {
                    k: v
                    for k, v in data["info"].items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                }
                upsert_data_cache(cache_key, today, _json.dumps(safe_info))
            except Exception:
                pass

        return data

    async def _fetch_per_pbr(self, stock_id: str) -> pd.DataFrame | None:
        """Fetch FinMind daily P/E, P/B, dividend_yield — DB-first + daily cached"""
        today = date.today()
        cached = _per_pbr_cache.get(stock_id)
        if cached and cached["date"] == today:
            logger.debug("P/E P/B cache hit: %s", stock_id)
            return cached["data"]

        # DB-first
        cache_key = f"per_pbr:{stock_id}"
        cached_json = get_data_cache(cache_key, today)
        if cached_json:
            try:
                df = pd.read_json(StringIO(cached_json), orient="records")
                if not df.empty:
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"]).dt.date
                    for c in ["PER", "PBR", "dividend_yield"]:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                    _per_pbr_cache[stock_id] = {"date": today, "data": df}
                    logger.debug("P/E P/B DB cache hit: %s", stock_id)
                    return df
            except Exception:
                pass

        try:
            fetcher = StockFetcher()
            start_date = today - timedelta(days=90)
            df = await asyncio.to_thread(
                fetcher.fetch_per_pbr,
                stock_id,
                start_date.isoformat(),
                today.isoformat(),
            )
            data = df if df is not None and not df.empty else None
            _per_pbr_cache[stock_id] = {"date": today, "data": data}

            # Save to DB
            if data is not None and not data.empty:
                try:
                    upsert_data_cache(
                        cache_key,
                        today,
                        data.to_json(orient="records", date_format="iso"),
                    )
                except Exception:
                    pass

            return data
        except Exception as e:
            logger.warning("FinMind P/E P/B fetch failed for %s: %s", stock_id, e)
            return None

    # ─── Phase 2: Feature Extraction ─────────────────────

    async def _extract_features(self, data: StockData) -> tuple[str, dict]:
        """並行提取: HMM 體制 + ML 預測 + LLM 情緒增強

        20 因子在 Phase 3 計算 (同步，由 score_stock 處理)。
        Returns: (regime, ml_scores)
        """
        regime_task = asyncio.to_thread(self._detect_regime, data.df)
        ml_task = asyncio.to_thread(self._predict_ml, data.stock_id, data.df)
        sentiment_task = self._enhance_sentiment(data)

        regime, ml_scores, enhanced_sentiment = await asyncio.gather(
            regime_task,
            ml_task,
            sentiment_task,
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
            regime = state.state_name
            # MA trend override: prevent bear when price above major MAs
            if regime == "bear" and len(close) >= 60:
                current_price = close.iloc[-1]
                ma20 = close.rolling(20).mean().iloc[-1]
                ma60 = close.rolling(60).mean().iloc[-1]
                if current_price > ma20 and current_price > ma60:
                    logger.info(
                        "HMM regime override: bear → sideways "
                        "(price %.1f > MA20 %.1f & MA60 %.1f)",
                        current_price,
                        ma20,
                        ma60,
                    )
                    regime = "sideways"
            return regime
        except Exception as e:
            logger.warning("HMM detection failed: %s", e)
            return "sideways"

    def _predict_ml(self, stock_id: str, df: pd.DataFrame) -> dict:
        """ML model prediction — always retrain for freshest signal"""
        ml_scores = {}
        try:
            trained = self._train_ml_quality(stock_id)
            if trained:
                ml_scores.update(trained)
        except Exception as e:
            logger.warning("ML predict failed for %s: %s", stock_id, e)
        return ml_scores

    def _ensure_training_data(self, stock_id: str, min_rows: int = 1000) -> int:
        """Ensure DB has enough trading days. Fetch from FinMind if needed."""
        df = get_stock_prices(stock_id)
        if len(df) >= min_rows:
            return len(df)
        try:
            fetcher = StockFetcher()
            end = date.today()
            start = end - timedelta(days=3000)
            full_df = fetcher.fetch_all(stock_id, start.isoformat(), end.isoformat())
            if not full_df.empty:
                upsert_stock_prices(full_df, stock_id)
                return max(len(df), len(full_df))
        except Exception as e:
            logger.warning("_ensure_training_data fetch failed for %s: %s", stock_id, e)
        return len(df)

    def _train_ml_quality(self, stock_id: str) -> dict:
        """On-demand quality ML training (epochs=100, 5-year data, quality gate).

        Uses per-stock lock to prevent concurrent training for the same stock.
        Returns {stock_id: score} on success, empty dict on failure.
        """
        with _training_locks_guard:
            if stock_id not in _training_locks:
                _training_locks[stock_id] = threading.Lock()
            lock = _training_locks[stock_id]

        if not lock.acquire(timeout=600):
            logger.warning("ML quality training lock timeout for %s", stock_id)
            return {}

        try:
            # Always retrain for freshest signal
            # Ensure we have enough data (5 years)
            n_rows = self._ensure_training_data(stock_id)
            logger.info(
                "ML quality training started for %s (%d rows in DB)", stock_id, n_rows
            )

            from src.models.trainer import ModelTrainer

            trainer = ModelTrainer(stock_id)
            today = date.today()
            start = today - timedelta(days=3000)

            # Adaptive max_features: ~30 samples per feature to avoid overfitting
            n_train_est = int(n_rows * 0.55)  # after 3-way split + purge
            adaptive_max_features = min(30, max(8, n_train_est // 30))
            logger.info(
                "ML adaptive features: %d rows → %d max_features",
                n_rows, adaptive_max_features,
            )

            # Try sector-pooled training first (10-20x more data)
            from api.services.market_service import STOCK_SECTOR
            sector = STOCK_SECTOR.get(stock_id)
            if sector:
                logger.info(
                    "Attempting sector training for %s (sector=%s)",
                    stock_id, sector,
                )
                train_result = trainer.train_sector(
                    sector=sector,
                    start_date=start.isoformat(),
                    end_date=today.isoformat(),
                    epochs=100,
                    max_features=adaptive_max_features,
                )
            else:
                train_result = trainer.train(
                    start_date=start.isoformat(),
                    end_date=today.isoformat(),
                    epochs=100,
                    seq_len=40,
                    val_ratio=0.2,
                    test_ratio=0.2,
                    max_features=adaptive_max_features,
                    use_chronos=True,
                )

            # Check quality gate result
            gate = train_result.get("quality_gate", {})
            if not gate.get("overall_passed", False):
                logger.warning(
                    "ML quality training for %s: quality gate FAILED (lstm_dir=%.3f, xgb_dir=%.3f, cls_score_dir=%.3f)",
                    stock_id,
                    gate.get("lstm_direction_acc", 0),
                    gate.get("xgb_direction_acc", 0),
                    gate.get("xgb_cls_score_dir_acc", 0),
                )

                # Fallback: pooled classifier training (if sector training failed)
                if not sector:
                    from api.services.market_service import STOCK_SECTOR
                    sector = STOCK_SECTOR.get(stock_id)
                if sector and not gate.get("pooled_training"):
                    peer_ids = [
                        sid for sid, sec in STOCK_SECTOR.items()
                        if sec == sector and sid != stock_id
                    ][:12]  # cap at 12 peers
                    if len(peer_ids) >= 3:
                        logger.info(
                            "Attempting pooled training for %s with %d %s peers",
                            stock_id, len(peer_ids), sector,
                        )
                        pooled_result = trainer.train_pooled_classifier(
                            peer_ids,
                            start_date=start.isoformat(),
                            end_date=today.isoformat(),
                            max_features=adaptive_max_features,
                        )
                        if pooled_result and pooled_result.get("passed_quality"):
                            gate["xgb_cls_passed"] = True
                            gate["overall_passed"] = True
                            gate["pooled_training"] = True
                            logger.info(
                                "Pooled training succeeded for %s: dir=%.3f",
                                stock_id,
                                pooled_result.get("test_direction_acc", 0),
                            )
                # (the prediction code below handles the dampened output)

            # Predict with freshly trained model (match training seq_len)
            # Use XGBoost classifier direction score as primary signal source.
            # Even if quality gate fails, the freshly-trained classifier provides
            # a weak directional signal that is useful as one of 15 scoring factors.
            from src.analysis.features import FeatureEngineer, FEATURE_COLUMNS

            fe = FeatureEngineer()
            pred_start = (today - timedelta(days=200)).isoformat()
            pred_df = fe.build_features(stock_id, pred_start, today.isoformat())

            continuous_score = 0.5  # default: no signal
            signal_source = "none"

            # 1. XGBoost classifier direction score (always available after training)
            # Use the freshly-trained classifier even if it didn't pass quality gate.
            # The classifier was trained in this session — it exists in trainer._xgb_cls_raw
            xgb_cls = getattr(trainer, "_xgb_cls_fresh", None) or trainer.xgb_cls
            if xgb_cls is not None and not pred_df.empty:
                feature_cols = trainer.feature_cols or [
                    c for c in FEATURE_COLUMNS if c in pred_df.columns
                ]
                X_tab, _ = fe.prepare_tabular(pred_df, feature_cols)
                if len(X_tab) > 0:
                    dir_score = xgb_cls.predict_direction_score(X_tab[-1:])
                    # dir_score in [-1, 1] → sigmoid mapping
                    # Scale=3.0 for quality-passed models (more spread: ±0.5 → [0.18, 0.82])
                    # Scale=2.0 for weak models (conservative)
                    if gate.get("xgb_cls_passed", False):
                        sigmoid_scale = 3.0
                    else:
                        sigmoid_scale = 2.0
                    raw_score = 1.0 / (1.0 + np.exp(-dir_score[0] * sigmoid_scale))
                    # Dampen to [0.3, 0.7] range if quality gate failed
                    if not gate.get("xgb_cls_passed", False):
                        continuous_score = 0.5 + (raw_score - 0.5) * 0.4  # [0.3, 0.7]
                        signal_source = "xgb_classifier(weak)"
                    else:
                        continuous_score = max(0.1, min(0.9, float(raw_score)))
                        signal_source = "xgb_classifier"

            # 2. Fallback: LSTM/XGB regressor ensemble
            if signal_source == "none":
                result = trainer.predict(
                    start_date=pred_start,
                    end_date=today.isoformat(),
                    seq_len=40,
                )
                if result is not None:
                    total_return = float(result.predicted_returns.sum())
                    continuous_score = 1.0 / (1.0 + np.exp(-total_return * 50.0))
                    continuous_score = max(0.1, min(0.9, continuous_score))
                    signal_source = f"ensemble({result.signal})"

            logger.info(
                "ML quality training completed for %s: signal=%s score=%.3f gate=%s",
                stock_id,
                signal_source,
                continuous_score,
                gate,
            )
            # Return score even if gate failed — the 15-factor system
            # uses it as a weak signal (ml_ensemble weight = 15%)
            return {stock_id: continuous_score}
        except ValueError as e:
            logger.warning(
                "ML quality training skipped for %s (insufficient data): %s",
                stock_id,
                e,
            )
            return {}
        except Exception as e:
            logger.warning("ML quality training failed for %s: %s", stock_id, e)
            return {}
        finally:
            lock.release()

    async def _enhance_sentiment(self, data: StockData) -> dict | None:
        """LLM 情緒特徵萃取 (Haiku, 1 call)

        Phase 2 的 LLM 呼叫 #1: 從新聞/PTT/法人動向萃取結構化情緒。
        """
        try:
            logger.info("  ├─ LLM 情緒萃取: 開始 (Haiku)")
            from src.agents.narrative_agent import extract_sentiment

            result = await extract_sentiment(
                stock_id=data.stock_id,
                sentiment_df=data.sentiment_df,
                trust_info=data.trust_info,
                global_data=data.global_data,
            )
            logger.info("  ├─ LLM 情緒萃取: 完成")
            return result
        except Exception as e:
            logger.warning("LLM sentiment extraction failed: %s", e)
            return None

    # ─── Phase 3: Multi-Factor Scoring ───────────────────

    def _score(self, data: StockData, regime: str, ml_scores: dict) -> ScoreResult:
        """20 因子加權評分 — 直接重用 market_service.score_stock()"""

        # Prepare sector data for sector_rotation factor (prefer cached from market scan)
        sector_data = None
        try:
            cached_json = get_data_cache("sector_aggregates", date.today())
            if not cached_json:
                cached_json = get_data_cache_latest("sector_aggregates")
            if cached_json:
                sector_data = json.loads(cached_json)
            else:
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

        # Detect ex-dividend window (±2 days from ex-date) — DB-first
        ex_div = False
        try:
            today = date.today()
            cache_key = f"dividend:{data.stock_id}"
            div_df = None

            # DB-first
            cached_json = get_data_cache(cache_key, today)
            if cached_json:
                try:
                    div_df = pd.read_json(StringIO(cached_json), orient="records")
                    if "date" in div_df.columns:
                        div_df["date"] = pd.to_datetime(div_df["date"]).dt.date
                except Exception:
                    div_df = None

            if div_df is None:
                fetcher = StockFetcher()
                div_df = fetcher.fetch_dividend_history(
                    data.stock_id,
                    start=(today - timedelta(days=5)).isoformat(),
                    end=(today + timedelta(days=2)).isoformat(),
                )
                if not div_df.empty:
                    try:
                        upsert_data_cache(
                            cache_key,
                            today,
                            div_df.to_json(orient="records", date_format="iso"),
                        )
                    except Exception:
                        pass

            if div_df is not None and not div_df.empty and "date" in div_df.columns:
                ex_div = any(0 <= (today - d).days <= 2 for d in div_df["date"])
        except Exception:
            pass

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
            ex_dividend_window=ex_div,
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
        risk_decision: RiskDecision | None = None,
    ) -> NarrativeResult:
        """LLM 敘事生成 (Sonnet, 1 call) — 或演算法 fallback

        risk_decision is passed so LLM can incorporate risk-adjusted position
        and stop levels into its narrative (pipeline order: risk → narrative).
        """
        try:
            # Compute technical reference levels for LLM key_levels guidance
            technical_data = None
            if data.df_tech is not None and len(data.df_tech) >= 20:
                try:
                    close = data.df_tech["close"]
                    technical_data = {
                        "ma20": close.rolling(20).mean().iloc[-1]
                        if len(close) >= 20
                        else None,
                        "ma60": close.rolling(60).mean().iloc[-1]
                        if len(close) >= 60
                        else None,
                        "low_20d": close.iloc[-20:].min(),
                    }
                    # ATR(14)
                    if (
                        len(data.df_tech) >= 14
                        and "high" in data.df_tech.columns
                        and "low" in data.df_tech.columns
                    ):
                        high = data.df_tech["high"]
                        low = data.df_tech["low"]
                        prev_close = close.shift(1)
                        tr = pd.concat(
                            [
                                high - low,
                                (high - prev_close).abs(),
                                (low - prev_close).abs(),
                            ],
                            axis=1,
                        ).max(axis=1)
                        technical_data["atr14"] = tr.rolling(14).mean().iloc[-1]
                except Exception:
                    pass  # Non-critical, LLM can still generate without it

            # Build risk context for LLM
            risk_context = None
            if risk_decision:
                risk_context = {
                    "action": risk_decision.action,
                    "position_size_pct": round(risk_decision.position_size * 100, 1),
                    "approved": risk_decision.approved,
                    "stop_loss": risk_decision.stop_loss,
                    "take_profit": risk_decision.take_profit,
                    "risk_notes": risk_decision.risk_notes[:3],
                }

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
                technical_data=technical_data,
                risk_context=risk_context,
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
                verdict=result.get("verdict", ""),
                verdict_short=result.get("verdict_short", ""),
                risk_warning=result.get("risk_warning", ""),
                confidence_comment=result.get("confidence_comment", ""),
            )
        except Exception as e:
            logger.warning("LLM narrative failed, using algorithm fallback: %s", e)
            return self._algorithm_narrative(data, score_result, risk_decision)

    def _algorithm_narrative(
        self,
        data: StockData,
        score_result: ScoreResult,
        risk_decision: RiskDecision | None = None,
    ) -> NarrativeResult:
        """Enhanced algorithm fallback narrative with risk-adjusted recommendations."""
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
            [
                (name, d)
                for name, d in score_result.factor_details.items()
                if d.get("available")
            ],
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

        # Risk-adjusted position suggestion
        position_suggestion = ""
        verdict_short = ""
        risk_warning = ""
        if risk_decision:
            if risk_decision.approved and risk_decision.action == "buy":
                position_suggestion = (
                    f"建議部位 {risk_decision.position_size * 100:.0f}%"
                )
                if risk_decision.stop_loss:
                    position_suggestion += f"，停損 ${risk_decision.stop_loss:.1f}"
                verdict_short = f"{outlook}，建議{risk_decision.action}"
            elif not risk_decision.approved:
                verdict_short = "風控否決，建議觀望"
                risk_warning = (
                    "；".join(risk_decision.risk_notes[:2])
                    if risk_decision.risk_notes
                    else ""
                )
            else:
                verdict_short = outlook
        else:
            verdict_short = outlook

        # Confidence comment
        conf = score_result.confidence
        if conf > 0.7:
            confidence_comment = "信心度高，因子高度一致"
        elif conf > 0.5:
            confidence_comment = "信心度中等，部分因子分歧"
        else:
            confidence_comment = "信心度偏低，建議降低部位或觀望"

        return NarrativeResult(
            outlook=outlook,
            outlook_horizon="1-2 週",
            key_drivers=drivers,
            risks=risks[:3],
            catalysts=[],
            key_levels={},
            position_suggestion=position_suggestion,
            source="algorithm",
            verdict_short=verdict_short,
            risk_warning=risk_warning,
            confidence_comment=confidence_comment,
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

        # Volatility-adaptive ATR-based stop loss/take profit
        stop_loss = None
        take_profit = None
        if action == "buy" and not data.df.empty and len(data.df) >= 20:
            try:
                close = data.df["close"].dropna()
                high = data.df["high"].dropna() if "high" in data.df.columns else close
                low = data.df["low"].dropna() if "low" in data.df.columns else close
                # ATR(14)
                tr = pd.concat(
                    [
                        high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                atr = float(tr.tail(14).mean())
                if atr > 0:
                    # Adaptive ATR multiplier based on VIX
                    # High vol (VIX>30) → wider stops (3.5x) to avoid noise
                    # Low vol (VIX<15) → tighter stops (2.0x)
                    vix = 20.0
                    if data.macro_data:
                        vix = data.macro_data.get("vix", 20.0)
                    atr_mult = max(1.5, min(4.0, 2.0 + (vix - 15) * 0.05))
                    stop_loss = round(data.current_price - atr_mult * atr, 2)
                    take_profit = round(data.current_price + atr_mult * 2.0 * atr, 2)
                    risk_notes.append(
                        f"ATR={atr:.1f} mult={atr_mult:.1f}x (VIX={vix:.0f})"
                    )
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
            signal=risk_decision.action
            if not risk_decision.approved
            else score_result.signal,
            confidence=score_result.confidence,
            confidence_breakdown=score_result.confidence_breakdown,
            factor_details=score_result.factor_details,
            regime=regime,
            reasoning="；".join(score_result.reasoning)
            if score_result.reasoning
            else "",
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
            "effective_coverage": round(
                sum(
                    BASE_WEIGHTS.get(f.name, 0)
                    for f in score_result.factors
                    if f.available
                ),
                2,
            ),
            **result.confidence_breakdown,
        }

        # Add backward-compatible score fields
        def _fs(name: str) -> float:
            d = result.factor_details.get(name, {})
            return round(d.get("score", 0.5), 4) if d else 0.5

        scan_record.update(
            {
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
            }
        )

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
            "technical_data": {
                k: v.get("score") for k, v in result.factor_details.items()
            },
            "institutional_data": {
                "foreign_net_5d": data.trust_info.get("foreign_cumulative", 0),
                "trust_net_5d": data.trust_info.get("trust_cumulative", 0),
            }
            if data.trust_info
            else None,
            "risk_approved": 1
            if asdict(RiskDecision(**result.risk_decision)).get("approved", True)
            else 0,
            "pipeline_version": "3.0",
        }

        try:
            await asyncio.to_thread(save_pipeline_result_record, pipeline_record)
            logger.info("  │  ✓ PipelineResult 已儲存")
        except Exception as e:
            logger.error("  │  ✗ PipelineResult 儲存失敗: %s", e)

        # 2b. Save to Prediction + TradeJournal (for get_prediction_history)
        try:
            risk = result.risk_decision
            agent_dict = {
                "action": risk.get("action", result.signal),
                "confidence": result.confidence,
                "position_size": risk.get("position_size", 0),
                "reasoning": result.reasoning,
                "approved": risk.get("approved", True),
                "risk_notes": "; ".join(risk.get("risk_notes", [])),
                "analyst_reports": [],
            }
            predicted_change = result.total_score - 0.5
            target_price = (
                data.current_price * (1 + predicted_change * 0.1)
                if data.current_price
                else 0
            )

            await asyncio.to_thread(
                save_pipeline_result,
                stock_id=data.stock_id,
                stock_name=data.stock_name,
                current_price=data.current_price,
                signal=result.signal,
                confidence=result.confidence,
                target_price=round(target_price, 2),
                predicted_change=round(predicted_change * 0.1, 4),
                reasoning=result.reasoning,
                agent_decision=agent_dict,
                technical_data={
                    k: v.get("score") for k, v in result.factor_details.items()
                },
            )
            logger.info("  │  ✓ Prediction + TradeJournal 已儲存")
        except Exception as e:
            logger.error("  │  ✗ Prediction + TradeJournal 儲存失敗: %s", e)

        # 3. Factor IC records
        try:
            ic_records = []
            for f in score_result.factors:
                if f.available and f.raw_value is not None:
                    ic_records.append(
                        {
                            "record_date": date.today(),
                            "stock_id": data.stock_id,
                            "factor_name": f.name,
                            "factor_score": f.score,
                        }
                    )
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
