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
        stock_name = stock_name or STOCK_LIST.get(stock_id, stock_id)

        # Phase 1: 數據收集
        yield _sse_event("data_collection", "running", 5, f"收集 {stock_id} {stock_name} 數據...")
        try:
            data = await self._collect_data(stock_id, stock_name)
        except Exception as e:
            logger.error("Phase 1 failed for %s: %s", stock_id, e)
            yield _sse_event("data_collection", "error", 5, f"數據收集失敗: {e}")
            yield _sse_event("complete", "error", 100, "分析失敗", {
                "stock_id": stock_id, "error": str(e),
            })
            return

        if data.df.empty or len(data.df) < 20:
            yield _sse_event("complete", "error", 100, "價格資料不足", {
                "stock_id": stock_id, "error": "價格資料不足，無法分析",
            })
            return

        yield _sse_event("data_collection", "done", 15, "數據收集完成")

        # Phase 2: 特徵萃取
        yield _sse_event("feature_extraction", "running", 20, "計算 20 因子 + HMM + ML...")
        regime, ml_scores = await self._extract_features(data)
        yield _sse_event("feature_extraction", "done", 40, "特徵萃取完成")

        # Phase 3: 多因子評分
        yield _sse_event("scoring", "running", 45, "多因子加權評分...")
        score_result = self._score(data, regime, ml_scores)
        yield _sse_event("scoring", "done", 55, f"評分完成: {score_result.total_score:.2f} ({score_result.signal})")

        # Phase 4: LLM 敘事生成
        yield _sse_event("narrative", "running", 60, "生成分析報告...")
        narrative = await self._generate_narrative(
            stock_id, stock_name, data, score_result, regime, ml_scores
        )
        yield _sse_event("narrative", "done", 75,
                         f"敘事生成完成 (source={narrative.source})")

        # Phase 5: 風險控制
        yield _sse_event("risk_control", "running", 80, "風險檢查...")
        risk_decision = self._apply_risk_controls(
            stock_id, data, score_result, regime
        )
        yield _sse_event("risk_control", "done", 88,
                         f"風控完成: {risk_decision.action} (approved={risk_decision.approved})")

        # Phase 6: 儲存 + 警報
        yield _sse_event("finalize", "running", 90, "儲存結果...")
        result = self._build_result(
            data, score_result, narrative, risk_decision, regime
        )
        await self._save_and_alert(data, score_result, result)
        yield _sse_event("finalize", "done", 95, "儲存完成")

        # Complete
        yield _sse_event("complete", "done", 100, "分析完成", asdict(result))

    # ─── Phase 1: Data Collection ────────────────────────

    async def _collect_data(self, stock_id: str, stock_name: str) -> StockData:
        """並行收集所有數據源"""

        # Parallel fetches
        (df, trust_info, revenue_df, global_data, macro_data, sentiment_result) = (
            await asyncio.gather(
                self._fetch_price_data(stock_id),
                self._fetch_trust_info(stock_id),
                self._fetch_revenue(stock_id),
                asyncio.to_thread(_fetch_global_market_data),
                asyncio.to_thread(_fetch_macro_data),
                self._fetch_sentiment(stock_id),
                return_exceptions=True,
            )
        )

        # Handle exceptions from gather
        if isinstance(df, Exception):
            logger.error("Price fetch failed: %s", df)
            df = pd.DataFrame()
        if isinstance(trust_info, Exception):
            logger.warning("Trust info fetch failed: %s", trust_info)
            trust_info = {}
        if isinstance(revenue_df, Exception):
            logger.warning("Revenue fetch failed: %s", revenue_df)
            revenue_df = None
        if isinstance(global_data, Exception):
            logger.warning("Global data fetch failed: %s", global_data)
            global_data = {}
        if isinstance(macro_data, Exception):
            logger.warning("Macro data fetch failed: %s", macro_data)
            macro_data = {}
        if isinstance(sentiment_result, Exception):
            logger.warning("Sentiment fetch failed: %s", sentiment_result)
            sentiment_result = (None, {})

        sentiment_df, sentiment_scores = sentiment_result

        # Technical analysis
        df_tech = pd.DataFrame()
        signals = {}
        if not df.empty and len(df) >= 20:
            try:
                analyzer = TechnicalAnalyzer()
                df_tech = analyzer.compute_all(df)
                signals = analyzer.get_signals(df_tech)
            except Exception as e:
                logger.warning("Technical analysis failed: %s", e)

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
        """Fetch sentiment data from DB"""
        try:
            today = date.today()
            sent_df = await asyncio.to_thread(
                get_sentiment, stock_id, today - timedelta(days=14), today
            )
            sentiment_scores = {}
            if sent_df is not None and not sent_df.empty:
                scores = sent_df["sentiment_score"].dropna()
                if not scores.empty:
                    avg = float(scores.mean())
                    sentiment_scores[stock_id] = (avg + 1) / 2  # Normalize to 0-1
            return sent_df, sentiment_scores
        except Exception:
            return None, {}

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

        # 1. Save as MarketScanResult (reuse existing DB schema)
        scan_record = {
            "stock_id": data.stock_id,
            "stock_name": data.stock_name,
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
        except Exception as e:
            logger.error("Failed to save scan result: %s", e)

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
        except Exception as e:
            logger.error("Failed to save pipeline result: %s", e)

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
        except Exception as e:
            logger.error("Failed to save factor IC records: %s", e)

        # 4. Alert generation
        try:
            generate_alerts_from_scan([scan_record])
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
