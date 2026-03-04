"""統一管線 (StockAnalysisService) 測試

測試 6 階段管線的各組件：
- Phase 1-3 不依賴 LLM (純演算法)
- Phase 4 LLM fallback 驗證
- Phase 5 風控邏輯驗證
- 端對端整合測試 (mock LLM)
"""

import asyncio
import json
import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from unittest.mock import patch, AsyncMock, MagicMock

from api.services.stock_analysis_service import (
    StockAnalysisService,
    StockData,
    ScoreResult,
    NarrativeResult,
    RiskDecision,
    AnalysisResult,
    _sse_event,
)
from api.services.market_service import FactorResult


# ═══════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════


@pytest.fixture
def service():
    return StockAnalysisService()


@pytest.fixture
def sample_stock_data(sample_price_df):
    """Create a StockData instance from sample_price_df"""
    from src.analysis.technical import TechnicalAnalyzer

    df = sample_price_df
    analyzer = TechnicalAnalyzer()
    df_tech = analyzer.compute_all(df)
    signals = analyzer.get_signals(df_tech)

    return StockData(
        stock_id="2330",
        stock_name="台積電",
        df=df,
        df_tech=df_tech,
        signals=signals,
        trust_info={
            "foreign_cumulative": 5000,
            "foreign_consecutive_days": 3,
            "trust_cumulative": 1000,
            "trust_consecutive_days": 2,
            "dealer_cumulative": -200,
        },
        revenue_df=None,
        global_data={
            "sox_return": 0.02,
            "tsm_return": 0.015,
            "ewt_return_1d": 0.01,
            "ewt_return_20d": 0.05,
            "ewt_return_60d": 0.08,
        },
        macro_data={
            "vix": 18.5,
            "vix_level": "normal",
            "fx_trend": -0.01,
            "yield_10y_change": 0.02,
            "xli_return_20d": 0.03,
            "xli_vs_sma200": 1.02,
            "xli_spy_ratio_trend": 0.01,
        },
        sentiment_df=None,
        sentiment_scores={"2330": 0.65},
        current_price=650.0,
        price_change_pct=1.5,
    )


# ═══════════════════════════════════════════════════════
# SSE Helper Tests
# ═══════════════════════════════════════════════════════


class TestSSEEvent:
    def test_basic_event(self):
        event = _sse_event("scoring", "running", 45, "計算中...")
        assert event.startswith("data: ")
        assert event.endswith("\n\n")
        payload = json.loads(event[6:].strip())
        assert payload["phase"] == "scoring"
        assert payload["status"] == "running"
        assert payload["progress"] == 45

    def test_event_with_data(self):
        event = _sse_event("complete", "done", 100, "完成", {"score": 0.75})
        payload = json.loads(event[6:].strip())
        assert payload["data"]["score"] == 0.75


# ═══════════════════════════════════════════════════════
# Phase 2: Feature Extraction Tests
# ═══════════════════════════════════════════════════════


class TestFeatureExtraction:
    def test_detect_regime_returns_valid_state(self, service, sample_price_df):
        regime = service._detect_regime(sample_price_df)
        assert regime in ("bull", "bear", "sideways")

    def test_detect_regime_short_df_returns_sideways(self, service):
        short_df = pd.DataFrame({"close": [100, 101, 102]})
        regime = service._detect_regime(short_df)
        assert regime == "sideways"

    def test_detect_regime_empty_df_returns_sideways(self, service):
        regime = service._detect_regime(pd.DataFrame())
        assert regime == "sideways"

    def test_predict_ml_no_model_returns_empty(self, service):
        ml_scores = service._predict_ml("9999", pd.DataFrame())
        assert ml_scores == {}


# ═══════════════════════════════════════════════════════
# Phase 3: Scoring Tests
# ═══════════════════════════════════════════════════════


class TestScoring:
    def test_score_returns_valid_result(self, service, sample_stock_data):
        result = service._score(sample_stock_data, "sideways", {})
        assert isinstance(result, ScoreResult)
        assert 0 <= result.total_score <= 1
        assert result.signal in ("strong_buy", "buy", "hold", "sell", "strong_sell")
        assert 0 <= result.confidence <= 1
        assert result.regime == "sideways"
        assert len(result.factor_details) > 0
        assert len(result.factors) == 20

    def test_score_with_ml(self, service, sample_stock_data):
        ml_scores = {"2330": 0.85}
        result = service._score(sample_stock_data, "bull", ml_scores)
        # ML factor should be available
        ml_detail = result.factor_details.get("ml_ensemble", {})
        assert ml_detail.get("available") is True

    def test_score_regime_affects_weights(self, service, sample_stock_data):
        result_bull = service._score(sample_stock_data, "bull", {})
        result_bear = service._score(sample_stock_data, "bear", {})
        # Different regimes should produce different weights
        assert result_bull.weights != result_bear.weights

    def test_score_confidence_breakdown(self, service, sample_stock_data):
        result = service._score(sample_stock_data, "sideways", {})
        bd = result.confidence_breakdown
        assert "confidence_agreement" in bd
        assert "confidence_strength" in bd
        assert "confidence_coverage" in bd
        assert "confidence_freshness" in bd
        assert "risk_discount" in bd

    def test_score_reasoning_not_empty(self, service, sample_stock_data):
        result = service._score(sample_stock_data, "sideways", {})
        assert len(result.reasoning) > 0


# ═══════════════════════════════════════════════════════
# Phase 4: Narrative Tests
# ═══════════════════════════════════════════════════════


class TestNarrative:
    def test_algorithm_narrative_bullish(self, service, sample_stock_data):
        score_result = ScoreResult(
            total_score=0.75,
            signal="strong_buy",
            confidence=0.8,
            confidence_breakdown={},
            factor_details={
                "foreign_flow": {"score": 0.8, "weight": 0.11, "available": True},
                "technical_signal": {"score": 0.7, "weight": 0.08, "available": True},
                "short_momentum": {"score": 0.6, "weight": 0.07, "available": True},
            },
            factors=[],
            weights={},
            reasoning=["外資買超"],
            regime="bull",
        )
        result = service._algorithm_narrative(sample_stock_data, score_result)
        assert isinstance(result, NarrativeResult)
        assert "多" in result.outlook
        assert result.source == "algorithm"
        assert len(result.key_drivers) > 0

    def test_algorithm_narrative_bearish(self, service, sample_stock_data):
        score_result = ScoreResult(
            total_score=0.25,
            signal="strong_sell",
            confidence=0.7,
            confidence_breakdown={},
            factor_details={
                "foreign_flow": {"score": 0.2, "weight": 0.11, "available": True},
            },
            factors=[],
            weights={},
            reasoning=["外資賣超"],
            regime="bear",
        )
        result = service._algorithm_narrative(sample_stock_data, score_result)
        assert "空" in result.outlook

    def test_algorithm_narrative_neutral(self, service, sample_stock_data):
        score_result = ScoreResult(
            total_score=0.50,
            signal="hold",
            confidence=0.5,
            confidence_breakdown={},
            factor_details={},
            factors=[],
            weights={},
            reasoning=[],
            regime="sideways",
        )
        result = service._algorithm_narrative(sample_stock_data, score_result)
        assert "中性" in result.outlook


# ═══════════════════════════════════════════════════════
# Phase 5: Risk Control Tests
# ═══════════════════════════════════════════════════════


class TestRiskControl:
    def test_buy_signal_produces_position(self, service, sample_stock_data):
        score_result = ScoreResult(
            total_score=0.7,
            signal="buy",
            confidence=0.7,
            confidence_breakdown={},
            factor_details={},
            factors=[],
            weights={},
            reasoning=[],
            regime="bull",
        )
        decision = service._apply_risk_controls(
            "2330", sample_stock_data, score_result, "bull"
        )
        assert isinstance(decision, RiskDecision)
        assert decision.action == "buy"
        assert decision.position_size > 0
        assert decision.position_size <= 0.20

    def test_hold_signal_zero_position(self, service, sample_stock_data):
        score_result = ScoreResult(
            total_score=0.5,
            signal="hold",
            confidence=0.5,
            confidence_breakdown={},
            factor_details={},
            factors=[],
            weights={},
            reasoning=[],
            regime="sideways",
        )
        decision = service._apply_risk_controls(
            "2330", sample_stock_data, score_result, "sideways"
        )
        assert decision.action == "hold"
        assert decision.position_size == 0.0

    def test_bear_regime_reduces_position(self, service, sample_stock_data):
        score_result = ScoreResult(
            total_score=0.65,
            signal="buy",
            confidence=0.6,
            confidence_breakdown={},
            factor_details={},
            factors=[],
            weights={},
            reasoning=[],
            regime="bear",
        )
        decision = service._apply_risk_controls(
            "2330", sample_stock_data, score_result, "bear"
        )
        assert decision.action == "buy"
        assert any("熊市" in note for note in decision.risk_notes)
        # Position should be smaller due to bear regime
        assert decision.position_size < 0.12

    def test_sell_signal_produces_action(self, service, sample_stock_data):
        score_result = ScoreResult(
            total_score=0.3,
            signal="sell",
            confidence=0.6,
            confidence_breakdown={},
            factor_details={},
            factors=[],
            weights={},
            reasoning=[],
            regime="bear",
        )
        decision = service._apply_risk_controls(
            "2330", sample_stock_data, score_result, "bear"
        )
        assert decision.action == "sell"

    def test_circuit_breaker_blocks_buy(self, service, sample_stock_data):
        rm = service._get_risk_manager()
        rm._circuit_breaker_active = True

        score_result = ScoreResult(
            total_score=0.8,
            signal="strong_buy",
            confidence=0.9,
            confidence_breakdown={},
            factor_details={},
            factors=[],
            weights={},
            reasoning=[],
            regime="bull",
        )
        decision = service._apply_risk_controls(
            "2330", sample_stock_data, score_result, "bull"
        )
        assert decision.action == "hold"
        assert decision.position_size == 0.0
        assert any("迴路斷路器" in note for note in decision.risk_notes)

        # Cleanup
        rm._circuit_breaker_active = False

    def test_atr_stop_loss_calculated(self, service, sample_stock_data):
        score_result = ScoreResult(
            total_score=0.7,
            signal="buy",
            confidence=0.7,
            confidence_breakdown={},
            factor_details={},
            factors=[],
            weights={},
            reasoning=[],
            regime="bull",
        )
        decision = service._apply_risk_controls(
            "2330", sample_stock_data, score_result, "bull"
        )
        # Should have ATR-based stop loss
        assert decision.stop_loss is not None
        assert decision.stop_loss < sample_stock_data.current_price
        assert decision.take_profit is not None
        assert decision.take_profit > sample_stock_data.current_price


# ═══════════════════════════════════════════════════════
# Phase 6: Build Result Tests
# ═══════════════════════════════════════════════════════


class TestBuildResult:
    def test_build_result_structure(self, service, sample_stock_data):
        score_result = ScoreResult(
            total_score=0.65,
            signal="buy",
            confidence=0.7,
            confidence_breakdown={"confidence_agreement": 0.7},
            factor_details={"foreign_flow": {"score": 0.8, "available": True}},
            factors=[],
            weights={},
            reasoning=["外資買超"],
            regime="bull",
        )
        narrative = NarrativeResult(
            outlook="短期偏多",
            key_drivers=["外資買超"],
            source="algorithm",
        )
        risk_decision = RiskDecision(
            action="buy",
            position_size=0.10,
            approved=True,
        )

        result = service._build_result(
            sample_stock_data, score_result, narrative, risk_decision, "bull"
        )
        assert isinstance(result, AnalysisResult)
        assert result.stock_id == "2330"
        assert result.total_score == 0.65
        assert result.signal == "buy"
        assert result.regime == "bull"
        assert result.pipeline_version == "3.0"
        assert "outlook" in result.narrative


# ═══════════════════════════════════════════════════════
# Data Structures Tests
# ═══════════════════════════════════════════════════════


class TestDataStructures:
    def test_stock_data_creation(self):
        data = StockData(
            stock_id="2330",
            stock_name="台積電",
            df=pd.DataFrame(),
            df_tech=pd.DataFrame(),
            signals={},
            trust_info={},
            revenue_df=None,
            global_data=None,
            macro_data=None,
            sentiment_df=None,
            sentiment_scores={},
        )
        assert data.current_price == 0.0
        assert data.price_change_pct == 0.0

    def test_narrative_result_defaults(self):
        n = NarrativeResult()
        assert n.outlook == ""
        assert n.key_drivers == []
        assert n.source == "algorithm"

    def test_risk_decision_defaults(self):
        r = RiskDecision(action="hold", position_size=0.0)
        assert r.approved is True
        assert r.risk_notes == []
        assert r.stop_loss is None

    def test_analysis_result_pipeline_version(self):
        result = AnalysisResult(
            stock_id="2330",
            stock_name="台積電",
            current_price=650,
            price_change_pct=1.5,
            total_score=0.65,
            signal="buy",
            confidence=0.7,
            confidence_breakdown={},
            factor_details={},
            regime="bull",
            reasoning="外資買超",
            narrative={},
            risk_decision={},
            analysis_date="2025-01-01",
        )
        assert result.pipeline_version == "3.0"


# ═══════════════════════════════════════════════════════
# Integration Tests (end-to-end with mocks)
# ═══════════════════════════════════════════════════════


class TestIntegration:
    @pytest.mark.asyncio
    async def test_analyze_stock_end_to_end(self, service, sample_price_df):
        """端對端測試：mock 外部數據源，驗證完整管線"""

        # Mock all external dependencies
        with patch("api.services.stock_analysis_service.get_stock_prices") as mock_prices, \
             patch("api.services.stock_analysis_service.upsert_stock_prices"), \
             patch("api.services.stock_analysis_service.get_sentiment") as mock_sent, \
             patch("api.services.stock_analysis_service._fetch_global_market_data") as mock_global, \
             patch("api.services.stock_analysis_service._fetch_macro_data") as mock_macro, \
             patch("api.services.stock_analysis_service.save_market_scan"), \
             patch("api.services.stock_analysis_service.save_pipeline_result_record"), \
             patch("api.services.stock_analysis_service.save_factor_ic_records"), \
             patch("api.services.stock_analysis_service.generate_alerts_from_scan"), \
             patch("src.agents.narrative_agent.call_claude") as mock_llm:

            mock_prices.return_value = sample_price_df
            mock_sent.return_value = pd.DataFrame()
            mock_global.return_value = {"sox_return": 0.01, "tsm_return": 0.01,
                                         "ewt_return_1d": 0.005, "ewt_return_20d": 0.03,
                                         "ewt_return_60d": 0.05}
            mock_macro.return_value = {"vix": 18, "vix_level": "normal",
                                        "fx_trend": 0, "yield_10y_change": 0,
                                        "xli_return_20d": 0.02, "xli_vs_sma200": 1.01,
                                        "xli_spy_ratio_trend": 0}
            # LLM returns structured sentiment
            mock_llm.return_value = json.dumps({
                "sentiment_score": 0.6,
                "key_themes": ["AI需求"],
                "contrarian_flag": False,
                "geopolitical_risk": "low",
                "sector_sentiment": "positive",
                "catalyst_timeline": "near",
            })

            # Also mock TWSEScanner
            with patch("api.services.stock_analysis_service.TWSEScanner") as mock_scanner:
                mock_scanner_inst = MagicMock()
                mock_scanner_inst.get_trust_info.return_value = {
                    "foreign_cumulative": 3000,
                    "foreign_consecutive_days": 3,
                    "trust_cumulative": 500,
                    "trust_consecutive_days": 2,
                }
                mock_scanner.return_value = mock_scanner_inst

                events = []
                async for event_str in service.analyze_stock("2330", "台積電"):
                    for line in event_str.strip().split("\n"):
                        if line.startswith("data: "):
                            events.append(json.loads(line[6:]))

                # Verify events
                assert len(events) > 0
                phases = [e["phase"] for e in events]
                assert "data_collection" in phases
                assert "feature_extraction" in phases
                assert "scoring" in phases
                assert "risk_control" in phases
                assert "complete" in phases

                # Find final result
                complete_event = next(e for e in events if e["phase"] == "complete")
                assert complete_event["status"] == "done"
                result_data = complete_event.get("data", {})
                assert result_data["stock_id"] == "2330"
                assert "total_score" in result_data
                assert "signal" in result_data
                assert "narrative" in result_data
                assert "risk_decision" in result_data

    @pytest.mark.asyncio
    async def test_analyze_stock_insufficient_data(self, service):
        """資料不足時應回傳錯誤"""
        with patch("api.services.stock_analysis_service.get_stock_prices") as mock_prices, \
             patch("api.services.stock_analysis_service.get_sentiment") as mock_sent, \
             patch("api.services.stock_analysis_service._fetch_global_market_data") as mock_global, \
             patch("api.services.stock_analysis_service._fetch_macro_data") as mock_macro, \
             patch("api.services.stock_analysis_service.TWSEScanner") as mock_scanner:

            mock_prices.return_value = pd.DataFrame()
            mock_sent.return_value = pd.DataFrame()
            mock_global.return_value = {}
            mock_macro.return_value = {}
            mock_scanner_inst = MagicMock()
            mock_scanner_inst.get_trust_info.return_value = {}
            mock_scanner.return_value = mock_scanner_inst

            events = []
            async for event_str in service.analyze_stock("9999"):
                for line in event_str.strip().split("\n"):
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))

            complete = next(e for e in events if e["phase"] == "complete")
            assert complete["status"] == "error"

    @pytest.mark.asyncio
    async def test_llm_fallback_narrative(self, service, sample_stock_data):
        """LLM 不可用時 fallback 到演算法敘事"""
        score_result = service._score(sample_stock_data, "sideways", {})

        # Mock LLM to fail
        with patch("src.agents.narrative_agent.call_claude", side_effect=Exception("LLM unavailable")):
            narrative = await service._generate_narrative(
                "2330", "台積電", sample_stock_data,
                score_result, "sideways", {}
            )
            assert narrative.source == "algorithm"
            assert narrative.outlook != ""
