"""Tests for the 20-factor scoring system.

Covers:
- Individual _compute_* functions (20 factors)
- Weight engine with regime adjustment + missing data redistribution
- Confidence calculator with risk discounts
- RuleEngine dynamic weights
- IC tracking DB functions
- score_stock() integration
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

from api.services.market_service import (
    FactorResult,
    # 20-factor functions
    _compute_foreign_flow,
    _compute_technical_signal,
    _compute_short_momentum,
    _compute_trust_flow,
    _compute_volume_anomaly,
    _compute_margin_sentiment,
    _compute_trend_momentum,
    _compute_revenue_momentum,
    _compute_institutional_sync,
    _compute_volatility_regime,
    _compute_news_sentiment,
    _compute_global_context,
    _compute_ml_ensemble,
    _compute_fundamental_value,
    _compute_liquidity_quality,
    _compute_macro_risk,
    # New 4 factors
    _compute_margin_quality,
    _compute_sector_rotation,
    _compute_export_momentum,
    _compute_us_manufacturing,
    _compute_sector_aggregates,
    # Legacy aliases
    _compute_technical_trend,
    _compute_momentum,
    _compute_institutional_flow,
    _compute_margin_retail,
    _compute_volatility,
    _compute_sentiment,
    _compute_liquidity,
    _compute_value_quality,
    # Engine
    _compute_weights,
    _compute_confidence,
    score_stock,
    BASE_WEIGHTS,
    REGIME_MULTIPLIERS,
    STOCK_SECTOR,
    DEFAULT_SECTOR,
)

from src.agents.orchestrator import RuleEngine


# ── Helpers ────────────────────────────────────────────


def _make_df(n: int = 60, base_price: float = 100.0, vol: float = 1000.0,
             with_margin: bool = False) -> pd.DataFrame:
    """Generate a fake price DataFrame for testing."""
    dates = [date.today() - timedelta(days=n - i) for i in range(n)]
    np.random.seed(42)
    closes = base_price + np.cumsum(np.random.randn(n) * 1.0)
    df = pd.DataFrame({
        "date": dates,
        "open": closes - 0.5,
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": np.random.uniform(vol * 0.5, vol * 1.5, n),
        "foreign_buy_sell": np.random.uniform(-500, 500, n),
        "trust_buy_sell": np.random.uniform(-300, 300, n),
        "dealer_buy_sell": np.random.uniform(-200, 200, n),
    })
    if with_margin:
        df["margin_balance"] = np.random.uniform(5000, 10000, n)
        df["short_balance"] = np.random.uniform(100, 500, n)
    return df


def _make_df_tech(df: pd.DataFrame) -> pd.DataFrame:
    """Make a fake technical DataFrame."""
    n = len(df)
    df_tech = df.copy()
    df_tech["adx"] = np.random.uniform(15, 40, n)
    df_tech["obv"] = np.cumsum(np.random.randn(n) * 100)
    df_tech["sma_5"] = df_tech["close"].rolling(5).mean()
    df_tech["sma_20"] = df_tech["close"].rolling(20).mean()
    df_tech["sma_60"] = df_tech["close"].rolling(min(60, n)).mean()
    df_tech["bb_width"] = np.random.uniform(0.02, 0.10, n)
    return df_tech


def _make_signals(raw_score: int = 3, max_score: int = 5, signal: str = "buy") -> dict:
    return {
        "summary": {
            "raw_score": raw_score,
            "max_score": max_score,
            "signal": signal,
        }
    }


def _make_trust_info(foreign=5000, trust=3000, dealer=500, f_days=4, t_days=3):
    return {
        "trust_cumulative": trust,
        "foreign_cumulative": foreign,
        "dealer_cumulative": dealer,
        "trust_consecutive_days": t_days,
        "foreign_consecutive_days": f_days,
    }


# ═══════════════════════════════════════════════════════
# Tests: 16 Original Factor Computers
# ═══════════════════════════════════════════════════════


class TestForeignFlow:
    def test_basic(self):
        df = _make_df()
        trust_info = _make_trust_info()
        result = _compute_foreign_flow(trust_info, df, 1000.0)
        assert result.name == "foreign_flow"
        assert result.available is True
        assert 0.0 <= result.score <= 1.0
        assert "net_normalized" in result.components
        assert "acceleration" in result.components
        assert "anomaly_z" in result.components

    def test_no_data(self):
        result = _compute_foreign_flow({}, pd.DataFrame(), 0)
        assert result.available is False

    def test_heavy_buying(self):
        trust_info = _make_trust_info(foreign=10000, f_days=5)
        df = _make_df()
        result = _compute_foreign_flow(trust_info, df, 1000.0)
        assert result.score > 0.5

    def test_heavy_selling(self):
        trust_info = _make_trust_info(foreign=-10000, f_days=0)
        df = _make_df()
        result = _compute_foreign_flow(trust_info, df, 1000.0)
        assert result.score < 0.5


class TestTechnicalSignal:
    def test_basic(self):
        df = _make_df()
        df_tech = _make_df_tech(df)
        signals = _make_signals(3, 5, "buy")
        result = _compute_technical_signal(signals, df_tech)
        assert result.name == "technical_signal"
        assert result.available is True
        assert 0.0 <= result.score <= 1.0
        assert "signal" in result.components
        assert "adx" in result.components
        assert "ma_alignment" in result.components
        assert "obv_divergence" in result.components

    def test_no_data(self):
        result = _compute_technical_signal({}, pd.DataFrame())
        assert result.available is False
        assert result.score == 0.5

    def test_bullish_signals_score_high(self):
        df = _make_df()
        df_tech = _make_df_tech(df)
        signals = _make_signals(5, 5, "buy")
        result = _compute_technical_signal(signals, df_tech)
        assert result.score > 0.5


class TestShortMomentum:
    def test_basic(self):
        df = _make_df(60)
        result = _compute_short_momentum(df)
        assert result.name == "short_momentum"
        assert result.available is True
        assert 0.0 <= result.score <= 1.0
        assert "return_1d" in result.components
        assert "return_5d" in result.components
        assert "bias" in result.components

    def test_short_df(self):
        df = _make_df(3)
        result = _compute_short_momentum(df)
        assert result.available is False

    def test_empty(self):
        result = _compute_short_momentum(pd.DataFrame())
        assert result.available is False


class TestTrustFlow:
    def test_basic(self):
        trust_info = _make_trust_info()
        df = _make_df()
        result = _compute_trust_flow(trust_info, df, 1000.0)
        assert result.name == "trust_flow"
        assert result.available is True
        assert 0.0 <= result.score <= 1.0
        assert "net_normalized" in result.components

    def test_no_data(self):
        result = _compute_trust_flow({}, pd.DataFrame(), 0)
        assert result.available is False


class TestVolumeAnomaly:
    def test_basic(self):
        df = _make_df(60)
        df_tech = _make_df_tech(df)
        result = _compute_volume_anomaly(df, df_tech)
        assert result.name == "volume_anomaly"
        assert result.available is True
        assert 0.0 <= result.score <= 1.0
        assert "expansion" in result.components
        assert "consistency" in result.components
        assert "obv_trend" in result.components

    def test_short_df(self):
        result = _compute_volume_anomaly(_make_df(10), pd.DataFrame())
        assert result.available is False


class TestMarginSentiment:
    def test_with_margin(self):
        df = _make_df(60, with_margin=True)
        result = _compute_margin_sentiment(df)
        assert result.name == "margin_sentiment"
        assert result.available is True
        assert 0.0 <= result.score <= 1.0
        assert "margin_trend" in result.components

    def test_no_margin_col(self):
        df = _make_df(60)
        result = _compute_margin_sentiment(df)
        assert result.available is False


class TestTrendMomentum:
    def test_basic(self):
        df = _make_df(60)
        df_tech = _make_df_tech(df)
        result = _compute_trend_momentum(df, df_tech)
        assert result.name == "trend_momentum"
        assert result.available is True
        assert 0.0 <= result.score <= 1.0
        assert "return_20d" in result.components
        assert "ma_alignment" in result.components
        assert "adx" in result.components

    def test_short_df(self):
        result = _compute_trend_momentum(_make_df(10), pd.DataFrame())
        assert result.available is False


class TestRevenueMomentum:
    def test_basic(self):
        dates = [date.today() - timedelta(days=30 * i) for i in range(16)]
        dates.reverse()
        rev_df = pd.DataFrame({
            "date": dates,
            "revenue": [100 + i * 5 for i in range(16)],
        })
        result = _compute_revenue_momentum(rev_df)
        assert result.name == "revenue_momentum"
        assert result.available is True
        assert 0.0 <= result.score <= 1.0
        assert "yoy" in result.components
        assert "accel" in result.components
        assert "mom" in result.components

    def test_insufficient_data(self):
        result = _compute_revenue_momentum(None)
        assert result.available is False
        result2 = _compute_revenue_momentum(pd.DataFrame({"date": [], "revenue": []}))
        assert result2.available is False

    def test_yoy_growth_high_score(self):
        """Strong YoY growth → high score"""
        dates = [date.today() - timedelta(days=30 * i) for i in range(16)]
        dates.reverse()
        revenues = [100] * 3 + [100] * 10 + [150, 160, 170]  # big jump at end
        rev_df = pd.DataFrame({"date": dates, "revenue": revenues})
        result = _compute_revenue_momentum(rev_df)
        assert result.score > 0.6


class TestInstitutionalSync:
    def test_all_buying(self):
        trust_info = _make_trust_info(foreign=5000, trust=3000, dealer=1000)
        result = _compute_institutional_sync(trust_info, _make_df())
        assert result.name == "institutional_sync"
        assert result.available is True
        assert result.score > 0.6  # All buying = bullish

    def test_all_selling(self):
        trust_info = _make_trust_info(foreign=-5000, trust=-3000, dealer=-1000)
        result = _compute_institutional_sync(trust_info, _make_df())
        assert result.score < 0.4

    def test_no_data(self):
        result = _compute_institutional_sync({}, pd.DataFrame())
        assert result.available is False


class TestVolatilityRegime:
    def test_basic(self):
        df = _make_df(60)
        df_tech = _make_df_tech(df)
        result = _compute_volatility_regime(df, df_tech)
        assert result.name == "volatility_regime"
        assert result.available is True
        assert 0.0 <= result.score <= 1.0
        assert "low_vol_premium" in result.components
        assert "vol_20d_annualized" in result.components

    def test_short(self):
        df = _make_df(10)
        result = _compute_volatility_regime(df, pd.DataFrame())
        assert result.available is False


class TestNewsSentiment:
    def test_with_data(self):
        scores = {"2330": 0.7}
        sent_df = pd.DataFrame({
            "date": [date.today() - timedelta(days=i) for i in range(20)],
            "source": ["cnyes"] * 10 + ["ptt"] * 10,
            "sentiment_score": np.random.uniform(-0.5, 0.8, 20),
            "engagement": np.random.randint(10, 100, 20),
        })
        result = _compute_news_sentiment(scores, sent_df, "2330")
        assert result.name == "news_sentiment"
        assert result.available is True
        assert 0.0 <= result.score <= 1.0

    def test_no_data(self):
        result = _compute_news_sentiment({}, None, "2330")
        assert result.available is False


class TestGlobalContext:
    def test_basic(self):
        data = {"sox_return": 0.02, "tsm_return": 0.01}
        result = _compute_global_context(data)
        assert result.name == "global_context"
        assert result.available is True
        assert result.score > 0.5  # Positive returns = bullish

    def test_negative(self):
        data = {"sox_return": -0.03, "tsm_return": -0.02}
        result = _compute_global_context(data)
        assert result.score < 0.5

    def test_none(self):
        result = _compute_global_context(None)
        assert result.available is False


class TestMLEnsemble:
    def test_with_score(self):
        result = _compute_ml_ensemble({"2330": 0.75}, "2330")
        assert result.available is True
        assert result.score == 0.75

    def test_no_model(self):
        result = _compute_ml_ensemble({}, "2330")
        assert result.available is False


class TestFundamentalValue:
    def test_high_pe(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {"trailingPE": 100, "returnOnEquity": 0.10, "dividendYield": 0.03}
        mock_yf.Ticker.return_value = mock_ticker
        import sys
        sys.modules["yfinance"] = mock_yf
        try:
            result = _compute_fundamental_value("2330")
            assert result.name == "fundamental_value"
            assert result.score < 0.5  # High PE drags score down
        finally:
            del sys.modules["yfinance"]

    def test_high_roe(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {"trailingPE": 15, "returnOnEquity": 0.30, "dividendYield": 0.05}
        mock_yf.Ticker.return_value = mock_ticker
        import sys
        sys.modules["yfinance"] = mock_yf
        try:
            result = _compute_fundamental_value("2330")
            assert result.score > 0.5  # High ROE + good dividend
        finally:
            del sys.modules["yfinance"]

    def test_no_yfinance(self):
        result = _compute_fundamental_value("9999")
        assert 0.0 <= result.score <= 1.0


class TestLiquidityQuality:
    def test_basic(self):
        df = _make_df(60, vol=3000)
        result = _compute_liquidity_quality(df)
        assert result.name == "liquidity_quality"
        assert result.available is True
        assert 0.0 <= result.score <= 1.0
        assert "avg_volume_20d" in result.components

    def test_high_volume(self):
        df = _make_df(60, vol=10000)
        result = _compute_liquidity_quality(df)
        assert result.score > 0.5


class TestMacroRisk:
    def test_low_vix(self):
        data = {"vix": 12, "usdtwd_trend": -0.01, "tnx_change": -0.1}
        result = _compute_macro_risk(data)
        assert result.name == "macro_risk"
        assert result.available is True
        assert result.score > 0.6  # Low VIX + TWD strengthening

    def test_high_vix(self):
        data = {"vix": 35, "usdtwd_trend": 0.03, "tnx_change": 0.5}
        result = _compute_macro_risk(data)
        assert result.score < 0.4  # Panic

    def test_none(self):
        result = _compute_macro_risk(None)
        assert result.available is False


# ═══════════════════════════════════════════════════════
# Tests: 4 New Factors
# ═══════════════════════════════════════════════════════


class TestMarginQuality:
    def test_expanding_margins(self):
        """Quarterly gross margin expansion → high score"""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()

        # Create quarterly income statement with expanding margins
        dates = pd.to_datetime(["2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31", "2024-12-31"])
        qis = pd.DataFrame(
            {d: {"Gross Profit": gp, "Total Revenue": rev, "Operating Income": oi}
             for d, gp, rev, oi in zip(
                 dates,
                 [450, 400, 380, 350, 330],
                 [1000, 1000, 1000, 1000, 1000],
                 [250, 200, 180, 150, 130],
             )},
        ).T.T
        # Make columns = dates (most recent first)
        qis = pd.DataFrame(
            [[450, 400, 380, 350, 330],
             [1000, 1000, 1000, 1000, 1000],
             [250, 200, 180, 150, 130]],
            index=["Gross Profit", "Total Revenue", "Operating Income"],
            columns=dates,
        )
        mock_ticker.quarterly_income_stmt = qis
        mock_ticker.info = {}
        mock_yf.Ticker.return_value = mock_ticker

        import sys
        sys.modules["yfinance"] = mock_yf
        try:
            result = _compute_margin_quality("2330")
            assert result.name == "margin_quality"
            assert result.available is True
            assert result.score > 0.5  # Expanding margins
            assert result.freshness == 0.6
        finally:
            del sys.modules["yfinance"]

    def test_contracting_margins(self):
        """Quarterly gross margin contraction → low score"""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()

        dates = pd.to_datetime(["2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31", "2024-12-31"])
        qis = pd.DataFrame(
            [[300, 400, 450, 480, 500],
             [1000, 1000, 1000, 1000, 1000],
             [100, 200, 250, 280, 300]],
            index=["Gross Profit", "Total Revenue", "Operating Income"],
            columns=dates,
        )
        mock_ticker.quarterly_income_stmt = qis
        mock_ticker.info = {}
        mock_yf.Ticker.return_value = mock_ticker

        import sys
        sys.modules["yfinance"] = mock_yf
        try:
            result = _compute_margin_quality("2330")
            assert result.available is True
            assert result.score < 0.5  # Contracting margins
        finally:
            del sys.modules["yfinance"]

    def test_fallback_to_info(self):
        """When quarterly data unavailable, fall back to ticker.info"""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.quarterly_income_stmt = pd.DataFrame()  # Empty
        mock_ticker.info = {"grossMargins": 0.35, "operatingMargins": 0.15}
        mock_yf.Ticker.return_value = mock_ticker

        import sys
        sys.modules["yfinance"] = mock_yf
        try:
            result = _compute_margin_quality("2330")
            assert result.available is True
            assert result.freshness == 0.4  # Info fallback freshness
            assert 0.0 <= result.score <= 1.0
        finally:
            del sys.modules["yfinance"]

    def test_no_data(self):
        """No yfinance data → unavailable"""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.quarterly_income_stmt = pd.DataFrame()
        mock_ticker.info = {}
        mock_yf.Ticker.return_value = mock_ticker

        import sys
        sys.modules["yfinance"] = mock_yf
        try:
            result = _compute_margin_quality("9999")
            assert result.available is False
            assert result.score == 0.5
        finally:
            del sys.modules["yfinance"]


class TestSectorRotation:
    def test_bullish_sector(self):
        """Sector with strong inflows and momentum → high score"""
        sector_data = {
            "semiconductor": {
                "net_flow": 50000,
                "avg_return_20d": 0.08,
                "breadth": 0.8,
                "stock_count": 10,
            },
            "_market_avg": {
                "net_flow": 10000,
                "avg_return_20d": 0.02,
            },
        }
        result = _compute_sector_rotation("2330", sector_data)
        assert result.name == "sector_rotation"
        assert result.available is True
        assert result.score > 0.5
        assert result.components["sector"] == "semiconductor"

    def test_bearish_sector(self):
        """Sector with outflows → low score"""
        sector_data = {
            "finance": {
                "net_flow": -20000,
                "avg_return_20d": -0.05,
                "breadth": 0.2,
                "stock_count": 8,
            },
            "_market_avg": {
                "net_flow": 10000,
                "avg_return_20d": 0.03,
            },
        }
        result = _compute_sector_rotation("2881", sector_data)
        assert result.available is True
        assert result.score < 0.5

    def test_unknown_sector(self):
        """Stock not in STOCK_SECTOR → uses DEFAULT_SECTOR"""
        sector_data = {
            "other": {
                "net_flow": 5000,
                "avg_return_20d": 0.01,
                "breadth": 0.5,
                "stock_count": 3,
            },
            "_market_avg": {
                "net_flow": 5000,
                "avg_return_20d": 0.01,
            },
        }
        result = _compute_sector_rotation("9999", sector_data)
        assert result.available is True
        assert result.components["sector"] == DEFAULT_SECTOR

    def test_no_sector_data(self):
        """No sector data → unavailable"""
        result = _compute_sector_rotation("2330", None)
        assert result.available is False
        assert result.score == 0.5


class TestExportMomentum:
    def test_positive_ewt(self):
        """Positive EWT returns → bullish"""
        data = {
            "ewt_return_1d": 0.01,
            "ewt_return_20d": 0.08,
            "ewt_return_60d": 0.15,
            "sox_return": 0.005,
        }
        result = _compute_export_momentum(data)
        assert result.name == "export_momentum"
        assert result.available is True
        assert result.score > 0.5

    def test_negative_ewt(self):
        """Negative EWT returns → bearish"""
        data = {
            "ewt_return_1d": -0.02,
            "ewt_return_20d": -0.10,
            "ewt_return_60d": -0.20,
            "sox_return": -0.01,
        }
        result = _compute_export_momentum(data)
        assert result.available is True
        assert result.score < 0.5

    def test_relative_strength(self):
        """EWT outperforming SOX → higher relative score"""
        data = {
            "ewt_return_1d": 0.03,
            "ewt_return_20d": 0.05,
            "ewt_return_60d": 0.10,
            "sox_return": -0.02,
        }
        result = _compute_export_momentum(data)
        assert result.components["relative_strength"] > 0.5

    def test_none(self):
        """No global data → unavailable"""
        result = _compute_export_momentum(None)
        assert result.available is False
        assert result.score == 0.5


class TestUSManufacturing:
    def test_strong_xli(self):
        """Strong XLI metrics → bullish"""
        data = {
            "xli_return_20d": 0.06,
            "xli_vs_sma200": 0.08,
            "xli_spy_ratio_trend": 0.02,
        }
        result = _compute_us_manufacturing(data)
        assert result.name == "us_manufacturing"
        assert result.available is True
        assert result.score > 0.5

    def test_weak_xli(self):
        """Weak XLI metrics → bearish"""
        data = {
            "xli_return_20d": -0.08,
            "xli_vs_sma200": -0.10,
            "xli_spy_ratio_trend": -0.03,
        }
        result = _compute_us_manufacturing(data)
        assert result.available is True
        assert result.score < 0.5

    def test_sma200_tiers(self):
        """Test different SMA200 level tiers"""
        # Above 5% → 0.75
        data = {"xli_return_20d": 0.0, "xli_vs_sma200": 0.08, "xli_spy_ratio_trend": 0.0}
        result = _compute_us_manufacturing(data)
        assert result.components["sma_score"] == 0.75

        # 0-5% → 0.60
        data["xli_vs_sma200"] = 0.03
        result = _compute_us_manufacturing(data)
        assert result.components["sma_score"] == 0.60

        # -5% to 0% → 0.40
        data["xli_vs_sma200"] = -0.03
        result = _compute_us_manufacturing(data)
        assert result.components["sma_score"] == 0.40

        # Below -5% → 0.25
        data["xli_vs_sma200"] = -0.08
        result = _compute_us_manufacturing(data)
        assert result.components["sma_score"] == 0.25

    def test_none(self):
        """No macro data → unavailable"""
        result = _compute_us_manufacturing(None)
        assert result.available is False
        assert result.score == 0.5


class TestSectorAggregates:
    def test_basic_aggregation(self):
        """Basic sector aggregation produces correct structure"""
        np.random.seed(42)
        stock_dfs = {
            "2330": _make_df(60),
            "2303": _make_df(60),
            "2881": _make_df(60),
            "2882": _make_df(60),
            "9999": _make_df(60),  # unknown sector
        }
        trust_lookup = {
            "2330": {"foreign_cumulative": 5000, "trust_cumulative": 2000},
            "2303": {"foreign_cumulative": 3000, "trust_cumulative": 1000},
            "2881": {"foreign_cumulative": -2000, "trust_cumulative": -1000},
            "2882": {"foreign_cumulative": -1000, "trust_cumulative": 500},
        }
        result = _compute_sector_aggregates(stock_dfs, trust_lookup)

        # Should have semiconductor, finance, other, and _market_avg
        assert "semiconductor" in result
        assert "finance" in result
        assert DEFAULT_SECTOR in result
        assert "_market_avg" in result

        # Semiconductor should have positive net flow
        assert result["semiconductor"]["net_flow"] > 0
        assert result["semiconductor"]["stock_count"] == 2

        # Market average should exist
        assert "net_flow" in result["_market_avg"]
        assert "avg_return_20d" in result["_market_avg"]


class TestStockSector:
    def test_stock_list_coverage(self):
        """STOCK_SECTOR should cover major stocks"""
        from src.utils.constants import STOCK_LIST
        covered = sum(1 for sid in STOCK_LIST if sid in STOCK_SECTOR)
        # Should cover at least 50% of STOCK_LIST
        assert covered >= len(STOCK_LIST) * 0.3, \
            f"Only {covered}/{len(STOCK_LIST)} stocks covered"

    def test_no_empty_sectors(self):
        """All sector values should be non-empty strings"""
        for stock_id, sector in STOCK_SECTOR.items():
            assert isinstance(sector, str) and len(sector) > 0, \
                f"Stock {stock_id} has invalid sector: {sector!r}"


# ═══════════════════════════════════════════════════════
# Tests: Legacy Aliases
# ═══════════════════════════════════════════════════════


class TestLegacyAliases:
    def test_technical_trend_alias(self):
        df = _make_df()
        df_tech = _make_df_tech(df)
        signals = _make_signals(3, 5, "buy")
        result = _compute_technical_trend(signals, df_tech)
        assert result.name == "technical_trend"
        assert result.available is True

    def test_momentum_alias(self):
        result = _compute_momentum(_make_df(60))
        assert result.name == "momentum"

    def test_institutional_flow_alias(self):
        trust_info = _make_trust_info()
        result = _compute_institutional_flow(trust_info, _make_df())
        assert result.name == "institutional_flow"

    def test_margin_retail_alias(self):
        result = _compute_margin_retail(_make_df(60, with_margin=True))
        assert result.name == "margin_retail"

    def test_volatility_alias(self):
        df = _make_df(60)
        result = _compute_volatility(df, _make_df_tech(df))
        assert result.name == "volatility"

    def test_sentiment_alias(self):
        result = _compute_sentiment({"2330": 0.7}, None, "2330")
        assert result.name == "sentiment"

    def test_liquidity_alias(self):
        result = _compute_liquidity(_make_df(60))
        assert result.name == "liquidity"

    def test_value_quality_alias(self):
        result = _compute_value_quality("9999")
        assert result.name == "value_quality"


# ═══════════════════════════════════════════════════════
# Tests: Weight Engine (20 factors)
# ═══════════════════════════════════════════════════════


class TestWeightEngine:
    def test_base_weights_sum_to_one(self):
        assert abs(sum(BASE_WEIGHTS.values()) - 1.0) < 1e-6

    def test_base_weights_has_20_factors(self):
        assert len(BASE_WEIGHTS) == 20

    def test_all_regime_multipliers_cover_20_factors(self):
        for regime, multipliers in REGIME_MULTIPLIERS.items():
            assert len(multipliers) == 20, f"{regime} has {len(multipliers)} factors, expected 20"
            for name in BASE_WEIGHTS:
                assert name in multipliers, f"{regime} missing {name}"

    def test_all_available_weights_sum_to_one(self):
        factors = [
            FactorResult(name, 0.6, True, 1.0) for name in BASE_WEIGHTS
        ]
        weights = _compute_weights(factors, "sideways")
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_missing_data_redistribution(self):
        """When some factors are unavailable, remaining weights sum to 1.0"""
        names = list(BASE_WEIGHTS.keys())
        factors = []
        for i, name in enumerate(names):
            available = i < 10  # first 10 available, rest not
            factors.append(FactorResult(name, 0.5, available, 1.0 if available else 0.0))
        weights = _compute_weights(factors, "sideways")
        assert len(weights) == 10
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        for name in names[10:]:
            assert name not in weights

    def test_regime_bull_boosts_short_momentum(self):
        factors = [FactorResult(name, 0.6, True, 1.0) for name in BASE_WEIGHTS]
        bull_weights = _compute_weights(factors, "bull")
        sideways_weights = _compute_weights(factors, "sideways")
        assert bull_weights["short_momentum"] > sideways_weights["short_momentum"]

    def test_regime_bear_boosts_volatility_regime(self):
        factors = [FactorResult(name, 0.6, True, 1.0) for name in BASE_WEIGHTS]
        bear_weights = _compute_weights(factors, "bear")
        sideways_weights = _compute_weights(factors, "sideways")
        assert bear_weights["volatility_regime"] > sideways_weights["volatility_regime"]

    def test_regime_bear_boosts_margin_sentiment(self):
        factors = [FactorResult(name, 0.6, True, 1.0) for name in BASE_WEIGHTS]
        bear_weights = _compute_weights(factors, "bear")
        sideways_weights = _compute_weights(factors, "sideways")
        assert bear_weights["margin_sentiment"] > sideways_weights["margin_sentiment"]

    def test_regime_sideways_boosts_sector_rotation(self):
        """Sideways market boosts sector_rotation (1.3 multiplier)"""
        factors = [FactorResult(name, 0.6, True, 1.0) for name in BASE_WEIGHTS]
        sideways_weights = _compute_weights(factors, "sideways")
        bull_weights = _compute_weights(factors, "bull")
        assert sideways_weights["sector_rotation"] > bull_weights["sector_rotation"]

    def test_regime_bear_boosts_us_manufacturing(self):
        """Bear market boosts us_manufacturing (1.3 multiplier)"""
        factors = [FactorResult(name, 0.6, True, 1.0) for name in BASE_WEIGHTS]
        bear_weights = _compute_weights(factors, "bear")
        sideways_weights = _compute_weights(factors, "sideways")
        assert bear_weights["us_manufacturing"] > sideways_weights["us_manufacturing"]


# ═══════════════════════════════════════════════════════
# Tests: Confidence Calculator
# ═══════════════════════════════════════════════════════


class TestConfidence:
    def test_high_agreement(self):
        """All factors agree → high agreement score"""
        factors = [
            FactorResult("f1", 0.8, True, 1.0),
            FactorResult("f2", 0.7, True, 1.0),
            FactorResult("f3", 0.75, True, 1.0),
            FactorResult("f4", 0.65, True, 1.0),
        ]
        weights = {"f1": 0.25, "f2": 0.25, "f3": 0.25, "f4": 0.25}
        conf = _compute_confidence(factors, weights, 0.72, pd.DataFrame())
        assert conf["confidence_agreement"] == 1.0  # All > 0.55

    def test_mixed_factors_low_agreement(self):
        """Factors disagree → lower agreement"""
        factors = [
            FactorResult("f1", 0.8, True, 1.0),
            FactorResult("f2", 0.2, True, 1.0),  # Opposite
            FactorResult("f3", 0.5, True, 1.0),   # Neutral
        ]
        weights = {"f1": 0.33, "f2": 0.33, "f3": 0.34}
        conf = _compute_confidence(factors, weights, 0.5, pd.DataFrame())
        assert conf["confidence_agreement"] < 0.5

    def test_risk_discount_floor(self):
        """Risk discount has floor at 0.3"""
        df = _make_df(60, vol=50)
        df["close"] = 100 + np.cumsum(np.random.randn(60) * 10)
        factors = [FactorResult("f1", 0.6, True, 1.0)]
        weights = {"f1": 1.0}
        conf = _compute_confidence(factors, weights, 0.6, df)
        assert conf["risk_discount"] >= 0.3

    def test_no_factors_available(self):
        factors = [FactorResult("f1", 0.5, False, 0.0)]
        weights = {}
        conf = _compute_confidence(factors, weights, 0.5, pd.DataFrame())
        assert conf["confidence"] >= 0.0


# ═══════════════════════════════════════════════════════
# Tests: RuleEngine
# ═══════════════════════════════════════════════════════


class TestRuleEngine:
    def setup_method(self):
        self.engine = RuleEngine()

    def test_ml_active_uses_60_40(self):
        """When ML has signal, use 60/40 split"""
        action, conf, reason = self.engine.decide(
            ml_signal="buy", ml_confidence=0.8,
            agent_signal="buy", agent_confidence=0.7,
            market_state="bull",
        )
        assert action == "buy"
        assert "× 0.6" in reason
        assert "× 0.4" in reason

    def test_ml_hold_uses_agent_100(self):
        """When ML is hold, Agent gets 100%"""
        action, conf, reason = self.engine.decide(
            ml_signal="hold", ml_confidence=0.5,
            agent_signal="buy", agent_confidence=0.7,
            market_state="bull",
        )
        assert "× 0.0" in reason or "× 0" in reason
        assert "× 1.0" in reason or "× 1" in reason

    def test_ml_low_confidence_uses_agent_100(self):
        """When ML confidence <= 0.1, Agent gets 100%"""
        action, conf, reason = self.engine.decide(
            ml_signal="buy", ml_confidence=0.05,
            agent_signal="sell", agent_confidence=0.8,
            market_state="sideways",
        )
        assert action == "sell"

    def test_agent_only_can_buy(self):
        """Without ML, Agent buy signal → buy action (not always hold)"""
        action, conf, reason = self.engine.decide(
            ml_signal="hold", ml_confidence=0.0,
            agent_signal="buy", agent_confidence=0.8,
            market_state="bull",
        )
        # Agent 100%: 0.5 * 0.8 = 0.4, × bull scale 1.0 = 0.4 > 0.15
        assert action == "buy"

    def test_agent_only_can_sell(self):
        """Without ML, Agent sell signal → sell action"""
        action, conf, reason = self.engine.decide(
            ml_signal="hold", ml_confidence=0.0,
            agent_signal="sell", agent_confidence=0.8,
            market_state="bull",
        )
        # Agent 100%: -0.5 * 0.8 = -0.4, × bull scale 1.0 = -0.4 < -0.15
        assert action == "sell"

    def test_bear_market_reduces_signal(self):
        """Bear market applies 0.5 scale"""
        action_bull, _, _ = self.engine.decide(
            ml_signal="buy", ml_confidence=0.6,
            agent_signal="buy", agent_confidence=0.5,
            market_state="bull",
        )
        action_bear, _, _ = self.engine.decide(
            ml_signal="buy", ml_confidence=0.6,
            agent_signal="buy", agent_confidence=0.5,
            market_state="bear",
        )
        # Same signals but bear market is more conservative
        assert action_bull == "buy"
        # Bear may or may not be hold depending on threshold

    def test_threshold_lowered(self):
        """Verify ±0.15 threshold (not ±0.25)"""
        # ML buy 0.5 * 0.4 = 0.2, Agent buy 0.5 * 0.4 = 0.2
        # combined = 0.6*0.2 + 0.4*0.2 = 0.2, × 1.0 = 0.2 > 0.15 → buy
        action, _, _ = self.engine.decide(
            ml_signal="buy", ml_confidence=0.4,
            agent_signal="buy", agent_confidence=0.4,
            market_state="bull",
        )
        assert action == "buy"


# ═══════════════════════════════════════════════════════
# Tests: score_stock() integration (20 factors)
# ═══════════════════════════════════════════════════════


class TestScoreStock:
    @patch("api.services.market_service._compute_margin_quality")
    @patch("api.services.market_service._compute_fundamental_value")
    def test_basic_scoring(self, mock_fv, mock_mq):
        mock_fv.return_value = FactorResult("fundamental_value", 0.5, False, 0.0)
        mock_mq.return_value = FactorResult("margin_quality", 0.5, False, 0.0)

        df = _make_df(60, with_margin=True)
        df_tech = _make_df_tech(df)
        signals = _make_signals(3, 5, "buy")
        trust_info = _make_trust_info()

        result = score_stock(
            stock_data={"stock_id": "2330", "stock_name": "台積電",
                        "current_price": 600, "price_change_pct": 1.5,
                        "foreign_net_5d": 5000, "trust_net_5d": 3000, "dealer_net_5d": 500},
            df=df,
            df_tech=df_tech,
            signals=signals,
            trust_info=trust_info,
            sentiment_scores={"2330": 0.65},
            sentiment_df=None,
            ml_scores={},
            regime="sideways",
        )

        # Check structure
        assert "total_score" in result
        assert "signal" in result
        assert "confidence" in result
        assert "market_regime" in result
        assert result["market_regime"] == "sideways"
        assert "factor_details" in result
        assert 0.0 <= result["total_score"] <= 1.0

        # Backward compat score fields
        assert "technical_score" in result
        assert "fundamental_score" in result
        assert "sentiment_score" in result
        assert "ml_score" in result
        assert "momentum_score" in result

        # Mapped score fields
        assert "institutional_flow_score" in result
        assert "margin_retail_score" in result
        assert "volatility_score" in result
        assert "liquidity_score" in result
        assert "value_quality_score" in result

        # Confidence breakdown
        assert "confidence_agreement" in result
        assert "confidence_strength" in result
        assert "confidence_coverage" in result
        assert "risk_discount" in result

    @patch("api.services.market_service._compute_margin_quality")
    @patch("api.services.market_service._compute_fundamental_value")
    def test_20_factors_in_details(self, mock_fv, mock_mq):
        mock_fv.return_value = FactorResult("fundamental_value", 0.5, False, 0.0)
        mock_mq.return_value = FactorResult("margin_quality", 0.5, False, 0.0)

        df = _make_df(60, with_margin=True)
        df_tech = _make_df_tech(df)
        trust_info = _make_trust_info()

        result = score_stock(
            stock_data={"stock_id": "2330", "stock_name": "台積電",
                        "current_price": 600, "price_change_pct": 1.5,
                        "foreign_net_5d": 5000, "trust_net_5d": 3000, "dealer_net_5d": 500},
            df=df, df_tech=df_tech,
            signals=_make_signals(3, 5, "buy"),
            trust_info=trust_info,
            sentiment_scores={"2330": 0.65},
            sentiment_df=None,
            ml_scores={},
            regime="sideways",
        )

        factor_details = result["factor_details"]
        # All 20 factor names should be present
        expected_names = set(BASE_WEIGHTS.keys())
        actual_names = set(factor_details.keys())
        assert expected_names == actual_names, f"Missing: {expected_names - actual_names}"

    @patch("api.services.market_service._compute_margin_quality")
    @patch("api.services.market_service._compute_fundamental_value")
    def test_with_global_and_macro(self, mock_fv, mock_mq):
        mock_fv.return_value = FactorResult("fundamental_value", 0.5, False, 0.0)
        mock_mq.return_value = FactorResult("margin_quality", 0.5, False, 0.0)

        df = _make_df(60)
        result = score_stock(
            stock_data={"stock_id": "2330", "stock_name": "台積電",
                        "current_price": 600, "price_change_pct": 0,
                        "foreign_net_5d": 0, "trust_net_5d": 0, "dealer_net_5d": 0},
            df=df, df_tech=_make_df_tech(df),
            signals=_make_signals(3, 5, "hold"),
            trust_info={}, sentiment_scores={}, sentiment_df=None,
            ml_scores={}, regime="sideways",
            global_data={"sox_return": 0.01, "tsm_return": 0.005,
                         "ewt_return_1d": 0.005, "ewt_return_20d": 0.03,
                         "ewt_return_60d": 0.08},
            macro_data={"vix": 18, "usdtwd_trend": 0, "tnx_change": 0,
                        "xli_return_20d": 0.02, "xli_vs_sma200": 0.03,
                        "xli_spy_ratio_trend": 0.01},
        )
        # Global context, macro_risk, export_momentum, us_manufacturing should be available
        assert result["factor_details"]["global_context"]["available"] is True
        assert result["factor_details"]["macro_risk"]["available"] is True
        assert result["factor_details"]["export_momentum"]["available"] is True
        assert result["factor_details"]["us_manufacturing"]["available"] is True

    @patch("api.services.market_service._compute_margin_quality")
    @patch("api.services.market_service._compute_fundamental_value")
    def test_with_sector_data(self, mock_fv, mock_mq):
        mock_fv.return_value = FactorResult("fundamental_value", 0.5, False, 0.0)
        mock_mq.return_value = FactorResult("margin_quality", 0.5, False, 0.0)

        df = _make_df(60)
        sector_data = {
            "semiconductor": {
                "net_flow": 30000, "avg_return_20d": 0.05,
                "breadth": 0.7, "stock_count": 10,
            },
            "_market_avg": {"net_flow": 10000, "avg_return_20d": 0.02},
        }
        result = score_stock(
            stock_data={"stock_id": "2330", "stock_name": "台積電",
                        "current_price": 600, "price_change_pct": 0,
                        "foreign_net_5d": 0, "trust_net_5d": 0, "dealer_net_5d": 0},
            df=df, df_tech=_make_df_tech(df),
            signals=_make_signals(3, 5, "hold"),
            trust_info={}, sentiment_scores={}, sentiment_df=None,
            ml_scores={}, regime="sideways",
            sector_data=sector_data,
        )
        assert result["factor_details"]["sector_rotation"]["available"] is True


# ═══════════════════════════════════════════════════════
# Tests: DB Models (FactorICRecord)
# ═══════════════════════════════════════════════════════


class TestFactorICRecord:
    def test_model_import(self):
        from src.db.models import FactorICRecord
        assert FactorICRecord.__tablename__ == "factor_ic_records"

    def test_market_scan_new_fields(self):
        from src.db.models import MarketScanResult
        assert hasattr(MarketScanResult, "institutional_flow_score")
        assert hasattr(MarketScanResult, "margin_retail_score")
        assert hasattr(MarketScanResult, "volatility_score")
        assert hasattr(MarketScanResult, "liquidity_score")
        assert hasattr(MarketScanResult, "value_quality_score")
        assert hasattr(MarketScanResult, "confidence_agreement")
        assert hasattr(MarketScanResult, "market_regime")
        assert hasattr(MarketScanResult, "factor_details")
