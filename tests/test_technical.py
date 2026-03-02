"""Tests for technical analysis module"""

import pytest
import pandas as pd
import numpy as np

from src.analysis.technical import TechnicalAnalyzer


@pytest.fixture
def sample_data():
    """產生模擬股價資料"""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(1000, 50000, n).astype(float)

    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestTechnicalAnalyzer:
    def setup_method(self):
        self.analyzer = TechnicalAnalyzer()

    def test_compute_all_adds_columns(self, sample_data):
        result = self.analyzer.compute_all(sample_data)
        expected_cols = [
            "sma_5", "sma_20", "sma_60",
            "kd_k", "kd_d", "rsi_14",
            "macd", "macd_signal", "macd_hist",
            "bias_5", "bias_10", "bias_20",
            "bb_upper", "bb_lower",
            "obv", "adx",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_compute_all_preserves_length(self, sample_data):
        result = self.analyzer.compute_all(sample_data)
        assert len(result) == len(sample_data)

    def test_sma_values_reasonable(self, sample_data):
        result = self.analyzer.compute_all(sample_data)
        sma5 = result["sma_5"].dropna()
        # SMA5 應在 close 附近
        diff = (sma5 - result.loc[sma5.index, "close"]).abs()
        assert diff.mean() < 5  # 合理的平均偏差

    def test_rsi_range(self, sample_data):
        result = self.analyzer.compute_all(sample_data)
        rsi = result["rsi_14"].dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_kd_range(self, sample_data):
        result = self.analyzer.compute_all(sample_data)
        k = result["kd_k"].dropna()
        d = result["kd_d"].dropna()
        assert k.min() >= 0
        assert k.max() <= 100
        assert d.min() >= 0
        assert d.max() <= 100

    def test_get_signals(self, sample_data):
        result = self.analyzer.compute_all(sample_data)
        signals = self.analyzer.get_signals(result)
        assert "kd" in signals
        assert "rsi" in signals
        assert "macd" in signals
        assert "summary" in signals
        assert signals["summary"]["signal"] in ("buy", "sell", "hold")

    def test_generate_chart_data(self, sample_data):
        result = self.analyzer.compute_all(sample_data)
        chart = self.analyzer.generate_chart_data(result)
        assert "ohlcv" in chart
        assert "ma_lines" in chart
        assert "kd" in chart
        assert "rsi" in chart
        assert "macd" in chart
