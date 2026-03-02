"""回測引擎測試"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.report import generate_report, generate_comparison_report


@pytest.fixture
def config():
    return BacktestConfig(
        initial_capital=1_000_000,
        max_positions=5,
        commission_rate=0.001425,
        tax_rate=0.003,
        slippage_pct=0.001,
    )


@pytest.fixture
def synthetic_data():
    """合成信號 + 價格資料"""
    n_days = 50
    dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(n_days)]
    stock_id = "2330"

    # 價格：穩定上漲
    prices = pd.DataFrame({
        "date": dates,
        "stock_id": stock_id,
        "close": np.linspace(500, 550, n_days),
    })

    # 信號：第 5 天買進，第 30 天賣出
    signals = pd.DataFrame([
        {"date": dates[5], "stock_id": stock_id, "signal": "buy",
         "confidence": 0.8, "position_size": 0.1,
         "stop_loss": 480.0, "take_profit": 560.0},
        {"date": dates[30], "stock_id": stock_id, "signal": "sell",
         "confidence": 0.7, "position_size": 0.0},
    ])

    return signals, prices


class TestBacktestEngine:
    def test_run_returns_metrics(self, config, synthetic_data):
        engine = BacktestEngine(config)
        signals, prices = synthetic_data
        result = engine.run(signals, prices)

        assert "total_return" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "win_rate" in result
        assert "total_trades" in result
        assert "equity_curve" in result

    def test_buy_sell_generates_trade(self, config):
        """買進後賣出應產生交易紀錄"""
        n_days = 50
        dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(n_days)]
        stock_id = "2330"

        prices = pd.DataFrame({
            "date": dates,
            "stock_id": stock_id,
            "close": np.linspace(100, 110, n_days),  # 穩定上漲
        })

        signals = pd.DataFrame([
            {"date": dates[2], "stock_id": stock_id, "signal": "buy",
             "confidence": 0.9, "position_size": 0.5},
            {"date": dates[40], "stock_id": stock_id, "signal": "sell",
             "confidence": 0.8},
        ])

        engine = BacktestEngine(config)
        result = engine.run(signals, prices)
        assert result["total_trades"] >= 1

    def test_equity_curve_starts_at_capital(self, config, synthetic_data):
        engine = BacktestEngine(config)
        signals, prices = synthetic_data
        result = engine.run(signals, prices)

        assert result["equity_curve"][0] == config.initial_capital

    def test_no_signals_no_trades(self, config):
        engine = BacktestEngine(config)
        dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(10)]
        prices = pd.DataFrame({
            "date": dates, "stock_id": "2330",
            "close": [100.0] * 10,
        })
        # hold 信號不做交易
        signals = pd.DataFrame([{
            "date": dates[0], "stock_id": "2330", "signal": "hold",
            "confidence": 0.5,
        }])
        result = engine.run(signals, prices)
        assert result["total_trades"] == 0

    def test_profit_on_rising_prices(self, config, synthetic_data):
        engine = BacktestEngine(config)
        signals, prices = synthetic_data
        result = engine.run(signals, prices)
        # 穩定上漲應有正收益
        assert result["total_return"] >= 0


class TestReportGeneration:
    def test_generate_report(self, config, synthetic_data):
        engine = BacktestEngine(config)
        signals, prices = synthetic_data
        result = engine.run(signals, prices)
        report = generate_report(result)
        assert "回測績效報告" in report
        assert "Sharpe" in report

    def test_generate_comparison_report(self):
        result_a = {
            "total_return": 0.05, "annualized_return": 0.10,
            "sharpe_ratio": 1.2, "max_drawdown": -0.05,
            "win_rate": 0.55, "total_trades": 10,
            "profit_factor": 1.5, "final_equity": 1_050_000,
        }
        result_b = {
            "total_return": 0.08, "annualized_return": 0.16,
            "sharpe_ratio": 1.8, "max_drawdown": -0.04,
            "win_rate": 0.60, "total_trades": 12,
            "profit_factor": 2.0, "final_equity": 1_080_000,
        }
        report = generate_comparison_report(result_a, result_b)
        assert "比較報告" in report
        assert "Approach A" in report
        assert "Approach B" in report
