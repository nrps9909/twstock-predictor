"""Walk-Forward 回測框架

Expanding window 策略：
- Window 1: 訓練 1-6 月, 測試第 7 月
- Window 2: 訓練 1-7 月, 測試第 8 月
- ...

指標：MSE, 方向準確率, Sharpe ratio, 最大回撤, 模擬損益（含台股手續費）
"""

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 台股交易成本
COMMISSION_RATE = 0.001425  # 手續費 0.1425%
TAX_RATE = 0.003  # 證交稅 0.3%（賣出時收取）
COMMISSION_DISCOUNT = 0.28  # 券商折扣（通常 2.8 折）


@dataclass
class BacktestMetrics:
    """回測績效指標"""
    total_return: float  # 累積報酬率
    annualized_return: float  # 年化報酬率
    sharpe_ratio: float  # Sharpe ratio（假設無風險利率 1.5%）
    max_drawdown: float  # 最大回撤
    win_rate: float  # 勝率
    direction_accuracy: float  # 方向準確率
    total_trades: int  # 總交易次數
    avg_mse: float  # 平均 MSE
    monthly_returns: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


def calculate_trading_cost(trade_value: float, is_sell: bool = False) -> float:
    """計算台股交易成本

    Args:
        trade_value: 交易金額
        is_sell: 是否為賣出（賣出需付證交稅）
    """
    commission = trade_value * COMMISSION_RATE * COMMISSION_DISCOUNT
    commission = max(commission, 20)  # 最低手續費 20 元
    tax = trade_value * TAX_RATE if is_sell else 0
    return commission + tax


class WalkForwardBacktester:
    """Walk-forward expanding window 回測引擎"""

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        risk_free_rate: float = 0.015,  # 年化無風險利率 1.5%
    ):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        prices: np.ndarray,
        dates: list[date] | None = None,
        threshold: float = 0.01,
    ) -> BacktestMetrics:
        """執行回測

        Args:
            predictions: 模型預測報酬率
            actuals: 實際報酬率
            prices: 實際收盤價
            dates: 交易日期（可選）
            threshold: 交易閾值

        Returns:
            BacktestMetrics
        """
        capital = self.initial_capital
        position = 0  # 持倉股數
        equity_curve = [capital]
        trade_returns = []
        entry_price = 0.0
        total_trades = 0

        for i in range(len(predictions)):
            pred_return = predictions[i]
            actual_price = prices[i]

            # 訊號生成
            if pred_return > threshold and position == 0:
                # 買進
                shares = int(capital * 0.95 / actual_price / 1000) * 1000  # 整張
                if shares > 0:
                    trade_value = shares * actual_price
                    cost = calculate_trading_cost(trade_value, is_sell=False)
                    capital -= trade_value + cost
                    position = shares
                    entry_price = actual_price
                    total_trades += 1

            elif pred_return < -threshold and position > 0:
                # 賣出
                trade_value = position * actual_price
                cost = calculate_trading_cost(trade_value, is_sell=True)
                capital += trade_value - cost
                trade_return = (actual_price - entry_price) / entry_price
                trade_returns.append(trade_return)
                position = 0
                total_trades += 1

            # 記錄權益曲線
            portfolio_value = capital + position * actual_price
            equity_curve.append(portfolio_value)

        # 結算（如有未平倉）
        if position > 0:
            final_value = position * prices[-1]
            cost = calculate_trading_cost(final_value, is_sell=True)
            capital += final_value - cost
            trade_return = (prices[-1] - entry_price) / entry_price
            trade_returns.append(trade_return)
            position = 0

        final_equity = capital
        equity_array = np.array(equity_curve)

        # 計算指標
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        n_days = len(predictions)
        annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # Sharpe ratio
        daily_returns = np.diff(equity_array) / equity_array[:-1]
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            excess_daily = daily_returns - self.risk_free_rate / 252
            sharpe = np.mean(excess_daily) / np.std(excess_daily) * np.sqrt(252)
        else:
            sharpe = 0.0

        # 最大回撤
        peak = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - peak) / peak
        max_drawdown = float(np.min(drawdowns))

        # 方向準確率
        if len(predictions) > 0:
            direction_acc = float(
                np.mean(np.sign(predictions) == np.sign(actuals))
            )
        else:
            direction_acc = 0.0

        # 勝率
        win_rate = float(np.mean(np.array(trade_returns) > 0)) if trade_returns else 0.0

        # MSE
        avg_mse = float(np.mean((predictions - actuals) ** 2))

        # 月報酬（簡化：每 21 天一組）
        monthly_returns = []
        for start in range(0, len(daily_returns), 21):
            chunk = daily_returns[start:start + 21]
            if len(chunk) > 0:
                monthly_returns.append(float(np.prod(1 + chunk) - 1))

        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=float(sharpe),
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            direction_accuracy=direction_acc,
            total_trades=total_trades,
            avg_mse=avg_mse,
            monthly_returns=monthly_returns,
            equity_curve=equity_curve,
        )
