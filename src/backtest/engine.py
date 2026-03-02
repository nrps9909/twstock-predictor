"""事件驅動回測引擎

含台股交易成本（手續費 0.1425% × 折扣 + 證交稅 0.3%）
"""

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.risk.portfolio import PortfolioManager

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回測設定"""
    initial_capital: float = 1_000_000
    max_positions: int = 5
    commission_rate: float = 0.001425
    commission_discount: float = 0.28
    tax_rate: float = 0.003
    slippage_pct: float = 0.001  # 滑價


def get_universe_at_date(
    target_date: date,
    active_stocks: dict[str, str] | None = None,
) -> list[str]:
    """取得特定日期的可投資股票宇宙（含已下市股票修正）

    用於回測時避免存活者偏誤：
    - 包含在 target_date 時仍在市的股票
    - 排除已在 target_date 之前下市的股票

    Args:
        target_date: 回測日期
        active_stocks: 當前活躍股票 dict (stock_id: name)

    Returns:
        可投資的 stock_id 列表
    """
    from src.utils.constants import STOCK_LIST, DELISTED_STOCKS

    if active_stocks is None:
        active_stocks = STOCK_LIST

    universe = list(active_stocks.keys())

    # 加入在 target_date 時仍在市的已下市股票
    for stock_id, info in DELISTED_STOCKS.items():
        delist_date_str = info.get("delist_date")
        if delist_date_str is None:
            continue
        delist_date = date.fromisoformat(delist_date_str)
        if delist_date > target_date:
            # 在 target_date 時仍在市
            if stock_id not in universe:
                universe.append(stock_id)
        else:
            # 已下市，從宇宙中移除
            if stock_id in universe:
                universe.remove(stock_id)

    return sorted(universe)


class BacktestEngine:
    """事件驅動回測引擎"""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.portfolio = PortfolioManager(
            initial_capital=self.config.initial_capital,
            max_positions=self.config.max_positions,
        )
        self.equity_curve: list[float] = [self.config.initial_capital]
        self.trades: list[dict] = []
        self.daily_returns: list[float] = []

    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> dict:
        """執行回測

        Args:
            signals: DataFrame(date, stock_id, signal, confidence, stop_loss, take_profit)
            prices: DataFrame(date, stock_id, close)

        Returns:
            回測結果 dict
        """
        dates = sorted(signals["date"].unique())

        for trade_date in dates:
            # 當日價格
            day_prices = prices[prices["date"] == trade_date]
            price_dict = dict(zip(day_prices["stock_id"], day_prices["close"]))

            # 檢查停損/止盈
            to_close = self.portfolio.check_stop_loss_take_profit(price_dict)
            for stock_id in to_close:
                if stock_id in price_dict:
                    result = self.portfolio.close_position(stock_id, price_dict[stock_id])
                    if result:
                        self.trades.append(result)

            # 處理當日信號
            day_signals = signals[signals["date"] == trade_date]
            for _, sig in day_signals.iterrows():
                stock_id = sig["stock_id"]
                signal = sig["signal"]
                price = price_dict.get(stock_id)
                if price is None:
                    continue

                # 加入滑價
                if signal == "buy":
                    exec_price = price * (1 + self.config.slippage_pct)
                elif signal == "sell":
                    exec_price = price * (1 - self.config.slippage_pct)
                else:
                    continue

                if signal == "buy":
                    can_open, reason = self.portfolio.can_open_position(stock_id)
                    if can_open:
                        # 計算持股數
                        position_pct = sig.get("position_size", 0.1)
                        invest_amount = self.portfolio.cash * position_pct
                        quantity = int(invest_amount / exec_price / 1000) * 1000
                        if quantity > 0:
                            self.portfolio.open_position(
                                stock_id, quantity, exec_price,
                                stop_loss=sig.get("stop_loss"),
                                take_profit=sig.get("take_profit"),
                            )

                elif signal == "sell" and stock_id in self.portfolio.positions:
                    result = self.portfolio.close_position(stock_id, exec_price)
                    if result:
                        self.trades.append(result)

            # 記錄權益曲線
            portfolio_value = self.portfolio.cash
            for sid, pos in self.portfolio.positions.items():
                p = price_dict.get(sid, pos.avg_cost)
                portfolio_value += pos.quantity * p
            self.equity_curve.append(portfolio_value)

            if len(self.equity_curve) > 1:
                daily_ret = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
                self.daily_returns.append(daily_ret)

        return self._calculate_metrics()

    def _calculate_metrics(self) -> dict:
        """計算績效指標"""
        equity = np.array(self.equity_curve)
        returns = np.array(self.daily_returns) if self.daily_returns else np.array([0])

        total_return = (equity[-1] - equity[0]) / equity[0]
        n_days = len(returns)
        ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # Sharpe
        if np.std(returns) > 0:
            sharpe = (np.mean(returns) - 0.015 / 252) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = float(np.min(dd))

        # Trade stats
        if self.trades:
            wins = [t for t in self.trades if t["pnl"] > 0]
            win_rate = len(wins) / len(self.trades)
            avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
            losses = [t for t in self.trades if t["pnl"] <= 0]
            avg_loss = abs(np.mean([t["pnl_pct"] for t in losses])) if losses else 0
            profit_factor = (
                sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses))
                if losses and sum(t["pnl"] for t in losses) != 0 else float("inf")
            )
        else:
            win_rate = avg_win = avg_loss = 0
            profit_factor = 0

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "sharpe_ratio": float(sharpe),
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "total_trades": len(self.trades),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "final_equity": float(equity[-1]),
            "equity_curve": self.equity_curve,
            "trades": self.trades,
        }
