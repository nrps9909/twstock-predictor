"""投資組合管理

Position 追蹤、最多 5 檔持倉、產業分散限制
"""

import logging
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """單一持倉"""
    stock_id: str
    quantity: int  # 股數
    avg_cost: float  # 平均成本
    entry_date: date
    stop_loss: float | None = None
    take_profit: float | None = None
    sector: str = ""

    @property
    def market_value(self) -> float:
        """市值（需要外部提供現價時使用 avg_cost）"""
        return self.quantity * self.avg_cost

    def unrealized_pnl(self, current_price: float) -> float:
        """未實現損益"""
        return (current_price - self.avg_cost) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """未實現損益比例"""
        if self.avg_cost == 0:
            return 0.0
        return (current_price - self.avg_cost) / self.avg_cost


class PortfolioManager:
    """投資組合管理器"""

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        max_positions: int = 5,
        max_sector_pct: float = 0.40,  # 單一產業最大佔比
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions
        self.max_sector_pct = max_sector_pct
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[dict] = []

    @property
    def total_value(self) -> float:
        """組合總價值（用成本估算，實際應用現價）"""
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def position_count(self) -> int:
        return len(self.positions)

    def can_open_position(self, stock_id: str, sector: str = "") -> tuple[bool, str]:
        """是否可開新倉"""
        if stock_id in self.positions:
            return False, "已有該股持倉"
        if self.position_count >= self.max_positions:
            return False, f"持倉已滿 ({self.position_count}/{self.max_positions})"

        # 產業分散檢查
        if sector:
            sector_value = sum(
                p.market_value for p in self.positions.values()
                if p.sector == sector
            )
            if sector_value / max(self.total_value, 1) > self.max_sector_pct:
                return False, f"產業 {sector} 佔比過高"

        return True, "OK"

    def open_position(
        self,
        stock_id: str,
        quantity: int,
        price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        sector: str = "",
    ) -> bool:
        """建立新倉"""
        cost = quantity * price
        # 台股手續費
        commission = max(cost * 0.001425 * 0.28, 20)

        if cost + commission > self.cash:
            logger.warning("資金不足: 需 %.0f, 可用 %.0f", cost + commission, self.cash)
            return False

        self.cash -= cost + commission
        self.positions[stock_id] = Position(
            stock_id=stock_id,
            quantity=quantity,
            avg_cost=price,
            entry_date=date.today(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            sector=sector,
        )
        logger.info(
            "建倉 %s: %d 股 @ %.2f (手續費 %.0f)",
            stock_id, quantity, price, commission,
        )
        return True

    def close_position(self, stock_id: str, price: float) -> dict | None:
        """平倉"""
        if stock_id not in self.positions:
            logger.warning("無 %s 持倉", stock_id)
            return None

        pos = self.positions.pop(stock_id)
        revenue = pos.quantity * price
        commission = max(revenue * 0.001425 * 0.28, 20)
        tax = revenue * 0.003  # 證交稅

        self.cash += revenue - commission - tax
        pnl = (price - pos.avg_cost) * pos.quantity - commission - tax

        trade_record = {
            "stock_id": stock_id,
            "entry_date": str(pos.entry_date),
            "exit_date": str(date.today()),
            "entry_price": pos.avg_cost,
            "exit_price": price,
            "quantity": pos.quantity,
            "pnl": pnl,
            "pnl_pct": (price - pos.avg_cost) / pos.avg_cost,
            "commission": commission,
            "tax": tax,
        }
        self.closed_trades.append(trade_record)

        logger.info(
            "平倉 %s: %d 股 @ %.2f | PnL: %.0f (%.2f%%)",
            stock_id, pos.quantity, price, pnl, trade_record["pnl_pct"] * 100,
        )
        return trade_record

    def check_stop_loss_take_profit(self, prices: dict[str, float]) -> list[str]:
        """檢查停損/止盈觸發

        Args:
            prices: {stock_id: current_price}

        Returns:
            list of stock_ids to close
        """
        to_close = []
        for stock_id, pos in self.positions.items():
            price = prices.get(stock_id)
            if price is None:
                continue
            if pos.stop_loss and price <= pos.stop_loss:
                logger.warning("停損觸發 %s: %.2f <= %.2f", stock_id, price, pos.stop_loss)
                to_close.append(stock_id)
            elif pos.take_profit and price >= pos.take_profit:
                logger.info("止盈觸發 %s: %.2f >= %.2f", stock_id, price, pos.take_profit)
                to_close.append(stock_id)
        return to_close

    def get_summary(self) -> dict:
        """組合摘要"""
        return {
            "total_value": self.total_value,
            "cash": self.cash,
            "position_count": self.position_count,
            "positions": {
                sid: {
                    "quantity": p.quantity,
                    "avg_cost": p.avg_cost,
                    "market_value": p.market_value,
                }
                for sid, p in self.positions.items()
            },
            "closed_trades_count": len(self.closed_trades),
            "total_realized_pnl": sum(t["pnl"] for t in self.closed_trades),
        }
