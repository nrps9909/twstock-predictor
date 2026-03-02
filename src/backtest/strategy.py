"""策略介面

定義通用策略介面，支援：
- ML 策略（LSTM/XGBoost/TFT ensemble）
- Agent 策略（Multi-Agent 決策）
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """交易信號"""
    date: str
    stock_id: str
    signal: str  # "buy", "sell", "hold"
    confidence: float = 0.5
    position_size: float = 0.1
    stop_loss: float | None = None
    take_profit: float | None = None


class BaseStrategy(ABC):
    """策略基礎介面"""

    @abstractmethod
    def generate_signals(
        self,
        df: pd.DataFrame,
        stock_id: str,
    ) -> list[TradeSignal]:
        """根據資料生成交易信號"""
        ...


class MLStrategy(BaseStrategy):
    """ML 模型策略"""

    def __init__(
        self,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
        position_size: float = 0.15,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
    ):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def generate_signals(
        self,
        df: pd.DataFrame,
        stock_id: str,
    ) -> list[TradeSignal]:
        """根據 ML 預測生成信號

        df 需包含 'date', 'close', 'predicted_return' 欄位
        """
        signals = []
        for _, row in df.iterrows():
            pred = row.get("predicted_return", 0)
            price = row["close"]

            if pred > self.buy_threshold:
                signals.append(TradeSignal(
                    date=str(row["date"]),
                    stock_id=stock_id,
                    signal="buy",
                    confidence=min(abs(pred) / 0.05, 1.0),
                    position_size=self.position_size,
                    stop_loss=price * (1 - self.stop_loss_pct),
                    take_profit=price * (1 + self.take_profit_pct),
                ))
            elif pred < self.sell_threshold:
                signals.append(TradeSignal(
                    date=str(row["date"]),
                    stock_id=stock_id,
                    signal="sell",
                    confidence=min(abs(pred) / 0.05, 1.0),
                ))
            else:
                signals.append(TradeSignal(
                    date=str(row["date"]),
                    stock_id=stock_id,
                    signal="hold",
                    confidence=0.5,
                ))

        return signals

    def signals_to_dataframe(self, signals: list[TradeSignal]) -> pd.DataFrame:
        """將信號列表轉為 DataFrame"""
        return pd.DataFrame([
            {
                "date": s.date,
                "stock_id": s.stock_id,
                "signal": s.signal,
                "confidence": s.confidence,
                "position_size": s.position_size,
                "stop_loss": s.stop_loss,
                "take_profit": s.take_profit,
            }
            for s in signals
        ])


class AgentStrategy(BaseStrategy):
    """Agent 策略 — 使用 multi-agent 系統生成信號"""

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    def generate_signals(
        self,
        df: pd.DataFrame,
        stock_id: str,
    ) -> list[TradeSignal]:
        """使用 Agent 系統（需要 async 呼叫，此處為同步包裝）"""
        import asyncio
        from src.agents.base import MarketContext

        signals = []
        for _, row in df.iterrows():
            if self.orchestrator is None:
                signals.append(TradeSignal(
                    date=str(row["date"]),
                    stock_id=stock_id,
                    signal="hold",
                ))
                continue

            context = MarketContext(
                stock_id=stock_id,
                current_price=row["close"],
                date=str(row["date"]),
            )
            try:
                decision = asyncio.get_event_loop().run_until_complete(
                    self.orchestrator.run_analysis(context)
                )
                signals.append(TradeSignal(
                    date=str(row["date"]),
                    stock_id=stock_id,
                    signal=decision.action,
                    confidence=decision.confidence,
                    position_size=decision.position_size,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                ))
            except Exception as e:
                logger.error("Agent 策略錯誤: %s", e)
                signals.append(TradeSignal(
                    date=str(row["date"]),
                    stock_id=stock_id,
                    signal="hold",
                ))

        return signals
