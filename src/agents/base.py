"""Agent 訊息格式與資料結構"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AgentRole(str, Enum):
    """Agent 角色"""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"
    QUANT = "quant"
    RESEARCHER = "researcher"
    TRADER = "trader"
    RISK = "risk"
    ORCHESTRATOR = "orchestrator"


class Signal(str, Enum):
    """交易訊號"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class AgentMessage:
    """Agent 間通訊訊息"""
    sender: AgentRole
    content: dict[str, Any]
    signal: Signal | None = None
    confidence: float = 0.0  # 0.0 ~ 1.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketContext:
    """市場環境快照"""
    stock_id: str
    current_price: float
    date: str  # YYYY-MM-DD
    technical_summary: dict[str, Any] = field(default_factory=dict)
    sentiment_summary: dict[str, Any] = field(default_factory=dict)
    fundamental_summary: dict[str, Any] = field(default_factory=dict)
    model_predictions: dict[str, Any] = field(default_factory=dict)
    position: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeDecision:
    """最終交易決策"""
    stock_id: str
    action: str  # "buy", "sell", "hold"
    quantity: int = 0
    price_limit: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size: float = 0.0  # 倉位比例 0.0 ~ 1.0
    confidence: float = 0.0
    reasoning: str = ""
    approved_by_risk: bool = False
    risk_notes: str = ""
    analyst_reports: list[AgentMessage] = field(default_factory=list)
    researcher_report: AgentMessage | None = None
    timestamp: datetime = field(default_factory=datetime.now)
