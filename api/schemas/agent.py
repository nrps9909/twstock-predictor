"""Agent 相關 Pydantic 模型"""

from pydantic import BaseModel


class AgentAnalyzeRequest(BaseModel):
    current_price: float
    date: str
    ml_signal: str = "hold"
    ml_confidence: float = 0.5
    available_capital: float = 1_000_000
    market_state: str | None = None


class AnalystReport(BaseModel):
    role: str
    signal: str | None = None
    confidence: float = 0.0
    reasoning: str = ""
    content: dict = {}


class TradeDecisionResponse(BaseModel):
    stock_id: str
    action: str
    confidence: float
    position_size: float = 0.0
    reasoning: str = ""
    approved_by_risk: bool = False
    risk_notes: str = ""
    analyst_reports: list[AnalystReport] = []
    researcher_report: AnalystReport | None = None
