"""Agent 分析 API"""

import asyncio
from fastapi import APIRouter, HTTPException

from src.utils.constants import STOCK_LIST
from src.agents.base import MarketContext
from src.agents.orchestrator import AgentOrchestrator
from api.schemas.agent import AgentAnalyzeRequest, AnalystReport, TradeDecisionResponse

router = APIRouter(prefix="/api/stocks", tags=["agent"])

# 全域 orchestrator 實例（保留記憶）
_orchestrator: AgentOrchestrator | None = None


def _get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


@router.post("/{stock_id}/agent/analyze", response_model=TradeDecisionResponse)
async def agent_analyze(stock_id: str, req: AgentAnalyzeRequest):
    """執行 Multi-Agent 分析"""
    if stock_id not in STOCK_LIST:
        raise HTTPException(404, f"股票 {stock_id} 不在支援清單中")

    context = MarketContext(
        stock_id=stock_id,
        current_price=req.current_price,
        date=req.date,
    )

    orchestrator = _get_orchestrator()

    try:
        decision = await orchestrator.run_analysis(
            context=context,
            available_capital=req.available_capital,
            ml_signal=req.ml_signal,
            ml_confidence=req.ml_confidence,
            market_state=req.market_state,
        )
    except Exception as e:
        raise HTTPException(500, f"Agent 分析失敗: {e}")

    # 轉換
    analyst_reports = []
    for msg in decision.analyst_reports:
        analyst_reports.append(AnalystReport(
            role=msg.sender.value,
            signal=msg.signal.value if msg.signal else None,
            confidence=msg.confidence,
            reasoning=msg.reasoning,
            content=msg.content,
        ))

    researcher = None
    if decision.researcher_report:
        r = decision.researcher_report
        researcher = AnalystReport(
            role=r.sender.value,
            signal=r.signal.value if r.signal else None,
            confidence=r.confidence,
            reasoning=r.reasoning,
            content=r.content,
        )

    return TradeDecisionResponse(
        stock_id=decision.stock_id,
        action=decision.action,
        confidence=decision.confidence,
        position_size=decision.position_size,
        reasoning=decision.reasoning,
        approved_by_risk=decision.approved_by_risk,
        risk_notes=decision.risk_notes,
        analyst_reports=analyst_reports,
        researcher_report=researcher,
    )
