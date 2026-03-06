"""Orchestrator — 統一分析管線入口

架構原則：
- 20 因子演算法評分是主引擎
- ML 模型是其中一個因子 (7%)
- LLM 只負責情緒萃取 + 敘事生成 (2 次呼叫)
- 硬性風控不可被覆蓋

統一管線 6 階段:
1. 數據收集 (並行)
2. 特徵萃取 (20 因子 + HMM + ML + LLM 情緒)
3. 多因子評分
4. LLM 敘事生成
5. 風險控制 + 部位建議
6. 儲存 + 警報
"""

import asyncio
import logging
from datetime import date

import numpy as np

from src.agents.base import (
    AgentMessage, AgentRole, MarketContext, Signal, TradeDecision,
)
from src.agents.memory import ShortTermMemory
from src.risk.manager import RiskManager

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """統一分析管線入口

    使用統一管線 (StockAnalysisService) 替代原有 4 Agent + 辯論機制。
    保留向後相容接口 (run_analysis → TradeDecision)。
    """

    def __init__(
        self,
        session_factory=None,
        risk_manager: RiskManager | None = None,
    ):
        # Meta-Labeler（可選，由外部注入）
        self.meta_labeler = None

        # 硬性風控（LLM 無法覆蓋）
        self.risk_manager = risk_manager or RiskManager()

        # 記憶系統
        self.memory = ShortTermMemory()

        # 狀態
        self.last_decision: TradeDecision | None = None
        self._previous_market_state: str | None = None
        self._positions: dict[str, dict] = {}
        self._pending_reduce_orders: list[dict] = []

    @staticmethod
    def _emit(queue: asyncio.Queue | None, substep: str, data: dict | None = None) -> None:
        """非阻塞發送進度事件到 queue"""
        if queue is not None:
            try:
                queue.put_nowait({"substep": substep, **(data or {})})
            except asyncio.QueueFull:
                pass

    async def run_analysis(
        self,
        context: MarketContext,
        available_capital: float = 1_000_000,
        ml_signal: str = "hold",
        ml_confidence: float = 0.5,
        market_state: str | None = None,
        progress_queue: asyncio.Queue | None = None,
    ) -> TradeDecision:
        """執行統一 6 階段分析管線

        委派給 StockAnalysisService，結果轉換為 TradeDecision 向後相容。
        """
        logger.info("=== 開始統一管線分析 %s ===", context.stock_id)

        # 0. 硬性風控前置檢查
        if self.risk_manager.is_circuit_breaker_active():
            logger.warning("熔斷中，跳過分析")
            decision = TradeDecision(
                stock_id=context.stock_id,
                action="hold",
                confidence=0.0,
                reasoning="最大回撤熔斷中，禁止一切交易",
                approved_by_risk=False,
                risk_notes="熔斷",
            )
            self.last_decision = decision
            return decision

        # 0.5 行情轉場檢查
        self._pending_reduce_orders = []
        if market_state and self._previous_market_state and market_state != self._previous_market_state:
            from src.models.ensemble import RegimeTransition
            transition = RegimeTransition.from_states(self._previous_market_state, market_state)
            if transition.severity > 0.5:
                reduce_orders = self.risk_manager.regime_transition_check(
                    transition, self._positions,
                )
                if reduce_orders:
                    self._pending_reduce_orders = reduce_orders
                    logger.warning(
                        "行情轉場 %s→%s (severity=%.1f), 減倉 %d 檔",
                        self._previous_market_state, market_state,
                        transition.severity, len(reduce_orders),
                    )
        self._previous_market_state = market_state

        # 統一管線: 委派給 StockAnalysisService
        self._emit(progress_queue, "pipeline_start")
        try:
            from api.services.stock_analysis_service import StockAnalysisService
            service = StockAnalysisService()

            # Collect the final result from SSE stream
            final_result = None
            async for event_str in service.analyze_stock(context.stock_id):
                import json
                try:
                    # Parse SSE data line
                    for line in event_str.strip().split("\n"):
                        if line.startswith("data: "):
                            payload = json.loads(line[6:])
                            if payload.get("status") == "done" and payload.get("phase") == "complete":
                                final_result = payload.get("data", {})
                            # Forward progress
                            self._emit(progress_queue, payload.get("phase", ""), payload)
                except (json.JSONDecodeError, KeyError):
                    pass

            if final_result:
                # Convert AnalysisResult to TradeDecision
                risk_dec = final_result.get("risk_decision", {})
                action = risk_dec.get("action", "hold")
                position_size = risk_dec.get("position_size", 0.0)
                confidence = final_result.get("confidence", 0.5)

                # Apply Phase 0.5 reduce orders
                reasoning = final_result.get("reasoning", "")
                if self._pending_reduce_orders:
                    for order in self._pending_reduce_orders:
                        if order.get("stock_id") == context.stock_id:
                            action = "sell"
                            position_size = order.get("reduce_pct", 0.5) * position_size
                            reasoning += f" | 行情轉場強制減倉"
                            break

                decision = TradeDecision(
                    stock_id=context.stock_id,
                    action=action,
                    confidence=confidence,
                    position_size=position_size,
                    reasoning=reasoning,
                    approved_by_risk=risk_dec.get("approved", True),
                    risk_notes="; ".join(risk_dec.get("risk_notes", [])),
                    stop_loss=risk_dec.get("stop_loss"),
                    take_profit=risk_dec.get("take_profit"),
                )
            else:
                decision = TradeDecision(
                    stock_id=context.stock_id,
                    action="hold",
                    confidence=0.0,
                    reasoning="統一管線未產生結果",
                )

        except Exception as e:
            logger.error("統一管線失敗: %s", e)
            decision = TradeDecision(
                stock_id=context.stock_id,
                action="hold",
                confidence=0.0,
                reasoning=f"統一管線錯誤: {e}",
            )

        # 記憶更新
        self.memory.add(
            context.date,
            {
                "stock_id": context.stock_id,
                "price": context.current_price,
                "action": decision.action,
                "confidence": decision.confidence,
                "approved": decision.approved_by_risk,
            },
        )

        self.last_decision = decision
        logger.info("=== 分析完成: %s %s ===", decision.action, context.stock_id)
        self._emit(progress_queue, "__done__")
        return decision

    def get_analysis_summary(self) -> dict:
        """取得最近一次分析的摘要"""
        if not self.last_decision:
            return {"status": "no_analysis"}

        d = self.last_decision
        return {
            "stock_id": d.stock_id,
            "action": d.action,
            "confidence": d.confidence,
            "approved": d.approved_by_risk,
            "risk_notes": d.risk_notes,
            "reasoning": d.reasoning,
        }
