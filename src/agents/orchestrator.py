"""Orchestrator — Agent 流程控制（DAG）

架構原則：LLM 是研究助手，不是交易員。ML 做預測，規則做風控。

執行順序：
1. 4 個分析 Agent 並行 → [技術, 情緒, 基本面, 量化]（LLM 提供觀點）
2. 研究員 Agent 彙整（多空辯論）→ 僅供參考
3. 規則引擎決策（ML 信號為主，Agent 為輔）
4. 硬性風控（LLM 無法覆蓋）
5. 記憶更新
"""

import asyncio
import logging
from datetime import date

import numpy as np

from src.agents.base import (
    AgentMessage, AgentRole, MarketContext, Signal, TradeDecision,
)
from src.agents.technical_agent import TechnicalAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.fundamental_agent import FundamentalAgent
from src.agents.quant_agent import QuantAgent
from src.agents.researcher_agent import ResearcherAgent
from src.agents.trader_agent import TraderAgent
from src.agents.risk_agent import RiskAgent, RiskLimits
from src.agents.memory import AgentMemorySystem
from src.risk.manager import RiskManager

logger = logging.getLogger(__name__)


class RuleEngine:
    """規則引擎 — ML 信號為主，Agent 觀點為輔

    核心原則：
    - ML 模型信號佔 70% 權重，Agent 觀點佔 30%
    - Agent 只能提供「建議」，不能直接「決策」
    - 硬性風控規則無法被任何來源覆蓋
    """

    ML_WEIGHT = 0.7
    AGENT_WEIGHT = 0.3

    # 信號映射
    SIGNAL_SCORE = {
        "strong_buy": 1.0,
        "buy": 0.5,
        "hold": 0.0,
        "sell": -0.5,
        "strong_sell": -1.0,
    }

    def decide(
        self,
        ml_signal: str,
        ml_confidence: float,
        agent_signal: str,
        agent_confidence: float,
        market_state: str | None = None,
    ) -> tuple[str, float, str]:
        """規則引擎決策

        Args:
            ml_signal: ML 模型信號 ("buy", "sell", "hold")
            ml_confidence: ML 模型信心度 (0-1)
            agent_signal: Agent 建議信號
            agent_confidence: Agent 信心度
            market_state: HMM 市場狀態 ("bull", "bear", "sideways")

        Returns:
            (final_action, final_confidence, reasoning)
        """
        ml_score = self.SIGNAL_SCORE.get(ml_signal, 0.0) * ml_confidence
        agent_score = self.SIGNAL_SCORE.get(agent_signal, 0.0) * agent_confidence

        combined = self.ML_WEIGHT * ml_score + self.AGENT_WEIGHT * agent_score

        # 市場狀態調整
        state_scale = 1.0
        if market_state == "bear":
            state_scale = 0.5  # 熊市中降低激進程度
        elif market_state == "sideways":
            state_scale = 0.7

        adjusted = combined * state_scale

        # 決策閾值
        if adjusted > 0.25:
            action = "buy"
        elif adjusted < -0.25:
            action = "sell"
        else:
            action = "hold"

        confidence = min(abs(adjusted), 1.0)

        # 建構推理說明
        reasoning = (
            f"規則引擎: ML({ml_signal} {ml_confidence:.0%}) × {self.ML_WEIGHT} "
            f"+ Agent({agent_signal} {agent_confidence:.0%}) × {self.AGENT_WEIGHT} "
            f"= {combined:.3f}"
        )
        if market_state:
            reasoning += f" | 市場狀態={market_state} (scale={state_scale})"
        reasoning += f" → {action} ({confidence:.0%})"

        return action, confidence, reasoning


class AgentOrchestrator:
    """Agent 流程控制器

    架構改進：LLM Agent 降級為「顧問」角色，
    最終決策由規則引擎（ML 信號 + 硬性風控）決定。
    """

    def __init__(
        self,
        risk_limits: RiskLimits | None = None,
        session_factory=None,
        risk_manager: RiskManager | None = None,
    ):
        # 分析 Agents（提供觀點，非決策）
        self.technical = TechnicalAgent()
        self.sentiment = SentimentAgent()
        self.fundamental = FundamentalAgent()
        self.quant = QuantAgent()

        # 研究員 Agent（綜合辯論，僅供參考）
        self.researcher = ResearcherAgent()
        # 交易員 Agent（降級為輔助角色）
        self.trader = TraderAgent()
        # 風控 Agent（規則檢查）
        self.risk = RiskAgent(limits=risk_limits)

        # 規則引擎（新核心）
        self.rule_engine = RuleEngine()

        # Meta-Labeler（可選，由外部注入）
        self.meta_labeler = None

        # 硬性風控（LLM 無法覆蓋）
        self.risk_manager = risk_manager or RiskManager()

        # 記憶系統
        self.memory = AgentMemorySystem(session_factory)

        # 狀態
        self.last_decision: TradeDecision | None = None
        self._previous_market_state: str | None = None
        self._positions: dict[str, dict] = {}  # stock_id → position info
        self._pending_reduce_orders: list[dict] = []  # Phase 0.5 減倉指令

    async def run_analysis(
        self,
        context: MarketContext,
        available_capital: float = 1_000_000,
        ml_signal: str = "hold",
        ml_confidence: float = 0.5,
        market_state: str | None = None,
    ) -> TradeDecision:
        """執行完整分析流程

        Args:
            context: 市場環境
            available_capital: 可用資金
            ml_signal: ML 模型信號（從 EnsemblePredictor 取得）
            ml_confidence: ML 模型信心度
            market_state: HMM 市場狀態

        Returns:
            TradeDecision — 含所有 Agent 分析結果
        """
        logger.info("=== 開始分析 %s ===", context.stock_id)

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
                        "Phase 0.5: 行情轉場 %s→%s (severity=%.1f), 減倉 %d 檔: %s",
                        self._previous_market_state, market_state,
                        transition.severity, len(reduce_orders), reduce_orders,
                    )
        self._previous_market_state = market_state

        # 1. 並行執行 4 個分析 Agent（LLM 提供觀點）
        logger.info("Phase 1: 分析師觀點收集...")
        analyst_tasks = [
            self.technical.analyze(context),
            self.sentiment.analyze(context),
            self.fundamental.analyze(context),
            self.quant.analyze(context),
        ]
        analyst_results = await asyncio.gather(*analyst_tasks, return_exceptions=True)

        # 過濾成功的結果
        analyst_messages: list[AgentMessage] = []
        for result in analyst_results:
            if isinstance(result, AgentMessage):
                analyst_messages.append(result)
                logger.info(
                    "  [%s] 觀點=%s, 信心=%.0f%%",
                    result.sender.value,
                    result.signal.value if result.signal else "N/A",
                    result.confidence * 100,
                )
            elif isinstance(result, Exception):
                logger.error("  分析師錯誤: %s", result)

        # 2. 研究員彙整（多空辯論）— 僅供參考
        logger.info("Phase 2: 研究員辯論（僅供參考）...")
        researcher_msg = await self.researcher.analyze(context, analyst_messages)
        logger.info(
            "  研究建議: %s (信心 %.0f%%)",
            researcher_msg.signal.value if researcher_msg.signal else "N/A",
            researcher_msg.confidence * 100,
        )

        # 3. 規則引擎決策（ML 信號為主，Agent 為輔）
        logger.info("Phase 3: 規則引擎決策 (ML=%.0f%%, Agent=%.0f%%)...",
                     RuleEngine.ML_WEIGHT * 100, RuleEngine.AGENT_WEIGHT * 100)

        agent_signal = researcher_msg.signal.value if researcher_msg.signal else "hold"
        agent_confidence = researcher_msg.confidence

        final_action, final_confidence, reasoning = self.rule_engine.decide(
            ml_signal=ml_signal,
            ml_confidence=ml_confidence,
            agent_signal=agent_signal,
            agent_confidence=agent_confidence,
            market_state=market_state,
        )
        logger.info("  規則引擎: %s (%.0f%%) — %s", final_action, final_confidence * 100, reasoning)

        # 倉位大小（若有 meta-labeler 則用其校準結果，否則線性縮放）
        position_size = final_confidence * 0.20
        if hasattr(self, 'meta_labeler') and self.meta_labeler is not None:
            try:
                meta_prob = self.meta_labeler.predict_proba(
                    self.meta_labeler.prepare_meta_features(
                        primary_pred=np.array([ml_confidence if final_action == "buy" else -ml_confidence]),
                        signal_strength=np.array([final_confidence]),
                    )
                )
                position_size = float(self.meta_labeler.bet_size(meta_prob)[0])
                reasoning += f" | meta_size={position_size:.1%}"
            except Exception:
                pass  # fallback to linear scaling

        # 若 Phase 0.5 產生了減倉指令且目標在其中，強制 sell
        if self._pending_reduce_orders:
            for order in self._pending_reduce_orders:
                if order.get("stock_id") == context.stock_id:
                    final_action = "sell"
                    position_size = order.get("reduce_pct", 0.5) * position_size
                    reasoning += f" | Phase 0.5 強制減倉 {order.get('reduce_pct', 0.5):.0%}"
                    break

        decision = TradeDecision(
            stock_id=context.stock_id,
            action=final_action,
            confidence=final_confidence,
            position_size=position_size,
            reasoning=reasoning,
            analyst_reports=analyst_messages,
            researcher_report=researcher_msg,
        )

        # 4. 硬性風控（LLM 無法覆蓋）
        logger.info("Phase 4: 硬性風控審核...")

        # 4a. RiskManager 硬性檢查
        passed, hard_reason = self.risk_manager.hard_risk_check(
            action=final_action,
            stock_id=context.stock_id,
            portfolio_value=available_capital,
            position_size_pct=decision.position_size,
        )
        if not passed:
            decision.action = "hold"
            decision.approved_by_risk = False
            decision.risk_notes = f"硬性風控否決: {hard_reason}"
            logger.warning("  硬性風控否決: %s", hard_reason)
        else:
            # 4b. RiskAgent 軟性檢查
            approved, reason, decision = self.risk.evaluate_trade(
                decision, context, available_capital
            )
            logger.info("  風控結果: %s — %s", "核准" if approved else "否決", reason)

        # 5. 記憶更新
        self._update_memory(context, decision, analyst_messages, researcher_msg)

        self.last_decision = decision
        logger.info("=== 分析完成: %s %s ===", decision.action, context.stock_id)

        return decision

    def _update_memory(
        self,
        context: MarketContext,
        decision: TradeDecision,
        analysts: list[AgentMessage],
        researcher: AgentMessage,
    ):
        """更新記憶系統"""
        # 短期記憶
        self.memory.short_term.add(
            context.date,
            {
                "stock_id": context.stock_id,
                "price": context.current_price,
                "action": decision.action,
                "confidence": decision.confidence,
                "approved": decision.approved_by_risk,
            },
        )

        # 如果有實際交易動作 → 記錄到情境記憶
        if decision.action != "hold" and decision.approved_by_risk:
            reasoning = {
                "technical": analysts[0].content if len(analysts) > 0 else None,
                "sentiment": analysts[1].content if len(analysts) > 1 else None,
                "fundamental": analysts[2].content if len(analysts) > 2 else None,
                "quant": analysts[3].content if len(analysts) > 3 else None,
                "researcher": researcher.content,
                "trader_reasoning": decision.reasoning,
                "risk": decision.risk_notes,
                "market_snapshot": {
                    "price": context.current_price,
                    "date": context.date,
                },
            }
            self.memory.episodic.record_trade(
                stock_id=context.stock_id,
                trade_date=context.date,
                action=decision.action,
                price=context.current_price,
                reasoning=reasoning,
            )

    def get_analysis_summary(self) -> dict:
        """取得最近一次分析的摘要"""
        if not self.last_decision:
            return {"status": "no_analysis"}

        d = self.last_decision
        summary = {
            "stock_id": d.stock_id,
            "action": d.action,
            "confidence": d.confidence,
            "approved": d.approved_by_risk,
            "risk_notes": d.risk_notes,
            "reasoning": d.reasoning,
            "analyst_signals": {
                msg.sender.value: {
                    "signal": msg.signal.value if msg.signal else None,
                    "confidence": msg.confidence,
                }
                for msg in d.analyst_reports
            },
        }
        if d.researcher_report:
            summary["researcher"] = {
                "signal": d.researcher_report.signal.value if d.researcher_report.signal else None,
                "confidence": d.researcher_report.confidence,
            }
        return summary
