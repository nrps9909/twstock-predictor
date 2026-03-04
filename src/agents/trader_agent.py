"""交易員 Agent — 最終交易決策

根據研究員報告做出具體的交易決策（買/賣/持有、數量、價格）。
使用 Claude Sonnet。
"""

import logging

from src.agents.base import (
    AgentMessage, AgentRole, BaseAgent, MarketContext, Signal, TradeDecision,
)
from src.utils.llm_client import call_claude, parse_json_response

logger = logging.getLogger(__name__)

TRADER_PROMPT = """你是專業的台股交易員。根據研究報告做出具體交易決策。

股票: {stock_id}
當前價格: {price}
日期: {date}

== 研究員報告 ==
{researcher_report}

== 當前倉位 ==
{position_info}

== 可用資金 ==
{available_capital}

請做出具體交易決策，用 JSON 格式回覆：
{{
    "action": "buy|sell|hold",
    "position_size": 0.0-1.0,
    "entry_reason": "進場理由",
    "stop_loss_pct": 0.0-0.1,
    "take_profit_pct": 0.0-0.2,
    "time_horizon": "days|weeks",
    "confidence": 0.0-1.0,
    "reasoning": "完整交易邏輯（3-5 句）"
}}

交易規則：
1. 單一個股倉位不超過總資金 20%
2. 停損不得超過 -8%
3. 風險報酬比至少 2:1
4. 不確定時選擇 hold

只回傳 JSON。"""


class TraderAgent(BaseAgent):
    """交易員 Agent — 做出最終交易決策"""

    def __init__(self):
        super().__init__(AgentRole.TRADER)

    async def analyze(self, context: MarketContext) -> AgentMessage:
        """基於 context 做出交易決策（不含研究報告的簡化版）"""
        return await self.make_decision(context, researcher_report=None)

    async def make_decision(
        self,
        context: MarketContext,
        researcher_report: AgentMessage | None = None,
        available_capital: float = 1_000_000,
    ) -> AgentMessage:
        """根據研究員報告做出交易決策"""
        # 格式化研究報告
        report_text = "（無研究報告）"
        if researcher_report:
            report_text = (
                f"訊號: {researcher_report.signal.value if researcher_report.signal else 'N/A'}\n"
                f"信心: {researcher_report.confidence:.0%}\n"
                f"分析: {researcher_report.reasoning}\n"
                f"詳細: {researcher_report.content}"
            )

        # 倉位資訊
        position = context.position
        position_text = (
            f"持股: {position.get('quantity', 0)} 股\n"
            f"成本: {position.get('avg_cost', 0)}\n"
            f"未實現損益: {position.get('unrealized_pnl', 0):.2%}"
        ) if position else "空倉"

        prompt = self._format_prompt(
            TRADER_PROMPT,
            stock_id=context.stock_id,
            price=context.current_price,
            date=context.date,
            researcher_report=report_text,
            position_info=position_text,
            available_capital=f"${available_capital:,.0f}",
        )

        result = await self._call_llm(prompt)

        # 轉換為 TradeDecision
        action = result.get("action", "hold")
        signal_map = {"buy": Signal.BUY, "sell": Signal.SELL, "hold": Signal.HOLD}

        return AgentMessage(
            sender=self.role,
            content=result,
            signal=signal_map.get(action, Signal.HOLD),
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
            metadata={
                "position_size": result.get("position_size", 0),
                "stop_loss_pct": result.get("stop_loss_pct", 0.05),
                "take_profit_pct": result.get("take_profit_pct", 0.1),
            },
        )

    def create_trade_decision(
        self,
        context: MarketContext,
        trader_msg: AgentMessage,
        analyst_reports: list[AgentMessage],
        researcher_report: AgentMessage | None = None,
    ) -> TradeDecision:
        """從 Agent 訊息建構 TradeDecision"""
        meta = trader_msg.metadata
        action = trader_msg.content.get("action", "hold")
        position_size = meta.get("position_size", 0)
        stop_loss_pct = meta.get("stop_loss_pct", 0.05)
        take_profit_pct = meta.get("take_profit_pct", 0.1)

        return TradeDecision(
            stock_id=context.stock_id,
            action=action,
            position_size=position_size,
            stop_loss=context.current_price * (1 - stop_loss_pct) if action == "buy" else None,
            take_profit=context.current_price * (1 + take_profit_pct) if action == "buy" else None,
            confidence=trader_msg.confidence,
            reasoning=trader_msg.reasoning,
            analyst_reports=analyst_reports,
            researcher_report=researcher_report,
        )

    async def _call_llm(self, prompt: str) -> dict:
        """呼叫 Claude Sonnet（透過 claude -p CLI）"""
        try:
            text = await call_claude(prompt, model="claude-sonnet-4-6", timeout=60)
            return parse_json_response(text)
        except Exception as e:
            self.logger.exception("LLM 呼叫失敗 (%s)", type(e).__name__)
            return {"action": "hold", "confidence": 0.3, "reasoning": f"LLM 錯誤 ({type(e).__name__}): {e}"}
