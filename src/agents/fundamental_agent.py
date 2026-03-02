"""基本面分析 Agent

分析法人買賣超、融資融券、營收等基本面資料。使用 Claude Haiku。
"""

import logging

from src.agents.base import (
    AgentMessage, AgentRole, BaseAgent, MarketContext, Signal,
)
from src.utils.config import settings

logger = logging.getLogger(__name__)

FUNDAMENTAL_PROMPT = """你是台股基本面分析師。分析以下 {stock_id} 的法人與基本面資料，給出交易觀點。

當前價格: {price}
日期: {date}

基本面資料：
{fundamental_data}

請用 JSON 格式回覆：
{{
    "institutional_flow": "net_buying|net_selling|neutral",
    "signal": "strong_buy|buy|hold|sell|strong_sell",
    "confidence": 0.0-1.0,
    "reasoning": "你的分析理由（2-3 句）",
    "key_factors": ["因素1", "因素2"],
    "flow_details": {{
        "foreign": "描述外資動向",
        "trust": "描述投信動向",
        "dealer": "描述自營商動向",
        "margin": "描述融資融券狀況"
    }}
}}

只回傳 JSON。"""


class FundamentalAgent(BaseAgent):
    """基本面分析 Agent"""

    def __init__(self):
        super().__init__(AgentRole.FUNDAMENTAL)
        self.api_key = settings.ANTHROPIC_API_KEY

    async def analyze(self, context: MarketContext) -> AgentMessage:
        """分析基本面"""
        fundamental_data = context.fundamental_summary

        data_text = "\n".join(
            f"- {k}: {v}" for k, v in fundamental_data.items()
        )

        prompt = self._format_prompt(
            FUNDAMENTAL_PROMPT,
            stock_id=context.stock_id,
            price=context.current_price,
            date=context.date,
            fundamental_data=data_text,
        )

        result = await self._call_llm(prompt)

        signal_map = {
            "strong_buy": Signal.STRONG_BUY,
            "buy": Signal.BUY,
            "hold": Signal.HOLD,
            "sell": Signal.SELL,
            "strong_sell": Signal.STRONG_SELL,
        }

        return AgentMessage(
            sender=self.role,
            content=result,
            signal=signal_map.get(result.get("signal", "hold"), Signal.HOLD),
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
        )

    async def _call_llm(self, prompt: str) -> dict:
        """呼叫 Claude Haiku"""
        import json
        import httpx

        if not self.api_key:
            return {"signal": "hold", "confidence": 0.3, "reasoning": "LLM 不可用"}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-haiku-4-5-20251001",
                        "max_tokens": 512,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["content"][0]["text"].strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                return json.loads(text)
        except Exception as e:
            self.logger.error("LLM 呼叫失敗: %s", e)
            return {"signal": "hold", "confidence": 0.3, "reasoning": f"LLM 錯誤: {e}"}
