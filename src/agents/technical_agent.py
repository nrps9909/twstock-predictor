"""技術面分析 Agent

分析技術指標並生成交易觀點。使用 Claude Haiku（快速回應）。
"""

import logging

from src.agents.base import (
    AgentMessage, AgentRole, BaseAgent, MarketContext, Signal,
)
from src.utils.config import settings

logger = logging.getLogger(__name__)

TECHNICAL_PROMPT = """你是專業的台股技術分析師。分析以下 {stock_id} 的技術指標，給出交易建議。

當前價格: {price}
日期: {date}

技術指標：
{indicators}

請用 JSON 格式回覆：
{{
    "trend": "uptrend|downtrend|sideways",
    "signal": "strong_buy|buy|hold|sell|strong_sell",
    "confidence": 0.0-1.0,
    "key_levels": {{"support": float, "resistance": float}},
    "reasoning": "你的分析理由（2-3 句）",
    "indicators_summary": {{
        "moving_averages": "描述",
        "momentum": "描述",
        "volatility": "描述"
    }}
}}

只回傳 JSON。"""


class TechnicalAgent(BaseAgent):
    """技術面分析 Agent"""

    def __init__(self):
        super().__init__(AgentRole.TECHNICAL)
        self.api_key = settings.ANTHROPIC_API_KEY

    async def analyze(self, context: MarketContext) -> AgentMessage:
        """分析技術指標"""
        indicators = context.technical_summary

        # 格式化指標文字
        indicators_text = "\n".join(
            f"- {k}: {v}" for k, v in indicators.items()
        )

        prompt = self._format_prompt(
            TECHNICAL_PROMPT,
            stock_id=context.stock_id,
            price=context.current_price,
            date=context.date,
            indicators=indicators_text,
        )

        # 呼叫 LLM
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
            self.logger.warning("未設定 ANTHROPIC_API_KEY，使用規則引擎 fallback")
            return self._rule_based_analysis()

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
            return self._rule_based_analysis()

    def _rule_based_analysis(self) -> dict:
        """規則引擎 fallback（不需 LLM）"""
        return {
            "trend": "sideways",
            "signal": "hold",
            "confidence": 0.3,
            "key_levels": {"support": 0, "resistance": 0},
            "reasoning": "LLM 不可用，使用預設 hold 訊號",
            "indicators_summary": {},
        }
