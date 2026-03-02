"""情緒面分析 Agent

分析社群輿論與新聞情緒。使用 Claude Sonnet（深度分析）。
"""

import logging

from src.agents.base import (
    AgentMessage, AgentRole, BaseAgent, MarketContext, Signal,
)
from src.utils.config import settings

logger = logging.getLogger(__name__)

SENTIMENT_PROMPT = """你是台股市場情緒分析專家。分析以下 {stock_id} 的社群與新聞情緒，給出交易觀點。

當前價格: {price}
日期: {date}

情緒資料：
{sentiment_data}

請用 JSON 格式回覆：
{{
    "overall_sentiment": "very_bullish|bullish|neutral|bearish|very_bearish",
    "signal": "strong_buy|buy|hold|sell|strong_sell",
    "confidence": 0.0-1.0,
    "reasoning": "你的分析理由（2-3 句）",
    "key_themes": ["主題1", "主題2"],
    "contrarian_indicator": true/false,
    "sentiment_details": {{
        "ptt": "描述",
        "news": "描述",
        "institutional": "描述"
    }}
}}

注意：
- 極端情緒可能是反向指標（contrarian_indicator）
- 區分短期恐慌 vs 長期趨勢變化
- 考慮訊息來源的可信度

只回傳 JSON。"""


class SentimentAgent(BaseAgent):
    """情緒面分析 Agent"""

    def __init__(self):
        super().__init__(AgentRole.SENTIMENT)
        self.api_key = settings.ANTHROPIC_API_KEY

    async def analyze(self, context: MarketContext) -> AgentMessage:
        """分析市場情緒"""
        sentiment_data = context.sentiment_summary

        sentiment_text = "\n".join(
            f"- {k}: {v}" for k, v in sentiment_data.items()
        )

        prompt = self._format_prompt(
            SENTIMENT_PROMPT,
            stock_id=context.stock_id,
            price=context.current_price,
            date=context.date,
            sentiment_data=sentiment_text,
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
        """呼叫 Claude Sonnet（深度分析）"""
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
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=30,
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
