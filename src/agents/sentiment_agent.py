"""情緒面分析 Agent — 結構化萃取版

統一管線使用 narrative_agent.extract_sentiment() 進行情緒萃取。
此 Agent 保留為向後相容接口，將請求委派給 narrative_agent。
"""

import logging

from src.agents.base import (
    AgentMessage, AgentRole, BaseAgent, MarketContext, Signal,
)

logger = logging.getLogger(__name__)


class SentimentAgent(BaseAgent):
    """情緒面分析 Agent — 委派給 narrative_agent.extract_sentiment()"""

    def __init__(self):
        super().__init__(AgentRole.SENTIMENT)

    async def analyze(self, context: MarketContext) -> AgentMessage:
        """分析市場情緒 — 委派給結構化萃取"""
        try:
            from src.agents.narrative_agent import extract_sentiment

            # Map context data to extract_sentiment params
            result = await extract_sentiment(
                stock_id=context.stock_id,
                sentiment_df=None,
                trust_info=context.fundamental_summary or {},
                global_data=None,
            )

            # Map sentiment_score (0-1) to signal
            score = result.get("sentiment_score", 0.5)
            if score > 0.75:
                signal = Signal.STRONG_BUY
            elif score > 0.6:
                signal = Signal.BUY
            elif score < 0.25:
                signal = Signal.STRONG_SELL
            elif score < 0.4:
                signal = Signal.SELL
            else:
                signal = Signal.HOLD

            confidence = abs(score - 0.5) * 2  # 0-1 range
            themes = result.get("key_themes", [])
            reasoning = f"情緒分數 {score:.2f}; 主題: {', '.join(themes)}" if themes else f"情緒分數 {score:.2f}"

            return AgentMessage(
                sender=self.role,
                content=result,
                signal=signal,
                confidence=min(confidence, 1.0),
                reasoning=reasoning,
            )

        except Exception as e:
            self.logger.warning("Sentiment extraction failed: %s", e)
            return AgentMessage(
                sender=self.role,
                content={"signal": "hold", "confidence": 0.3, "reasoning": f"萃取失敗: {e}"},
                signal=Signal.HOLD,
                confidence=0.3,
                reasoning=f"情緒萃取失敗: {e}",
            )
