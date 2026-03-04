"""基本面分析 Agent

分析法人買賣超、融資融券、營收等基本面資料。使用 Claude Haiku。
特別著重外資/投信動向分析 — 台股最重要的籌碼指標。
"""

import logging

from src.agents.base import (
    AgentMessage, AgentRole, BaseAgent, MarketContext, Signal,
)
from src.utils.llm_client import call_claude, parse_json_response

logger = logging.getLogger(__name__)

FUNDAMENTAL_PROMPT = """你是台股基本面與籌碼面分析師。分析以下 {stock_id} 的法人動向與籌碼資料，給出交易觀點。

## 重要背景知識
在台股中，法人（特別是外資和投信）的買賣超是最重要的領先指標之一：
- **外資**：佔台股成交量 25-30%，持股佔比更高。外資連續買超通常代表國際資金看好，是強力多方訊號。
- **投信**：代表國內法人基金，選股精準。投信連續買超代表國內專業法人認同，特別在中小型股更具指標性。
- **自營商**：含自行買賣 + 避險，短線波動較大，但大量買超仍有參考價值。
- **融資/融券**：散戶指標。融資增加=散戶看多，融券增加=散戶看空。法人與散戶方向背離時，通常跟隨法人。

## 分析框架
1. 三大法人近期動向趨勢（加速/減緩）
2. 外資與投信是否方向一致（同步買超=強力訊號）
3. 法人 vs 散戶（融資）是否背離（背離時跟法人）
4. 融券餘額變化（軋空風險）
5. 連續買賣超天數（動能持續性）

當前價格: {price}
日期: {date}

籌碼面資料：
{fundamental_data}

請用 JSON 格式回覆：
{{
    "institutional_flow": "net_buying|net_selling|neutral",
    "signal": "strong_buy|buy|hold|sell|strong_sell",
    "confidence": 0.0-1.0,
    "reasoning": "你的分析理由（2-3 句，必須引用具體數字）",
    "key_factors": ["因素1", "因素2", "因素3"],
    "flow_details": {{
        "foreign": "外資動向描述（含具體數字）",
        "trust": "投信動向描述（含具體數字）",
        "dealer": "自營商動向描述",
        "margin": "融資融券描述",
        "divergence": "法人散戶背離分析"
    }}
}}

注意：
- 若外資與投信同步大量買超 → 考慮 strong_buy
- 若法人持續買超但融資也大增（過熱） → 降低信心度
- 若無籌碼資料 → confidence 設為 0.3 以下，signal 為 hold

只回傳 JSON。"""


class FundamentalAgent(BaseAgent):
    """基本面分析 Agent"""

    def __init__(self):
        super().__init__(AgentRole.FUNDAMENTAL)

    async def analyze(self, context: MarketContext) -> AgentMessage:
        """分析基本面"""
        fundamental_data = context.fundamental_summary

        data_text = "\n".join(
            f"- {k}: {v}" for k, v in fundamental_data.items()
        )
        if not data_text.strip():
            data_text = "無可用的法人籌碼資料"

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
        """呼叫 Claude Haiku（透過 claude -p CLI）"""
        try:
            text = await call_claude(prompt, model="claude-haiku-4-5-20251001", timeout=90)
            return parse_json_response(text)
        except Exception as e:
            self.logger.exception("LLM 呼叫失敗 (%s)", type(e).__name__)
            return {"signal": "hold", "confidence": 0.3, "reasoning": f"LLM 錯誤 ({type(e).__name__}): {e}"}
