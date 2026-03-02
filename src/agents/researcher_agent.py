"""研究員 Agent — 多輪辯論

多輪 Bull vs Bear 辯論機制：
Round 1: 各自陳述（bull_case + bear_case）
Round 2-N: 互相反駁（rebuttal）
Early exit: 若信號差 < CONVERGENCE_THRESHOLD
Final: 綜合辯論歷史產出共識

預設 2 rounds → 5 LLM calls (vs 舊版 1 call)。
"""

import logging

from src.agents.base import (
    AgentMessage, AgentRole, BaseAgent, MarketContext, Signal,
)
from src.utils.config import settings

logger = logging.getLogger(__name__)

MAX_DEBATE_ROUNDS = 3
CONVERGENCE_THRESHOLD = 0.15

# ── Prompts ──

BULL_CASE_PROMPT = """你是一位看多分析師，負責為 {stock_id} 建構最強的看多論述。

價格: {price}
日期: {date}

分析師觀點摘要:
{analyst_reports}

請用 JSON 格式回覆你的看多論述:
{{"arguments": ["看多理由1", "看多理由2", "看多理由3"], "strength": 0.0-1.0, "signal": 0.5-1.0}}

只回傳 JSON。"""

BEAR_CASE_PROMPT = """你是一位看空分析師，負責為 {stock_id} 建構最強的看空論述。

價格: {price}
日期: {date}

分析師觀點摘要:
{analyst_reports}

請用 JSON 格式回覆你的看空論述:
{{"arguments": ["看空理由1", "看空理由2", "看空理由3"], "strength": 0.0-1.0, "signal": -1.0 到 -0.5}}

只回傳 JSON。"""

REBUTTAL_PROMPT = """你是一位{role}分析師。對方提出了以下論述:
{opponent_case}

你先前的論述:
{my_case}

請反駁對方的論點，並強化你的立場。用 JSON 格式回覆:
{{"rebuttal": ["反駁1", "反駁2"], "updated_arguments": ["更新理由1", "更新理由2"], "strength": 0.0-1.0, "signal": {signal_range}}}

只回傳 JSON。"""

SYNTHESIS_PROMPT = """你是資深研究員，負責綜合以下多輪多空辯論。

股票: {stock_id}, 價格: {price}, 日期: {date}

辯論歷史:
{debate_history}

請綜合所有輪次的辯論，給出最終判斷。用 JSON 格式回覆:
{{"signal": "strong_buy|buy|hold|sell|strong_sell", "confidence": 0.0-1.0, "reasoning": "綜合判斷理由（3-5 句）", "key_risks": ["風險1", "風險2"], "catalysts": ["催化劑1", "催化劑2"]}}

只回傳 JSON。"""


class ResearcherAgent(BaseAgent):
    """研究員 Agent — 多輪辯論"""

    def __init__(self, debate_rounds: int = 2):
        super().__init__(AgentRole.RESEARCHER)
        self.api_key = settings.ANTHROPIC_API_KEY
        self.debate_rounds = min(debate_rounds, MAX_DEBATE_ROUNDS)

    async def analyze(
        self,
        context: MarketContext,
        analyst_messages: list[AgentMessage] | None = None,
    ) -> AgentMessage:
        """多輪辯論分析"""
        if analyst_messages is None:
            analyst_messages = []

        reports_text = self._format_analyst_reports(analyst_messages)
        debate_history = []

        # Round 1: 各自陳述
        bull_case = await self._generate_case(
            "bull", context, reports_text,
        )
        bear_case = await self._generate_case(
            "bear", context, reports_text,
        )
        debate_history.append({"round": 1, "bull": bull_case, "bear": bear_case})

        # 檢查是否已經收斂
        if self._check_convergence(bull_case, bear_case):
            self.logger.info("Round 1 已收斂，跳過後續辯論")
        else:
            # Rounds 2-N: 互相反駁
            for round_num in range(2, self.debate_rounds + 1):
                bull_rebuttal = await self._generate_rebuttal(
                    "看多", bull_case, bear_case, "0.5 到 1.0",
                )
                bear_rebuttal = await self._generate_rebuttal(
                    "看空", bear_case, bull_case, "-1.0 到 -0.5",
                )
                debate_history.append({
                    "round": round_num,
                    "bull": bull_rebuttal,
                    "bear": bear_rebuttal,
                })
                bull_case = bull_rebuttal
                bear_case = bear_rebuttal

                if self._check_convergence(bull_case, bear_case):
                    self.logger.info("Round %d 收斂", round_num)
                    break

        # Final: 綜合辯論歷史
        result = await self._synthesize(context, debate_history)

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
            metadata={
                "analyst_count": len(analyst_messages),
                "debate_rounds": len(debate_history),
            },
        )

    async def _generate_case(
        self, side: str, context: MarketContext, reports_text: str,
    ) -> dict:
        """Round 1: 生成看多/看空論述"""
        prompt_template = BULL_CASE_PROMPT if side == "bull" else BEAR_CASE_PROMPT
        prompt = self._format_prompt(
            prompt_template,
            stock_id=context.stock_id,
            price=context.current_price,
            date=context.date,
            analyst_reports=reports_text or "（無分析師報告）",
        )
        return await self._call_llm(prompt)

    async def _generate_rebuttal(
        self, role: str, my_case: dict, opponent_case: dict, signal_range: str,
    ) -> dict:
        """Rounds 2-N: 生成反駁"""
        prompt = REBUTTAL_PROMPT.format(
            role=role,
            opponent_case=str(opponent_case),
            my_case=str(my_case),
            signal_range=signal_range,
        )
        return await self._call_llm(prompt)

    async def _synthesize(self, context: MarketContext, debate_history: list) -> dict:
        """Final: 綜合辯論歷史"""
        history_text = ""
        for round_data in debate_history:
            r = round_data["round"]
            history_text += f"\n=== Round {r} ===\n"
            history_text += f"看多: {round_data['bull']}\n"
            history_text += f"看空: {round_data['bear']}\n"

        prompt = SYNTHESIS_PROMPT.format(
            stock_id=context.stock_id,
            price=context.current_price,
            date=context.date,
            debate_history=history_text,
        )
        return await self._call_llm(prompt)

    def _check_convergence(self, bull_case: dict, bear_case: dict) -> bool:
        """檢查是否收斂（雙方信號差 < threshold）

        bull signal 為正 (0.5~1.0), bear signal 為負 (-1.0~-0.5)。
        比較原始值差距，不取 abs()。
        """
        bull_signal = float(bull_case.get("signal", 0.7))
        bear_signal = float(bear_case.get("signal", -0.7))
        diff = abs(bull_signal - bear_signal)
        return diff < CONVERGENCE_THRESHOLD

    def _format_analyst_reports(self, messages: list[AgentMessage]) -> str:
        text = ""
        for msg in messages:
            text += (
                f"\n[{msg.sender.value}] "
                f"訊號={msg.signal.value if msg.signal else 'N/A'}, "
                f"信心={msg.confidence:.0%}\n"
                f"  分析: {msg.reasoning}\n"
            )
        return text

    async def _call_llm(self, prompt: str) -> dict:
        """呼叫 Claude Sonnet"""
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
