"""情緒分析模組 — LLM 評分 + 聚合

使用 OpenClaw 或直接呼叫 LLM API 進行繁體中文情緒分析。
每篇文章/留言分類為 利多/利空/中性，並給予 -1.0 ~ 1.0 的情緒分數。
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import date

import httpx
import numpy as np
import pandas as pd

from src.utils.config import settings
from src.utils.llm_client import call_claude_sync, parse_json_response

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """單篇文章情緒評分"""
    label: str  # "bullish", "bearish", "neutral"
    score: float  # -1.0 ~ 1.0
    keywords: list[str]
    confidence: float  # 0.0 ~ 1.0


@dataclass
class DailySentiment:
    """每日聚合情緒"""
    date: date
    avg_score: float
    post_count: int
    bullish_ratio: float
    bearish_ratio: float
    neutral_ratio: float
    weighted_score: float


class SentimentAnalyzer:
    """情緒分析處理器"""

    def __init__(self, api_key: str | None = None, provider: str = "anthropic"):
        self.provider = provider
        if provider == "anthropic":
            self.api_key = api_key or settings.ANTHROPIC_API_KEY
            self.api_url = "https://api.anthropic.com/v1/messages"
        else:
            self.api_key = api_key or settings.OPENAI_API_KEY
            self.api_url = "https://api.openai.com/v1/chat/completions"
        self.client = httpx.Client(timeout=60)

    # ── 單篇分析 ────────────────────────────────────────

    def analyze_text(self, text: str) -> SentimentScore:
        """分析單篇文章/留言的市場情緒

        Returns:
            SentimentScore(label, score, keywords, confidence)
        """
        if not self.api_key:
            # 無 API key 時用簡單規則判斷
            return self._rule_based_analysis(text)

        prompt = (
            "你是台股市場情緒分析專家。分析以下文章的市場情緒。\n"
            "回傳 JSON 格式（不要其他文字）:\n"
            '{"label": "bullish"|"bearish"|"neutral", '
            '"score": -1.0到1.0的浮點數, '
            '"keywords": [最多5個關鍵詞], '
            '"confidence": 0.0到1.0的浮點數}\n\n'
            f"文章:\n{text[:2000]}"
        )

        try:
            if self.provider == "anthropic":
                content = call_claude_sync(prompt, model="claude-sonnet-4-6", timeout=30)
            else:
                resp = self.client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]

            if self.provider == "anthropic":
                result = parse_json_response(content)
            else:
                result = json.loads(content)
            return SentimentScore(
                label=result.get("label", "neutral"),
                score=float(result.get("score", 0)),
                keywords=result.get("keywords", []),
                confidence=float(result.get("confidence", 0.5)),
            )
        except Exception as e:
            logger.error("LLM 情緒分析失敗: %s", e)
            return self._rule_based_analysis(text)

    def _rule_based_analysis(self, text: str) -> SentimentScore:
        """基於規則的簡易情緒分析（備用方案）"""
        bullish_words = [
            "利多", "看多", "看漲", "上漲", "突破", "創新高", "買進", "加碼",
            "營收成長", "獲利", "紅包", "噴", "飆", "強勢", "多頭",
        ]
        bearish_words = [
            "利空", "看空", "看跌", "下跌", "跌破", "崩", "賣出", "減碼",
            "營收衰退", "虧損", "套牢", "弱勢", "空頭", "砍", "出貨",
        ]

        bull_count = sum(1 for w in bullish_words if w in text)
        bear_count = sum(1 for w in bearish_words if w in text)
        total = bull_count + bear_count

        if total == 0:
            return SentimentScore("neutral", 0.0, [], 0.3)

        score = (bull_count - bear_count) / total
        if score > 0.2:
            label = "bullish"
        elif score < -0.2:
            label = "bearish"
        else:
            label = "neutral"

        keywords = [w for w in bullish_words + bearish_words if w in text][:5]
        return SentimentScore(label, score, keywords, 0.5)

    # ── 每日聚合 ────────────────────────────────────────

    def aggregate_daily(
        self, scores: list[dict], target_date: date
    ) -> DailySentiment:
        """將多篇文章的情緒分數聚合為每日情緒

        Args:
            scores: list of {sentiment_score, engagement, sentiment_label}
            target_date: 聚合日期
        """
        if not scores:
            return DailySentiment(
                date=target_date,
                avg_score=0.0,
                post_count=0,
                bullish_ratio=0.0,
                bearish_ratio=0.0,
                neutral_ratio=0.0,
                weighted_score=0.0,
            )

        s_scores = [s.get("sentiment_score", 0) for s in scores]
        engagements = [max(s.get("engagement", 1), 1) for s in scores]
        labels = [s.get("sentiment_label", "neutral") for s in scores]

        total = len(scores)
        bullish_count = sum(1 for l in labels if l == "bullish")
        bearish_count = sum(1 for l in labels if l == "bearish")
        neutral_count = total - bullish_count - bearish_count

        # 加權平均（互動數多的文章權重大）
        weights = np.array(engagements, dtype=float)
        weights /= weights.sum()
        weighted_score = float(np.average(s_scores, weights=weights))

        return DailySentiment(
            date=target_date,
            avg_score=float(np.mean(s_scores)),
            post_count=total,
            bullish_ratio=bullish_count / total,
            bearish_ratio=bearish_count / total,
            neutral_ratio=neutral_count / total,
            weighted_score=weighted_score,
        )

    # ── 情緒移動平均 ────────────────────────────────────

    @staticmethod
    def compute_sentiment_ma(
        daily_scores: list[DailySentiment], window: int = 5
    ) -> pd.DataFrame:
        """計算情緒移動平均線

        Returns:
            DataFrame(date, sentiment_score, sentiment_ma{window})
        """
        if not daily_scores:
            return pd.DataFrame()

        df = pd.DataFrame([asdict(d) for d in daily_scores])
        df[f"sentiment_ma{window}"] = (
            df["weighted_score"].rolling(window=window, min_periods=1).mean()
        )
        df["sentiment_change"] = df["weighted_score"].diff()

        return df.rename(columns={"weighted_score": "sentiment_score"})
