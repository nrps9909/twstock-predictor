"""Tests for sentiment analysis module"""

import pytest
from datetime import date

from src.analysis.sentiment import SentimentAnalyzer, SentimentScore, DailySentiment


class TestSentimentAnalyzer:
    def setup_method(self):
        # 不帶 API key → 使用 rule-based fallback
        self.analyzer = SentimentAnalyzer(api_key="", provider="anthropic")

    def test_rule_based_bullish(self):
        text = "台積電利多不斷，股價看漲，外資大買，營收創新高！"
        score = self.analyzer.analyze_text(text)
        assert isinstance(score, SentimentScore)
        assert score.label == "bullish"
        assert score.score > 0

    def test_rule_based_bearish(self):
        text = "利空消息傳出，股價崩跌，法人出貨，投信砍倉"
        score = self.analyzer.analyze_text(text)
        assert score.label == "bearish"
        assert score.score < 0

    def test_rule_based_neutral(self):
        text = "今天天氣很好，適合去公園散步"
        score = self.analyzer.analyze_text(text)
        assert score.label == "neutral"
        assert score.score == 0

    def test_aggregate_daily_empty(self):
        result = self.analyzer.aggregate_daily([], date.today())
        assert isinstance(result, DailySentiment)
        assert result.post_count == 0
        assert result.avg_score == 0.0

    def test_aggregate_daily_normal(self):
        scores = [
            {"sentiment_score": 0.5, "engagement": 10, "sentiment_label": "bullish"},
            {"sentiment_score": -0.3, "engagement": 5, "sentiment_label": "bearish"},
            {"sentiment_score": 0.1, "engagement": 3, "sentiment_label": "neutral"},
        ]
        result = self.analyzer.aggregate_daily(scores, date.today())
        assert result.post_count == 3
        assert result.bullish_ratio == pytest.approx(1 / 3)
        assert result.bearish_ratio == pytest.approx(1 / 3)

    def test_sentiment_ma(self):
        daily_data = [
            DailySentiment(
                date=date(2024, 1, i + 1),
                avg_score=0.1 * i,
                post_count=5,
                bullish_ratio=0.6,
                bearish_ratio=0.2,
                neutral_ratio=0.2,
                weighted_score=0.1 * i,
            )
            for i in range(10)
        ]
        result = SentimentAnalyzer.compute_sentiment_ma(daily_data, window=3)
        assert "sentiment_ma3" in result.columns
        assert len(result) == 10
