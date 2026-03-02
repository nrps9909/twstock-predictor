"""RuleEngine 決策測試"""

import pytest

from src.agents.orchestrator import RuleEngine


@pytest.fixture
def engine():
    return RuleEngine()


class TestRuleEngine:
    def test_strong_buy_signal(self, engine):
        action, confidence, reasoning = engine.decide(
            ml_signal="buy", ml_confidence=0.9,
            agent_signal="buy", agent_confidence=0.8,
        )
        assert action == "buy"
        assert confidence > 0

    def test_strong_sell_signal(self, engine):
        action, confidence, reasoning = engine.decide(
            ml_signal="sell", ml_confidence=0.9,
            agent_signal="sell", agent_confidence=0.8,
        )
        assert action == "sell"

    def test_conflicting_signals_ml_wins(self, engine):
        """ML 與 Agent 衝突時，ML 權重佔優"""
        action, confidence, reasoning = engine.decide(
            ml_signal="strong_buy", ml_confidence=0.9,
            agent_signal="sell", agent_confidence=0.5,
        )
        # ML 70% * 1.0 * 0.9 = 0.63 vs Agent 30% * -0.5 * 0.5 = -0.075 → net 0.555
        assert action == "buy"

    def test_hold_on_weak_signals(self, engine):
        action, confidence, reasoning = engine.decide(
            ml_signal="hold", ml_confidence=0.5,
            agent_signal="hold", agent_confidence=0.5,
        )
        assert action == "hold"

    def test_bear_market_reduces_confidence(self, engine):
        action_bull, conf_bull, _ = engine.decide(
            ml_signal="buy", ml_confidence=0.7,
            agent_signal="buy", agent_confidence=0.6,
            market_state="bull",
        )
        action_bear, conf_bear, _ = engine.decide(
            ml_signal="buy", ml_confidence=0.7,
            agent_signal="buy", agent_confidence=0.6,
            market_state="bear",
        )
        # 熊市信心度應更低
        assert conf_bear <= conf_bull

    def test_sideways_market(self, engine):
        action, confidence, _ = engine.decide(
            ml_signal="buy", ml_confidence=0.6,
            agent_signal="buy", agent_confidence=0.5,
            market_state="sideways",
        )
        # sideways scale=0.7 應降低信號
        assert confidence < 0.6

    def test_reasoning_contains_info(self, engine):
        _, _, reasoning = engine.decide(
            ml_signal="buy", ml_confidence=0.8,
            agent_signal="hold", agent_confidence=0.5,
            market_state="bull",
        )
        assert "ML" in reasoning
        assert "Agent" in reasoning

    def test_signal_score_mapping(self, engine):
        """強買/強賣信號映射正確"""
        action, _, _ = engine.decide(
            ml_signal="strong_buy", ml_confidence=0.9,
            agent_signal="hold", agent_confidence=0.5,
        )
        assert action == "buy"

        action, _, _ = engine.decide(
            ml_signal="strong_sell", ml_confidence=0.9,
            agent_signal="hold", agent_confidence=0.5,
        )
        assert action == "sell"
