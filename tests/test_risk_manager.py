"""風險管理器測試"""

import pytest

from src.risk.manager import RiskManager, TrailingStopState


class TestTrailingStopState:
    def test_init(self):
        ts = TrailingStopState(
            entry_price=100.0, highest_price=100.0,
            atr=3.0, multiplier=2.5, current_stop=92.5,
        )
        assert ts.trailing_distance == 7.5
        assert ts.current_stop == 92.5
        assert not ts.triggered

    def test_update_moves_up(self):
        ts = TrailingStopState(
            entry_price=100.0, highest_price=100.0,
            atr=3.0, multiplier=2.0, current_stop=94.0,
        )
        ts.update(105.0)
        assert ts.highest_price == 105.0
        assert ts.current_stop == 99.0  # 105 - 2*3

    def test_update_no_down_movement(self):
        ts = TrailingStopState(
            entry_price=100.0, highest_price=105.0,
            atr=3.0, multiplier=2.0, current_stop=99.0,
        )
        # 價格回落不應降低停損
        ts.update(102.0)
        assert ts.current_stop == 99.0

    def test_trigger(self):
        ts = TrailingStopState(
            entry_price=100.0, highest_price=100.0,
            atr=3.0, multiplier=2.0, current_stop=94.0,
        )
        assert not ts.check_trigger(95.0)
        assert ts.check_trigger(93.0)
        assert ts.triggered


class TestRiskManager:
    @pytest.fixture
    def rm(self):
        return RiskManager(
            max_position_pct=0.20,
            max_portfolio_risk=0.02,
            max_drawdown_limit=0.15,
        )

    def test_kelly_position(self, rm):
        pos = rm.kelly_position(win_rate=0.6, avg_win=0.05, avg_loss=0.03)
        assert 0 < pos <= 0.20

    def test_kelly_zero_loss(self, rm):
        pos = rm.kelly_position(win_rate=0.6, avg_win=0.05, avg_loss=0.0)
        assert pos == 0.0

    def test_atr_stop_loss(self, rm):
        sl = rm.atr_stop_loss(100.0, atr=3.0, multiplier=2.0)
        assert sl == 94.0

    def test_atr_take_profit(self, rm):
        tp = rm.atr_take_profit(100.0, atr=3.0, multiplier=4.0)
        assert tp == 112.0

    def test_trailing_stop_lifecycle(self, rm):
        state = rm.init_trailing_stop("2330", 100.0, 3.0, 2.0)
        assert state.current_stop == 94.0

        stop, triggered = rm.update_trailing_stop("2330", 105.0, 103.0)
        assert stop == 99.0
        assert not triggered

        stop, triggered = rm.update_trailing_stop("2330", 105.0, 98.0)
        assert triggered

        rm.close_trailing_stop("2330")
        assert rm.get_trailing_stop_state("2330") is None

    def test_position_size_by_risk(self, rm):
        shares = rm.position_size_by_risk(
            portfolio_value=10_000_000,
            entry_price=100.0,
            stop_loss_price=95.0,
        )
        assert shares > 0
        assert shares % 1000 == 0  # 整張

    def test_circuit_breaker(self, rm):
        assert not rm.is_circuit_breaker_active()

        # 模擬大回撤
        rm.update_equity(1_000_000)
        rm.update_equity(800_000)  # -20%
        assert rm.is_circuit_breaker_active()

        rm.reset_circuit_breaker()
        assert not rm.is_circuit_breaker_active()

    def test_hard_risk_check_circuit_breaker(self, rm):
        rm._circuit_breaker_active = True
        passed, reason = rm.hard_risk_check("buy", "2330", 1_000_000, 0.10)
        assert not passed
        assert "熔斷" in reason

    def test_hard_risk_check_position_limit(self, rm):
        passed, reason = rm.hard_risk_check("buy", "2330", 1_000_000, 0.30)
        assert not passed
        assert "倉位超限" in reason

    def test_hard_risk_check_trailing_stop(self, rm):
        state = rm.init_trailing_stop("2330", 100.0, 3.0, 2.0)
        state.triggered = True
        passed, reason = rm.hard_risk_check("buy", "2330", 1_000_000, 0.10)
        assert not passed
        assert "Trailing Stop" in reason

    def test_hard_risk_check_pass(self, rm):
        passed, reason = rm.hard_risk_check("buy", "2330", 1_000_000, 0.10)
        assert passed
