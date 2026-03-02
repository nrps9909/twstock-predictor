"""Ensemble 預測器 + HMM 測試"""

import numpy as np
import pytest

from src.models.ensemble import (
    EnsemblePredictor,
    HMMStateDetector,
    MarketState,
    PredictionResult,
    StackingEnsemble,
)


@pytest.fixture
def ensemble():
    return EnsemblePredictor(lstm_weight=0.6, xgb_weight=0.4)


@pytest.fixture
def bull_returns():
    """牛市報酬率序列"""
    np.random.seed(42)
    return np.random.normal(0.002, 0.01, 200)


@pytest.fixture
def bear_returns():
    """熊市報酬率序列"""
    np.random.seed(42)
    return np.random.normal(-0.003, 0.02, 200)


class TestEnsemblePredictor:
    def test_predict_returns_result(self, ensemble):
        lstm_pred = np.array([0.01])
        xgb_pred = np.array([0.02])
        result = ensemble.predict(lstm_pred, xgb_pred, 100.0)
        assert isinstance(result, PredictionResult)
        assert result.signal in ("buy", "sell", "hold")
        assert 0 <= result.signal_strength <= 1

    def test_predict_weighted_average(self, ensemble):
        lstm_pred = np.array([0.10])
        xgb_pred = np.array([0.00])
        result = ensemble.predict(lstm_pred, xgb_pred, 100.0)
        expected_return = 0.6 * 0.10 + 0.4 * 0.00
        assert abs(result.predicted_returns[0] - expected_return) < 1e-6

    def test_predict_buy_signal(self, ensemble):
        # 強烈看漲
        result = ensemble.predict(
            np.array([0.05]), np.array([0.05]), 100.0,
            recent_returns_std=0.02,
        )
        assert result.signal == "buy"

    def test_predict_sell_signal(self, ensemble):
        result = ensemble.predict(
            np.array([-0.05]), np.array([-0.05]), 100.0,
            recent_returns_std=0.02,
        )
        assert result.signal == "sell"

    def test_predict_confidence_interval(self, ensemble):
        result = ensemble.predict(np.array([0.01]), np.array([0.01]), 100.0)
        assert (result.confidence_lower < result.predicted_prices).all()
        assert (result.confidence_upper > result.predicted_prices).all()

    def test_update_weights(self, ensemble):
        lstm_errors = np.random.normal(0, 0.01, 50)
        xgb_errors = np.random.normal(0, 0.02, 50)  # XGB 誤差更大
        ensemble.update_weights(lstm_errors, xgb_errors)
        # LSTM 誤差小 → 權重更高
        assert ensemble.lstm_weight > ensemble.xgb_weight

    def test_hmm_bear_market_signal_scaling(self, ensemble, bear_returns):
        ensemble.fit_hmm(bear_returns)
        result = ensemble.predict(
            np.array([0.02]), np.array([0.02]), 100.0,
            recent_returns=bear_returns,
            recent_returns_std=bear_returns.std(),
        )
        # 熊市中信號強度應被縮放
        assert result.signal_strength < 1.0


class TestHMMStateDetector:
    def test_fit_unfitted(self):
        hmm = HMMStateDetector()
        assert not hmm.is_fitted

    def test_fit_with_data(self, bull_returns):
        hmm = HMMStateDetector()
        hmm.fit(bull_returns)
        # hmmlearn 可能未安裝
        if hmm.is_fitted:
            state = hmm.predict_state(bull_returns)
            assert isinstance(state, MarketState)
            assert state.state_name in ("bull", "bear", "sideways")

    def test_predict_unfitted_returns_default(self):
        hmm = HMMStateDetector()
        state = hmm.predict_state(np.array([0.01, 0.02, 0.03]))
        assert state.state_name == "sideways"

    def test_insufficient_data(self):
        hmm = HMMStateDetector()
        hmm.fit(np.array([0.01, 0.02]))  # 太少
        assert not hmm.is_fitted


class TestStackingEnsemble:
    def test_fit_predict(self):
        stacking = StackingEnsemble(alpha=1.0)
        preds = {
            "lstm": np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
            "xgboost": np.array([0.02, 0.01, 0.04, 0.03, 0.06]),
        }
        y_true = np.array([0.015, 0.015, 0.035, 0.035, 0.055])
        stacking.fit(preds, y_true)
        assert stacking.is_fitted

        result = stacking.predict(preds)
        assert len(result) == 5

    def test_predict_unfitted_raises(self):
        stacking = StackingEnsemble()
        with pytest.raises(RuntimeError):
            stacking.predict({"lstm": np.array([0.01])})
