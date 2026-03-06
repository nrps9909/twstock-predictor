"""Tests for ML pipeline fixes (Phases 1-2).

Validates:
1. Feature selection runs on train-only data (no leakage)
2. Sample weights aligned via prepare_tabular
3. direction_accuracy_classify with tb_class
4. XGBoost adaptive hyperparameters
5. DirectionAwareLoss behavior
6. Quality gate decoupling (CPCV only kills XGBoost)
7. CPCV PBO reproducibility (fixed seed)
"""

import numpy as np

from src.models.ensemble import direction_accuracy, direction_accuracy_classify
from src.models.xgboost_model import StockXGBoost
from src.analysis.features import FeatureEngineer, TB_TARGET_COLUMN, TB_WEIGHT_COLUMN


# ── Phase 1 Tests ─────────────────────────────────────────


class TestDirectionAccuracyEpsilon:
    def test_filters_small_returns(self):
        """epsilon=0.003 should filter out ±0.3% returns"""
        pred = np.array([0.01, -0.01, 0.005, -0.002])
        target = np.array([0.001, -0.001, 0.01, -0.01])
        # target[0]=0.001 and target[1]=-0.001 are below epsilon=0.003
        # Only target[2]=0.01 and target[3]=-0.01 count
        acc = direction_accuracy(pred, target)
        # pred[2]=0.005 vs target[2]=0.01 → same sign ✓
        # pred[3]=-0.002 vs target[3]=-0.01 → same sign ✓
        assert acc == 1.0

    def test_all_small_returns_half(self):
        """If all targets are tiny (< epsilon), return 0.5"""
        pred = np.array([0.01, -0.01])
        target = np.array([0.001, -0.001])
        assert direction_accuracy(pred, target) == 0.5


class TestCPCVReproducibility:
    def test_pbo_deterministic(self):
        """PBO computation should be deterministic (fixed seed)"""
        from src.models.cpcv import CPCVAnalyzer

        def dummy_eval(train_idx, test_idx):
            return 0.55 + len(test_idx) * 0.0001

        analyzer = CPCVAnalyzer(n_blocks=6, k_test=2)
        r1 = analyzer.run_cpcv_analysis(200, dummy_eval)
        r2 = analyzer.run_cpcv_analysis(200, dummy_eval)
        assert r1["pbo"]["pbo"] == r2["pbo"]["pbo"]


class TestQualityGateDecoupling:
    def test_cpcv_only_kills_xgboost(self):
        """CPCV failure should only affect XGBoost, not LSTM"""
        from src.models.trainer import ModelTrainer
        from unittest.mock import patch

        trainer = ModelTrainer("test")

        results = {
            "lstm": {"test_direction_acc": 0.55, "test_beats_naive": True},
            "xgboost": {"test_direction_acc": 0.56, "test_beats_naive": True},
        }

        # Mock CPCV to return very high PBO + low mean_oos
        mock_cpcv = {"pbo": {"pbo": 0.85}, "mean_oos": 0.45}
        with patch.object(trainer, "cpcv_validate", return_value=mock_cpcv):
            gate = trainer.quality_gate(results, "2025-01-01", "2025-12-31")

        assert gate["lstm_passed"] is True
        assert gate["xgb_passed"] is False
        assert gate["overall_passed"] is True

    def test_moderate_pbo_warning_only(self):
        """PBO 0.6-0.8 should only warn, not kill"""
        from src.models.trainer import ModelTrainer
        from unittest.mock import patch

        trainer = ModelTrainer("test")
        results = {
            "xgboost": {"test_direction_acc": 0.56, "test_beats_naive": True},
        }

        mock_cpcv = {"pbo": {"pbo": 0.65}, "mean_oos": 0.52}
        with patch.object(trainer, "cpcv_validate", return_value=mock_cpcv):
            gate = trainer.quality_gate(results, "2025-01-01", "2025-12-31")

        assert gate["xgb_passed"] is True


# ── Phase 2 Tests ─────────────────────────────────────────


class TestSampleWeightAlignment:
    def test_prepare_tabular_with_weights(self, sample_features_df):
        """prepare_tabular with weight_col returns aligned weights"""
        fe = FeatureEngineer()
        df = sample_features_df.copy()
        cols = ["close", "sma_5", "rsi_14"]

        X, y, w = fe.prepare_tabular(
            df, cols, TB_TARGET_COLUMN, weight_col=TB_WEIGHT_COLUMN
        )
        assert len(X) == len(y) == len(w)
        assert w is not None
        assert w.min() >= 0

    def test_prepare_tabular_without_weights(self, sample_features_df):
        """prepare_tabular without weight_col returns (X, y) only"""
        fe = FeatureEngineer()
        df = sample_features_df.copy()
        cols = ["close", "sma_5", "rsi_14"]

        result = fe.prepare_tabular(df, cols, TB_TARGET_COLUMN)
        assert len(result) == 2  # (X, y), no weights

    def test_prepare_tabular_missing_weight_col(self, sample_features_df):
        """prepare_tabular with nonexistent weight_col returns None"""
        fe = FeatureEngineer()
        df = sample_features_df.copy()
        cols = ["close", "sma_5", "rsi_14"]

        X, y, w = fe.prepare_tabular(
            df, cols, TB_TARGET_COLUMN, weight_col="nonexistent"
        )
        assert w is None


class TestDirectionAccuracyClassify:
    def test_perfect_prediction(self):
        """Predictions matching tb_class signs → accuracy 1.0"""
        pred = np.array([0.05, -0.03, 0.02, -0.01])
        tb_class = np.array([1, -1, 1, -1])
        assert direction_accuracy_classify(pred, tb_class) == 1.0

    def test_excludes_class_zero(self):
        """Class 0 (time-expired) should be excluded"""
        pred = np.array([0.05, -0.03, 0.02])
        tb_class = np.array([1, 0, -1])
        # pred[0]=+, class[0]=1 → match
        # pred[1]=-, class[1]=0 → excluded
        # pred[2]=+, class[2]=-1 → mismatch
        acc = direction_accuracy_classify(pred, tb_class)
        assert acc == 0.5  # 1 match, 1 mismatch

    def test_all_zero_class(self):
        """All class 0 → return 0.5"""
        pred = np.array([0.1, -0.1])
        tb_class = np.array([0, 0])
        assert direction_accuracy_classify(pred, tb_class) == 0.5


class TestXGBoostAdaptiveParams:
    def test_small_dataset_relaxed(self):
        """< 800 samples should get relaxed regularization"""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 5).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)

        xgb = StockXGBoost()
        xgb.train(X, y)

        # After adaptive adjustment
        assert xgb.model.get_params()["max_depth"] == 5
        assert xgb.model.get_params()["min_child_weight"] == 3

    def test_large_dataset_defaults(self):
        """>= 2000 samples should keep default params"""
        np.random.seed(42)
        n = 2500
        X = np.random.randn(n, 5).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)

        xgb = StockXGBoost()
        xgb.train(X, y)

        # Defaults unchanged
        assert xgb.model.get_params()["max_depth"] == 4
        assert xgb.model.get_params()["min_child_weight"] == 5


class TestDirectionAwareLoss:
    def test_wrong_direction_penalty(self):
        """Predictions with wrong direction should incur higher loss"""
        import torch
        from src.models.lstm_model import DirectionAwareLoss

        loss_fn = DirectionAwareLoss(delta=2.0, direction_weight=0.2)

        # Same direction: pred=+, target=+
        pred_correct = torch.tensor([[0.5]])
        target = torch.tensor([[1.0]])
        loss_correct = loss_fn(pred_correct, target)

        # Wrong direction: pred=-, target=+
        pred_wrong = torch.tensor([[-0.5]])
        loss_wrong = loss_fn(pred_wrong, target)

        assert loss_wrong > loss_correct

    def test_no_penalty_when_aligned(self):
        """Perfectly aligned predictions → no direction penalty"""
        import torch
        from src.models.lstm_model import DirectionAwareLoss

        loss_fn = DirectionAwareLoss(delta=2.0, direction_weight=0.2)
        huber_fn = torch.nn.HuberLoss(delta=2.0)

        pred = torch.tensor([[1.0], [2.0], [-1.0]])
        target = torch.tensor([[0.5], [1.5], [-0.5]])

        da_loss = loss_fn(pred, target)
        huber_loss = huber_fn(pred, target)

        # When all directions match, DirectionAwareLoss == HuberLoss
        assert abs(da_loss.item() - huber_loss.item()) < 1e-5


# ── Phase 4 Tests ─────────────────────────────────────────


class TestICWeightAdjustment:
    def test_ic_adjusted_weights_structure(self):
        """IC adjustment should produce valid weight dict"""
        from unittest.mock import patch
        from api.services.market_service import (
            _get_ic_adjusted_weights,
            BASE_WEIGHTS,
            _ic_weights_cache,
        )

        # Reset cache
        _ic_weights_cache["date"] = None
        _ic_weights_cache["weights"] = None

        mock_ic = {
            "composite_institutional": {"ic_mean": 0.05, "icir": 0.6, "n_dates": 40},
            "technical_signal": {"ic_mean": -0.04, "icir": -0.3, "n_dates": 40},
        }

        with patch(
            "api.services.market_service.get_all_factor_ic_summary",
            return_value=mock_ic,
        ):
            weights = _get_ic_adjusted_weights()

        assert weights is not None
        assert abs(sum(weights.values()) - 1.0) < 0.001
        # composite_institutional should be boosted (ICIR > 0.5)
        assert (
            weights["composite_institutional"] > BASE_WEIGHTS["composite_institutional"]
        )
        # technical_signal should be penalized (IC < -0.03)
        assert weights["technical_signal"] < BASE_WEIGHTS["technical_signal"]

        # Reset cache
        _ic_weights_cache["date"] = None
        _ic_weights_cache["weights"] = None

    def test_ic_adjusted_fallback_insufficient_data(self):
        """No IC data → returns None (fallback to BASE_WEIGHTS)"""
        from unittest.mock import patch
        from api.services.market_service import (
            _get_ic_adjusted_weights,
            _ic_weights_cache,
        )

        _ic_weights_cache["date"] = None
        _ic_weights_cache["weights"] = None

        with patch(
            "api.services.market_service.get_all_factor_ic_summary", return_value={}
        ):
            result = _get_ic_adjusted_weights()

        assert result is None
        _ic_weights_cache["date"] = None
        _ic_weights_cache["weights"] = None
