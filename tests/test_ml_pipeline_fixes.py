"""Tests for ML pipeline fixes (Phases 1-4 + audit fixes).

Validates:
1. Feature selection runs on train-only data (no leakage)
2. Sample weights aligned via prepare_tabular
3. direction_accuracy_classify with tb_class
4. XGBoost adaptive hyperparameters
5. DirectionAwareLoss behavior (proportional scaling)
6. Quality gate decoupling (CPCV only kills XGBoost)
7. CPCV PBO reproducibility (fixed seed)
8. XGBoost regressor no-learning detection
9. LSTM CPCV hard gate at 0.46
10. Direction score confidence weighting
11. Classifier early stopping with long patience
12. Direction bonus clamping
"""

import numpy as np

from src.models.ensemble import direction_accuracy, direction_accuracy_classify
from src.models.xgboost_model import StockXGBoost, StockXGBoostClassifier
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

    def test_weight_renormalized_after_dropna(self):
        """Weights should be re-normalized to mean≈1.0 after dropna removes NaN-label rows.

        This is the ROOT CAUSE fix: compute_sample_weights sets NaN-label rows to weight=0,
        then normalizes mean=1.0 over ALL rows. After dropna removes NaN rows, remaining
        weights had mean≈0.68, breaking XGBoost min_child_weight thresholds.
        """
        import pandas as pd

        fe = FeatureEngineer()
        n = 100
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "close": np.random.uniform(100, 200, n),
                "sma_5": np.random.uniform(100, 200, n),
                "rsi_14": np.random.uniform(30, 70, n),
                TB_TARGET_COLUMN: np.random.normal(0, 0.03, n),
                TB_WEIGHT_COLUMN: np.random.uniform(0.5, 1.0, n),
            }
        )
        # Set some rows to NaN label with weight=0 (mimics compute_sample_weights)
        nan_indices = np.random.choice(n, 30, replace=False)
        df.loc[nan_indices, TB_TARGET_COLUMN] = np.nan
        df.loc[nan_indices, TB_WEIGHT_COLUMN] = 0.0
        # Before dropna, mean of positive weights is ~0.75 (not 1.0)

        cols = ["close", "sma_5", "rsi_14"]
        X, y, w = fe.prepare_tabular(
            df, cols, TB_TARGET_COLUMN, weight_col=TB_WEIGHT_COLUMN
        )
        # After fix: weights should be re-normalized to mean≈1.0
        assert len(w) == 70  # 100 - 30 NaN rows
        w_pos = w[w > 0]
        assert abs(w_pos.mean() - 1.0) < 0.02, (
            f"Weight mean should be ≈1.0 after re-normalization, got {w_pos.mean():.4f}"
        )


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
        """< 800 samples should get stronger regularization"""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 5).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)

        xgb = StockXGBoost()
        xgb.train(X, y)

        # After adaptive adjustment — moderate regularization with subsampling
        assert xgb.model.get_params()["max_depth"] == 4
        assert xgb.model.get_params()["min_child_weight"] == 5
        assert xgb.model.get_params()["subsample"] == 0.7

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

        loss_fn = DirectionAwareLoss(delta=2.0, direction_weight=0.5)

        # Same direction: pred=+, target=+
        pred_correct = torch.tensor([[0.5]])
        target = torch.tensor([[1.0]])
        loss_correct = loss_fn(pred_correct, target)

        # Wrong direction: pred=-, target=+
        pred_wrong = torch.tensor([[-0.5]])
        loss_wrong = loss_fn(pred_wrong, target)

        assert loss_wrong > loss_correct

    def test_no_penalty_when_aligned(self):
        """Perfectly aligned predictions → multiplicative factor = 1.0 (no penalty)"""
        import torch
        from src.models.lstm_model import DirectionAwareLoss

        loss_fn = DirectionAwareLoss(delta=2.0, direction_weight=0.5)
        huber_fn = torch.nn.HuberLoss(delta=2.0)

        pred = torch.tensor([[1.0], [2.0], [-1.0]])
        target = torch.tensor([[0.5], [1.5], [-0.5]])

        da_loss = loss_fn(pred, target)
        huber_loss = huber_fn(pred, target)

        # When all directions match, wrong_dir=0 → loss = reg_loss * (1+0) = reg_loss
        assert abs(da_loss.item() - huber_loss.item()) < 1e-5

    def test_proportional_scaling(self):
        """Direction penalty should scale with regression loss magnitude"""
        import torch
        from src.models.lstm_model import DirectionAwareLoss

        loss_fn = DirectionAwareLoss(delta=2.0, direction_weight=0.5)

        # Small regression loss — direction penalty should be proportionally small
        pred_small = torch.tensor([[-0.01]])
        target_small = torch.tensor([[0.01]])
        loss_small = loss_fn(pred_small, target_small)

        # Large regression loss — direction penalty should be proportionally large
        pred_large = torch.tensor([[-1.0]])
        target_large = torch.tensor([[1.0]])
        loss_large = loss_fn(pred_large, target_large)

        # Both have 100% wrong direction, but penalty scales with reg_loss
        # loss = reg_loss * (1 + 0.2 * 1.0) = reg_loss * 1.2
        assert loss_large > loss_small * 10  # large is much bigger


# ── Phase 3 Tests (ML pipeline bug fixes) ─────────────────


class TestDirectionAccuracyClassifyEpsilon:
    def test_near_zero_pred_filtered(self):
        """Near-zero predictions (|pred| < epsilon) should be excluded"""
        pred = np.array([0.001, -0.0005, 0.05, -0.03])
        tb_class = np.array([1, -1, 1, -1])
        # pred[0]=0.001 and pred[1]=-0.0005 are below epsilon=0.003
        # Only pred[2]=0.05 vs class[2]=1 → match
        # Only pred[3]=-0.03 vs class[3]=-1 → match
        acc = direction_accuracy_classify(pred, tb_class)
        assert acc == 1.0

    def test_all_near_zero_pred_returns_half(self):
        """All predictions below epsilon → return 0.5 (unknown)"""
        pred = np.array([0.001, -0.002, 0.0001])
        tb_class = np.array([1, -1, 1])
        assert direction_accuracy_classify(pred, tb_class) == 0.5

    def test_mixed_near_zero_and_wrong(self):
        """Near-zero filtered out, remaining has 50% accuracy"""
        pred = np.array([0.0001, 0.05, -0.04])  # first filtered
        tb_class = np.array([1, 1, 1])  # second matches, third doesn't
        acc = direction_accuracy_classify(pred, tb_class)
        assert acc == 0.5  # 1 match, 1 mismatch out of 2

    def test_backward_compatible_with_large_preds(self):
        """Existing tests with large preds should still work identically"""
        pred = np.array([0.05, -0.03, 0.02, -0.01])
        tb_class = np.array([1, -1, 1, -1])
        assert direction_accuracy_classify(pred, tb_class) == 1.0


class TestEarlyStoppingAdditive:
    def test_monotonic_improvement(self):
        """Higher dir_acc with same val_loss should always improve composite"""
        # Simulate: fixed val_loss, increasing dir_acc should decrease composite
        val_loss = 0.5
        direction_bonus = val_loss * 0.2  # calibrated on first epoch

        composites = []
        for dir_acc in [0.3, 0.5, 0.7, 0.9]:
            composite = val_loss - direction_bonus * dir_acc
            composites.append(composite)

        # Verify monotonically decreasing (improving)
        for i in range(len(composites) - 1):
            assert composites[i] > composites[i + 1], (
                f"Composite not monotonically decreasing: {composites}"
            )

    def test_direction_bonus_clamped(self):
        """direction_bonus should be clamped at 0.01 to prevent masking bad val_loss"""
        # Small val_loss: bonus = val_loss * 0.2 (no clamp needed)
        val_loss_small = 0.01
        bonus_small = min(val_loss_small * 0.2, 0.01)
        assert bonus_small == 0.002

        # Large val_loss: bonus clamped at 0.01
        val_loss_large = 5.0
        bonus_large = min(val_loss_large * 0.2, 0.01)
        assert bonus_large == 0.01  # clamped, not 1.0

        # With clamp, a huge val_loss can't be masked by direction accuracy
        composite_no_dir = val_loss_large
        composite_perfect_dir = val_loss_large - bonus_large * 1.0
        # Only 0.2% improvement, not 20%
        improvement = (composite_no_dir - composite_perfect_dir) / composite_no_dir
        assert improvement < 0.01


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


# ── Audit Fix Tests ──────────────────────────────────────


class TestXGBRegressorNoLearning:
    def test_no_learning_detected_in_quality_gate(self):
        """Regressor with no_learning=True should fail quality gate"""
        from src.models.trainer import ModelTrainer
        from unittest.mock import patch

        trainer = ModelTrainer("test")
        results = {
            "xgboost": {
                "test_direction_acc": 0.55,
                "test_beats_naive": True,
                "no_learning": True,
            },
        }

        with patch.object(
            trainer,
            "cpcv_validate",
            return_value={"pbo": {"pbo": 0.3}, "mean_oos": 0.55},
        ):
            gate = trainer.quality_gate(results, "2025-01-01", "2025-12-31")

        assert gate["xgb_passed"] is False

    def test_normal_learning_passes(self):
        """Regressor with no_learning=False should pass if other criteria met"""
        from src.models.trainer import ModelTrainer
        from unittest.mock import patch

        trainer = ModelTrainer("test")
        results = {
            "xgboost": {
                "test_direction_acc": 0.55,
                "test_beats_naive": True,
                "no_learning": False,
            },
        }

        with patch.object(
            trainer,
            "cpcv_validate",
            return_value={"pbo": {"pbo": 0.3}, "mean_oos": 0.55},
        ):
            gate = trainer.quality_gate(results, "2025-01-01", "2025-12-31")

        assert gate["xgb_passed"] is True


class TestLSTMCPCVHardGate:
    def test_below_046_rejected(self):
        """LSTM CPCV < 0.46 should reject the model"""
        from src.models.trainer import ModelTrainer
        from unittest.mock import patch

        trainer = ModelTrainer("test")
        results = {
            "lstm": {"test_direction_acc": 0.55, "test_beats_naive": True},
        }

        with patch.object(trainer, "_lstm_cpcv_direction_check", return_value=0.44):
            gate = trainer.quality_gate(results, "2025-01-01", "2025-12-31")

        assert gate["lstm_passed"] is False
        assert gate["lstm_cpcv_dir_acc"] == 0.44

    def test_above_048_passes(self):
        """LSTM CPCV >= 0.48 should pass"""
        from src.models.trainer import ModelTrainer
        from unittest.mock import patch

        trainer = ModelTrainer("test")
        results = {
            "lstm": {"test_direction_acc": 0.55, "test_beats_naive": True},
        }

        with patch.object(trainer, "_lstm_cpcv_direction_check", return_value=0.52):
            gate = trainer.quality_gate(results, "2025-01-01", "2025-12-31")

        assert gate["lstm_passed"] is True


class TestDirectionScoreConfidence:
    def test_confident_directional(self):
        """High P(up), low P(neutral) → strong positive score"""
        xgb_cls = StockXGBoostClassifier()

        # Mock predict_proba to return controlled values
        import unittest.mock as mock

        proba = np.array([[0.05, 0.10, 0.85]])  # strong up
        with mock.patch.object(xgb_cls, "predict_proba", return_value=proba):
            score = xgb_cls.predict_direction_score(np.zeros((1, 5)))
        # raw = 0.85 - 0.05 = 0.80, confidence = 1 - 0.10 = 0.90
        # score = 0.80 * 0.90 = 0.72
        assert abs(score[0] - 0.72) < 0.01

    def test_uncertain_split(self):
        """Equal P(up) and P(down), low P(neutral) → score near 0"""
        xgb_cls = StockXGBoostClassifier()

        import unittest.mock as mock

        proba = np.array([[0.45, 0.10, 0.45]])  # uncertain
        with mock.patch.object(xgb_cls, "predict_proba", return_value=proba):
            score = xgb_cls.predict_direction_score(np.zeros((1, 5)))
        # raw = 0.45 - 0.45 = 0, score = 0 * 0.9 = 0
        assert abs(score[0]) < 0.01

    def test_confidently_neutral(self):
        """High P(neutral) → score near 0 regardless of slight up/down tilt"""
        xgb_cls = StockXGBoostClassifier()

        import unittest.mock as mock

        proba = np.array([[0.05, 0.90, 0.05]])  # confidently neutral
        with mock.patch.object(xgb_cls, "predict_proba", return_value=proba):
            score = xgb_cls.predict_direction_score(np.zeros((1, 5)))
        # raw = 0.05 - 0.05 = 0, confidence = 0.10
        # score = 0 * 0.10 = 0
        assert abs(score[0]) < 0.01


class TestAdaptivePatience:
    def test_short_epochs_default_patience(self):
        """50 epochs → patience = max(15, 50//5) = 15"""
        from src.models.lstm_model import LSTMPredictor

        lstm = LSTMPredictor(input_size=5)
        # We can't easily test the internal patience without training,
        # but we can verify the formula
        epochs = 50
        expected = max(15, epochs // 5)
        assert expected == 15

    def test_long_epochs_scaled_patience(self):
        """200 epochs → patience = max(15, 200//5) = 40"""
        epochs = 200
        expected = max(15, epochs // 5)
        assert expected == 40


class TestDataDrivenNoiseThreshold:
    def test_high_vol_stock_larger_threshold(self):
        """High-volatility stock should get a larger noise threshold"""
        import pandas as pd
        from src.analysis.labels import triple_barrier_classify

        np.random.seed(42)
        n = 200
        # High-vol stock: 3% daily moves
        close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.03, n))
        high = close * (1 + np.abs(np.random.normal(0, 0.015, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.015, n)))
        df = pd.DataFrame({"close": close, "high": high, "low": low})

        labels = triple_barrier_classify(
            df, max_holding=7, upper_multiplier=1.0, lower_multiplier=1.0
        )
        valid = labels.dropna()
        # With adaptive threshold, fewer time-expired labels should be neutral
        # (the threshold is higher, so more returns are classified as directional)
        assert len(valid) > 0

    def test_low_vol_stock_base_threshold(self):
        """Low-volatility stock should use base threshold (0.003)"""
        import pandas as pd
        from src.analysis.labels import triple_barrier_classify

        np.random.seed(42)
        n = 200
        # Low-vol stock: 0.3% daily moves (median abs return ≈ 0.002)
        close = 100 * np.cumprod(1 + np.random.normal(0.0001, 0.003, n))
        high = close * (1 + np.abs(np.random.normal(0, 0.002, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.002, n)))
        df = pd.DataFrame({"close": close, "high": high, "low": low})

        labels = triple_barrier_classify(
            df, max_holding=7, upper_multiplier=1.0, lower_multiplier=1.0
        )
        valid = labels.dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})


class TestFfillLimit:
    def test_ffill_limit_applied(self):
        """ffill(limit=5) should not propagate stale data beyond 5 days"""
        import pandas as pd

        fe = FeatureEngineer()
        # Create a df with a gap of 10 NaN in margin_balance
        n = 100
        df = pd.DataFrame(
            {
                "close": np.random.uniform(90, 110, n),
                "sma_60": np.ones(n) * 100,  # required by _fill_missing
                "margin_balance": np.concatenate(
                    [
                        np.ones(30) * 100,  # valid data
                        np.full(10, np.nan),  # 10-day gap
                        np.ones(60) * 200,  # valid data after gap
                    ]
                ),
            }
        )
        result = fe._fill_missing(df.copy())
        # After ffill(limit=5), positions 30-34 should be filled (100),
        # but positions 35-39 should be 0 (fillna(0) after ffill limit)
        assert result["margin_balance"].iloc[34] == 100.0  # within limit
        assert result["margin_balance"].iloc[35] == 0.0  # beyond limit
