"""Tests for ChronosPredictor wrapper and sector training integration."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.chronos_model import ChronosPredictor, prepare_price_sequences


# ── Helper fixtures ─────────────────────────────────────────────


@pytest.fixture
def mock_pipeline():
    """Mock ChronosPipeline to avoid downloading model weights in tests."""
    mock = MagicMock()
    # Simulate forecast output: (batch, num_samples, prediction_length)
    import torch

    def fake_predict(contexts, prediction_length=1, num_samples=20, **kw):
        batch_size = len(contexts)
        # Return small random values as returns
        return torch.randn(batch_size, num_samples, prediction_length) * 0.01

    mock.predict = fake_predict
    return mock


@pytest.fixture
def predictor(mock_pipeline):
    """ChronosPredictor with mocked pipeline."""
    pred = ChronosPredictor(model_id="test/mock", device="cpu")
    pred.pipeline = mock_pipeline
    return pred


@pytest.fixture
def sample_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    n = 200
    prices = 100 * np.cumprod(1 + np.random.randn(n) * 0.02)
    returns = np.diff(np.log(prices))
    return prices, returns


# ── Unit Tests: ChronosPredictor ────────────────────────────────


class TestChronosPredictor:
    def test_init_defaults(self):
        pred = ChronosPredictor.__new__(ChronosPredictor)
        pred.__init__()
        assert pred.model_id == "amazon/chronos-bolt-small"
        assert pred.context_length == 64
        assert pred.prediction_length == 1
        assert pred.pipeline is None

    def test_price_to_returns(self):
        prices = np.array([100.0, 102.0, 101.0, 103.0])
        returns = ChronosPredictor._price_to_returns(prices)
        assert len(returns) == 3
        assert returns.dtype == np.float32
        # First return: log(102/100)
        np.testing.assert_almost_equal(returns[0], np.log(102 / 100), decimal=5)

    def test_price_to_returns_handles_zeros(self):
        prices = np.array([0.0, 100.0, 200.0])
        returns = ChronosPredictor._price_to_returns(prices)
        assert np.all(np.isfinite(returns))

    def test_predict_shape(self, predictor, sample_data):
        prices, _ = sample_data
        X = prices[-64:].reshape(1, -1)
        result = predictor.predict(X)
        assert result.shape == (1, 1)

    def test_predict_batch(self, predictor, sample_data):
        prices, _ = sample_data
        # Create batch of 5 sequences
        X = np.array([prices[i : i + 64] for i in range(5)])
        result = predictor.predict(X)
        assert result.shape == (5, 1)

    def test_train_calibration(self, predictor, sample_data):
        prices, returns = sample_data
        # Create training sequences
        ctx = 64
        X = np.array([prices[i : i + ctx] for i in range(len(prices) - ctx)])
        y = returns[ctx - 1 :]
        min_len = min(len(X), len(y))
        X, y = X[:min_len], y[:min_len]

        # Split
        split = int(len(X) * 0.8)
        history = predictor.train(X[:split], y[:split], X[split:], y[split:])

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 1
        assert len(history["val_loss"]) == 1
        assert predictor.scaler_mean is not None
        assert predictor.scaler_std is not None

    def test_evaluate_directional(self, predictor, sample_data):
        prices, returns = sample_data
        ctx = 64
        X = np.array([prices[i : i + ctx] for i in range(10)])
        y = returns[ctx - 1 : ctx - 1 + 10]

        result = predictor.evaluate_directional(X, y)
        assert "mse" in result
        assert "direction_acc" in result
        assert "beats_naive" in result
        assert 0 <= result["direction_acc"] <= 1

    def test_save_load(self, predictor):
        predictor.scaler_mean = 0.001
        predictor.scaler_std = 0.02

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chronos_cal.json"
            predictor.save(path)

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["model_id"] == "test/mock"
            assert data["scaler_mean"] == 0.001

            # Load into new instance
            new_pred = ChronosPredictor()
            new_pred.load(path)
            assert new_pred.model_id == "test/mock"
            assert new_pred.scaler_mean == 0.001
            assert new_pred._is_loaded is True

    def test_load_nonexistent(self, predictor, tmp_path):
        predictor.load(tmp_path / "nonexistent.json")
        # Should not crash, just warn

    def test_raw_predict_handles_nan(self, predictor):
        X = np.array([[np.nan] * 30 + [100.0] * 34])
        result = predictor._raw_predict_batch(X)
        assert len(result) == 1
        assert np.isfinite(result[0])


# ── Unit Tests: prepare_price_sequences ─────────────────────────


class TestPrepareSequences:
    def test_basic_extraction(self):
        import pandas as pd

        n = 100
        df = pd.DataFrame(
            {
                "close": np.random.randn(n).cumsum() + 100,
                "tb_label": np.random.randn(n) * 0.01,
            }
        )
        X, y = prepare_price_sequences(df, context_length=20, target_col="tb_label")

        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 20
        assert X.shape[0] == n - 20  # minus context_length, some may be dropped for NaN

    def test_handles_nan_target(self):
        import pandas as pd

        n = 50
        df = pd.DataFrame(
            {
                "close": np.arange(n, dtype=float) + 100,
                "tb_label": [np.nan] * 20 + list(np.random.randn(30) * 0.01),
            }
        )
        X, y = prepare_price_sequences(df, context_length=10)
        # Only rows 20-49 have valid targets (and need context_length=10 before them)
        assert X.shape[0] == 30
        assert not np.any(np.isnan(y))

    def test_missing_columns(self):
        import pandas as pd

        df = pd.DataFrame({"other_col": [1, 2, 3]})
        X, y = prepare_price_sequences(df)
        assert X.shape[0] == 0
        assert y.shape[0] == 0

    def test_empty_dataframe(self):
        import pandas as pd

        df = pd.DataFrame({"close": [], "tb_label": []})
        X, y = prepare_price_sequences(df, context_length=10)
        assert X.shape[0] == 0


# ── Integration: Quality Gate ───────────────────────────────────


class TestChronosQualityGate:
    def test_chronos_gate_pass(self):
        """Chronos passes quality gate when direction_acc >= 0.52 and beats naive."""
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer("TEST")
        results = {
            "chronos": {
                "test_direction_acc": 0.58,
                "test_beats_naive": True,
                "test_mse": 0.001,
            },
            "xgboost": {
                "test_direction_acc": 0.48,
                "test_beats_naive": False,
            },
        }
        # Skip CPCV by not having real data
        with patch.object(trainer, "cpcv_validate", side_effect=Exception("skip")):
            gate = trainer.quality_gate(results, "2020-01-01", "2025-01-01")

        assert gate["chronos_passed"] is True
        assert gate["chronos_direction_acc"] == 0.58
        assert gate["overall_passed"] is True

    def test_chronos_gate_fail(self):
        """Chronos fails quality gate when direction_acc < 0.52."""
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer("TEST")
        results = {
            "chronos": {
                "test_direction_acc": 0.49,
                "test_beats_naive": True,
            },
        }
        with patch.object(trainer, "cpcv_validate", side_effect=Exception("skip")):
            gate = trainer.quality_gate(results, "2020-01-01", "2025-01-01")

        assert gate["chronos_passed"] is False

    def test_overall_passes_with_chronos_only(self):
        """Overall gate passes even if only Chronos passes."""
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer("TEST")
        results = {
            "chronos": {
                "test_direction_acc": 0.55,
                "test_beats_naive": True,
            },
            "lstm": {
                "test_direction_acc": 0.40,
                "test_beats_naive": False,
            },
            "xgboost": {
                "test_direction_acc": 0.45,
                "test_beats_naive": False,
            },
        }
        with patch.object(trainer, "cpcv_validate", side_effect=Exception("skip")):
            gate = trainer.quality_gate(results, "2020-01-01", "2025-01-01")

        assert gate["chronos_passed"] is True
        assert gate["lstm_passed"] is False
        assert gate["xgb_passed"] is False
        assert gate["overall_passed"] is True


# ── Integration: Trainer predict with Chronos ───────────────────


class TestTrainerChronosPredict:
    def test_predict_uses_chronos_over_lstm(self):
        """When both chronos and lstm are available, predict() uses chronos."""
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer("2317")
        # Mock chronos
        mock_chronos = MagicMock()
        mock_chronos.context_length = 64
        mock_chronos.predict.return_value = np.array([[0.01]])
        trainer.chronos = mock_chronos
        # Mock lstm (should NOT be called)
        mock_lstm = MagicMock()
        trainer.lstm = mock_lstm

        # Mock feature engineer
        import pandas as pd

        fake_df = pd.DataFrame(
            {
                "close": np.random.randn(200).cumsum() + 100,
                "return_1d": np.random.randn(200) * 0.01,
                "realized_vol_20d": np.abs(np.random.randn(200)) * 0.02,
            }
        )
        with patch.object(trainer.feature_eng, "build_features", return_value=fake_df):
            result = trainer.predict("2024-01-01", "2025-01-01")

        # Chronos should have been called
        mock_chronos.predict.assert_called_once()
        # LSTM should NOT have been called
        mock_lstm.predict.assert_not_called()
        assert result is not None
