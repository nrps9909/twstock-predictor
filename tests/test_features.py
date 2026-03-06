"""特徵工程測試"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.features import (
    FEATURE_COLUMNS,
    TB_TARGET_COLUMN,
    FeatureEngineer,
)


@pytest.fixture
def fe():
    return FeatureEngineer()


class TestSelectFeatures:
    def test_returns_list(self, fe, sample_features_df):
        selected = fe.select_features(sample_features_df, max_features=10)
        assert isinstance(selected, list)
        assert len(selected) <= 10

    def test_skip_when_few_features(self, fe):
        """特徵數 <= max 時跳過篩選"""
        df = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104] * 20,
                "volume": range(100),
                "tb_label": np.random.normal(0, 0.01, 100),
            }
        )
        selected = fe.select_features(df, max_features=50)
        # 只有 close 和 volume 在 FEATURE_COLUMNS 中
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        assert selected == available

    def test_shap_method(self, fe, sample_features_df):
        selected = fe.select_features(
            sample_features_df, max_features=10, method="shap"
        )
        assert len(selected) <= 10

    def test_persist_importance(self, fe, sample_features_df, session_factory):
        """select_features 持久化重要性分數到 DB"""
        from src.db.models import FeatureImportanceRecord

        selected = fe.select_features(
            sample_features_df,
            max_features=10,
            session_factory=session_factory,
            stock_id="2330",
        )
        assert len(selected) > 0

        # 確認 DB 中有紀錄
        session = session_factory()
        try:
            records = (
                session.query(FeatureImportanceRecord)
                .filter_by(
                    stock_id="2330",
                )
                .all()
            )
            assert len(records) > 0
            # 確認 rank 是 1-based
            ranks = [r.rank for r in records]
            assert min(ranks) == 1

            # 再跑一次，確認不會重複（刪除再 INSERT）
            fe.select_features(
                sample_features_df,
                max_features=10,
                session_factory=session_factory,
                stock_id="2330",
            )
            records2 = (
                session.query(FeatureImportanceRecord)
                .filter_by(
                    stock_id="2330",
                )
                .all()
            )
            # 第二次跑完後的紀錄數應和第一次相同
            assert len(records2) == len(records)
        finally:
            session.close()


class TestRemoveCollinear:
    def test_drops_correlated_features(self, fe):
        n = 100
        df = pd.DataFrame(
            {
                "a": np.random.normal(0, 1, n),
                "b": np.random.normal(0, 1, n),
            }
        )
        df["c"] = df["a"] + np.random.normal(0, 0.01, n)  # 高度相關
        result = fe._remove_collinear(df, ["a", "b", "c"], threshold=0.95)
        # a 和 c 高度相關，c 應被移除（排名靠後）
        assert "a" in result
        assert "c" not in result

    def test_keeps_uncorrelated(self, fe):
        n = 100
        df = pd.DataFrame(
            {
                "a": np.random.normal(0, 1, n),
                "b": np.random.normal(0, 1, n),
            }
        )
        result = fe._remove_collinear(df, ["a", "b"], threshold=0.95)
        assert result == ["a", "b"]


class TestPrepareSequences:
    def test_output_shape(self, fe, sample_features_df):
        feature_cols = ["close", "volume", "return_1d", "sma_5", "rsi_14"]
        X, y = fe.prepare_sequences(
            sample_features_df,
            seq_len=10,
            feature_cols=feature_cols,
            target_col=TB_TARGET_COLUMN,
        )
        assert X.ndim == 3
        assert X.shape[1] == 10  # seq_len
        assert X.shape[2] == len(feature_cols)
        assert len(y) == len(X)

    def test_empty_on_short_data(self, fe):
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0],
                "tb_label": [0.01, 0.02],
            }
        )
        X, y = fe.prepare_sequences(df, seq_len=60, feature_cols=["close"])
        assert len(X) == 0

    def test_dtype_float32(self, fe, sample_features_df):
        X, y = fe.prepare_sequences(
            sample_features_df,
            seq_len=5,
            feature_cols=["close"],
            target_col=TB_TARGET_COLUMN,
        )
        assert X.dtype == np.float32
        assert y.dtype == np.float32


class TestPrepareTabular:
    def test_output_shape(self, fe, sample_features_df):
        feature_cols = ["close", "volume", "return_1d"]
        X, y = fe.prepare_tabular(
            sample_features_df,
            feature_cols,
            target_col=TB_TARGET_COLUMN,
        )
        assert X.ndim == 2
        assert X.shape[1] == len(feature_cols)
        assert len(y) == len(X)
