"""Triple Barrier 標籤測試"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.labels import (
    compute_atr,
    triple_barrier_label,
    triple_barrier_classify,
    compute_sample_weights,
)


@pytest.fixture
def price_df():
    """合成價格 DataFrame for label testing"""
    np.random.seed(42)
    n = 200
    close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.015, n))
    high = close * (1 + np.abs(np.random.normal(0, 0.008, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.008, n)))
    return pd.DataFrame({"close": close, "high": high, "low": low})


class TestComputeATR:
    def test_atr_shape(self, price_df):
        atr = compute_atr(price_df["high"], price_df["low"], price_df["close"])
        assert len(atr) == len(price_df)

    def test_atr_positive(self, price_df):
        atr = compute_atr(price_df["high"], price_df["low"], price_df["close"])
        valid = atr.dropna()
        assert (valid > 0).all()

    def test_atr_nan_warmup(self, price_df):
        atr = compute_atr(
            price_df["high"], price_df["low"], price_df["close"], window=14
        )
        assert atr.iloc[:13].isna().all()
        assert atr.iloc[13:].notna().all()


class TestTripleBarrierLabel:
    def test_output_shape(self, price_df):
        labels = triple_barrier_label(price_df)
        assert len(labels) == len(price_df)

    def test_nan_in_warmup(self, price_df):
        labels = triple_barrier_label(price_df, atr_window=14)
        # ATR 暖身期應為 NaN
        assert labels.iloc[:13].isna().all()

    def test_upper_touch_positive(self, price_df):
        """觸及上障礙時標籤應為正值"""
        labels = triple_barrier_label(price_df)
        valid = labels.dropna()
        positive = valid[valid > 0]
        # 至少有部分正值
        assert len(positive) > 0

    def test_lower_touch_negative(self, price_df):
        """觸及下障礙時標籤應為負值"""
        labels = triple_barrier_label(price_df)
        valid = labels.dropna()
        negative = valid[valid < 0]
        assert len(negative) > 0

    def test_label_range_bounded(self, price_df):
        """標籤值應該在合理範圍內"""
        labels = triple_barrier_label(
            price_df, upper_multiplier=2.0, lower_multiplier=2.0
        )
        valid = labels.dropna()
        # 報酬率不應超過 ±50%（短期內）
        assert valid.abs().max() < 0.5

    def test_continuous_values(self, price_df):
        """回歸標籤應為連續值"""
        labels = triple_barrier_label(price_df)
        valid = labels.dropna()
        unique_vals = valid.nunique()
        assert unique_vals > 3  # 不應只有離散值


class TestTripleBarrierClassify:
    def test_discrete_labels(self, price_df):
        labels = triple_barrier_classify(price_df)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_has_all_classes(self, price_df):
        """應包含正、負、零標籤"""
        labels = triple_barrier_classify(price_df)
        valid = labels.dropna()
        assert 1 in valid.values
        assert -1 in valid.values


class TestSampleWeights:
    def test_weight_shape(self, price_df):
        price_df["tb_label"] = triple_barrier_label(price_df)
        weights = compute_sample_weights(price_df)
        assert len(weights) == len(price_df)

    def test_weights_range(self, price_df):
        price_df["tb_label"] = triple_barrier_label(price_df)
        weights = compute_sample_weights(price_df)
        assert (weights >= 0).all()
        assert (weights <= 1).all()

    def test_monotonicity_with_overlap(self):
        """重疊越多的區域權重應越低"""
        # 全部相鄰標籤都有效 → 高重疊 → 低權重
        n = 50
        df = pd.DataFrame(
            {
                "tb_label": np.random.normal(0, 0.01, n),
            }
        )
        weights_dense = compute_sample_weights(df, max_holding=10)

        # 稀疏標籤（每隔 15 天一個有效標籤）
        df_sparse = pd.DataFrame(
            {
                "tb_label": [np.nan] * n,
            }
        )
        for i in range(0, n, 15):
            df_sparse.loc[i, "tb_label"] = 0.01
        weights_sparse = compute_sample_weights(df_sparse, max_holding=10)

        # 稀疏標籤的有效權重應 >= 密集標籤的有效權重
        dense_valid = weights_dense[~df["tb_label"].isna()]
        sparse_valid = weights_sparse[~df_sparse["tb_label"].isna()]
        assert sparse_valid.mean() >= dense_valid.mean()
