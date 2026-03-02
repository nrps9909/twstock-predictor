"""Triple Barrier 標籤方法（參考 Marcos López de Prado, AFML）

取代 naive pct_change(5).shift(-5) 的標籤方式。
三重障礙：
1. 上障礙（止盈）：價格觸及 entry + upper_barrier → 標籤 1（正報酬）
2. 下障礙（停損）：價格觸及 entry - lower_barrier → 標籤 -1（負報酬）
3. 時間障礙（到期）：max_holding 天後未觸及 → 標籤依實際報酬率方向

障礙寬度基於 ATR（Average True Range），自適應市場波動率。
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """計算 Average True Range"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def triple_barrier_label(
    df: pd.DataFrame,
    upper_multiplier: float = 2.0,
    lower_multiplier: float = 2.0,
    max_holding: int = 10,
    atr_window: int = 14,
) -> pd.Series:
    """Triple Barrier 標籤

    Args:
        df: 必須包含 'close', 'high', 'low' 欄位
        upper_multiplier: 上障礙 = ATR * upper_multiplier
        lower_multiplier: 下障礙 = ATR * lower_multiplier
        max_holding: 最大持有天數（時間障礙）
        atr_window: ATR 計算視窗

    Returns:
        Series of labels:
            - 連續值（觸障時的報酬率或到期時的報酬率）
            - 可用於回歸任務
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    atr = compute_atr(df["high"], df["low"], df["close"], window=atr_window)
    atr_vals = atr.values

    labels = np.full(n, np.nan)
    touch_type = np.full(n, np.nan)  # 1=upper, -1=lower, 0=time

    for i in range(n):
        if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
            continue

        entry_price = close[i]
        upper_barrier = entry_price + upper_multiplier * atr_vals[i]
        lower_barrier = entry_price - lower_multiplier * atr_vals[i]

        end_idx = min(i + max_holding, n - 1)
        if end_idx <= i:
            continue

        # 逐日檢查是否觸及障礙
        touched = False
        for j in range(i + 1, end_idx + 1):
            # 用 high/low 判斷日內是否觸及
            if high[j] >= upper_barrier:
                # 觸及上障礙：使用上障礙價格計算報酬率
                labels[i] = (upper_barrier - entry_price) / entry_price
                touch_type[i] = 1
                touched = True
                break
            elif low[j] <= lower_barrier:
                # 觸及下障礙：使用下障礙價格計算報酬率
                labels[i] = (lower_barrier - entry_price) / entry_price
                touch_type[i] = -1
                touched = True
                break

        if not touched:
            # 時間障礙：到期日的實際報酬率
            labels[i] = (close[end_idx] - entry_price) / entry_price
            touch_type[i] = 0

    return pd.Series(labels, index=df.index, name="tb_label")


def triple_barrier_classify(
    df: pd.DataFrame,
    upper_multiplier: float = 2.0,
    lower_multiplier: float = 2.0,
    max_holding: int = 10,
    atr_window: int = 14,
) -> pd.Series:
    """Triple Barrier 分類標籤（離散版）

    Returns:
        Series of {1, 0, -1}:
            1  = 觸及上障礙（正報酬）
            -1 = 觸及下障礙（負報酬）
            0  = 時間到期（方向不明確）
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    atr = compute_atr(df["high"], df["low"], df["close"], window=atr_window)
    atr_vals = atr.values

    labels = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
            continue

        entry_price = close[i]
        upper_barrier = entry_price + upper_multiplier * atr_vals[i]
        lower_barrier = entry_price - lower_multiplier * atr_vals[i]

        end_idx = min(i + max_holding, n - 1)
        if end_idx <= i:
            continue

        touched = False
        for j in range(i + 1, end_idx + 1):
            if high[j] >= upper_barrier:
                labels[i] = 1
                touched = True
                break
            elif low[j] <= lower_barrier:
                labels[i] = -1
                touched = True
                break

        if not touched:
            # 到期方向
            ret = (close[end_idx] - entry_price) / entry_price
            labels[i] = 1 if ret > 0 else (-1 if ret < 0 else 0)

    return pd.Series(labels, index=df.index, name="tb_class")


def compute_sample_weights(
    df: pd.DataFrame,
    label_col: str = "tb_label",
    max_holding: int = 10,
) -> pd.Series:
    """計算樣本唯一性權重（Average Uniqueness）

    衡量每個樣本的資訊重疊程度：
    - 若某天屬於多個標籤的持有期，其唯一性較低
    - 權重 = 平均唯一性（未與其他樣本重疊的比例）

    Args:
        df: 包含 label_col 的 DataFrame
        max_holding: 最大持有天數

    Returns:
        每個樣本的權重 (0, 1]
    """
    n = len(df)
    valid_mask = ~df[label_col].isna()

    # concurrency[t] = 在時間 t，有多少個標籤的持有期涵蓋該時間點
    concurrency = np.zeros(n)

    valid_indices = np.where(valid_mask.values)[0]
    for i in valid_indices:
        end = min(i + max_holding, n - 1)
        concurrency[i:end + 1] += 1

    # 避免除零
    concurrency = np.maximum(concurrency, 1)

    # 每個樣本的平均唯一性
    weights = np.ones(n)
    for i in valid_indices:
        end = min(i + max_holding, n - 1)
        avg_uniqueness = np.mean(1.0 / concurrency[i:end + 1])
        weights[i] = avg_uniqueness

    return pd.Series(weights, index=df.index, name="sample_weight")
