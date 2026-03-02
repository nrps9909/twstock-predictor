"""資料模組 — 為各模型提供統一資料介面

支援：
- LSTM 序列資料
- XGBoost 表格資料
- TFT TimeSeriesDataSet 格式
- 多股票批量訓練
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analysis.features import FeatureEngineer, FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """資料切分結果"""
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame
    feature_cols: list[str]


class StockDataModule:
    """統一資料管線

    負責：
    1. 特徵建立
    2. 時間序列切分（train/val/test）
    3. 為不同模型格式化資料
    """

    def __init__(self, stock_id: str):
        self.stock_id = stock_id
        self.feature_eng = FeatureEngineer()

    def prepare(
        self,
        start_date: str,
        end_date: str,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
    ) -> DataSplit:
        """建立特徵並切分資料"""
        df = self.feature_eng.build_features(self.stock_id, start_date, end_date)
        if df.empty:
            raise ValueError(f"無法建立 {self.stock_id} 特徵資料")

        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        n = len(df)
        train_end = int(n * (1 - val_ratio - test_ratio))
        val_end = int(n * (1 - test_ratio))

        return DataSplit(
            df_train=df.iloc[:train_end],
            df_val=df.iloc[train_end:val_end],
            df_test=df.iloc[val_end:],
            feature_cols=feature_cols,
        )

    def get_lstm_data(
        self, split: DataSplit, seq_len: int = 60
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """取得 LSTM 格式資料"""
        return {
            "train": self.feature_eng.prepare_sequences(split.df_train, seq_len, split.feature_cols),
            "val": self.feature_eng.prepare_sequences(split.df_val, seq_len, split.feature_cols),
            "test": self.feature_eng.prepare_sequences(split.df_test, seq_len, split.feature_cols),
        }

    def get_tabular_data(
        self, split: DataSplit
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """取得 XGBoost 格式資料"""
        return {
            "train": self.feature_eng.prepare_tabular(split.df_train, split.feature_cols),
            "val": self.feature_eng.prepare_tabular(split.df_val, split.feature_cols),
            "test": self.feature_eng.prepare_tabular(split.df_test, split.feature_cols),
        }

    def get_tft_dataframe(self, split: DataSplit) -> dict[str, pd.DataFrame]:
        """取得 TFT 格式 DataFrame（帶 time_idx + group_id）"""
        result = {}
        for name, df_part in [("train", split.df_train), ("val", split.df_val), ("test", split.df_test)]:
            tft_df = df_part[split.feature_cols + [TARGET_COLUMN]].copy()
            tft_df = tft_df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
            tft_df["time_idx"] = range(len(tft_df))
            tft_df["group_id"] = self.stock_id
            result[name] = tft_df
        return result
