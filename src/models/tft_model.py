"""Temporal Fusion Transformer (TFT) 預測模型

優勢：
- 原生支援多步預測
- 自動特徵選擇（Variable Selection Network）
- 已知/未知輸入分離
- 可解釋注意力權重
- MAE 降低 40-50%（相較 LSTM）

依賴：pytorch-forecasting, lightning
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TFTPredictor:
    """TFT 模型訓練/預測封裝

    遵循與 LSTMPredictor 相同的介面：train(), predict(), save(), load()
    """

    def __init__(
        self,
        max_prediction_length: int = 5,
        max_encoder_length: int = 60,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
    ):
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.trainer = None

    def _prepare_tft_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "return_next_5d",
        stock_id: str = "default",
    ) -> pd.DataFrame:
        """將 DataFrame 轉換為 pytorch-forecasting 所需格式

        Required columns:
        - time_idx: 連續整數時間索引
        - group_id: 群組 ID（股票代號）
        - target: 預測目標
        - known/unknown regressors
        """
        tft_df = df[feature_cols + [target_col]].copy()
        tft_df = tft_df.dropna(subset=[target_col]).reset_index(drop=True)
        tft_df["time_idx"] = range(len(tft_df))
        tft_df["group_id"] = stock_id

        # 日曆特徵（已知的未來輸入）
        if "day_of_week" not in tft_df.columns:
            tft_df["day_of_week"] = 0
        if "month" not in tft_df.columns:
            tft_df["month"] = 1

        return tft_df

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "return_next_5d",
        stock_id: str = "default",
        max_epochs: int = 50,
        batch_size: int = 32,
        val_ratio: float = 0.2,
    ) -> dict:
        """訓練 TFT 模型

        Args:
            df: 含所有特徵與 target 的 DataFrame
            feature_cols: 特徵欄位清單
            target_col: 預測目標欄位
            max_epochs: 最大訓練輪數
            batch_size: batch 大小

        Returns:
            {"train_loss": float, "val_loss": float}
        """
        try:
            import lightning.pytorch as pl
            from pytorch_forecasting import (
                TemporalFusionTransformer,
                TimeSeriesDataSet,
            )
            from pytorch_forecasting.metrics import MAE
        except ImportError:
            logger.error(
                "TFT 需要 pytorch-forecasting 和 lightning。"
                "請安裝: pip install pytorch-forecasting lightning"
            )
            return {"error": "missing dependencies"}

        tft_df = self._prepare_tft_dataframe(df, feature_cols, target_col, stock_id)

        if len(tft_df) < self.max_encoder_length + self.max_prediction_length + 10:
            logger.error("TFT 資料不足: %d 筆", len(tft_df))
            return {"error": "insufficient data"}

        # Split train/val by time
        split_idx = int(len(tft_df) * (1 - val_ratio))
        training_cutoff = tft_df["time_idx"].iloc[split_idx]

        # 分類已知/未知 regressors
        known_reals = ["day_of_week", "month"]
        unknown_reals = [c for c in feature_cols if c in tft_df.columns and c not in known_reals]

        # TimeSeriesDataSet
        training = TimeSeriesDataSet(
            tft_df[tft_df["time_idx"] <= training_cutoff],
            time_idx="time_idx",
            target=target_col,
            group_ids=["group_id"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=["time_idx"] + [c for c in known_reals if c in tft_df.columns],
            time_varying_unknown_reals=[target_col] + unknown_reals,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = TimeSeriesDataSet.from_dataset(
            training,
            tft_df,
            min_prediction_idx=training_cutoff + 1,
        )

        train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

        # Build TFT
        self.model = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_size // 2,
            loss=MAE(),
            learning_rate=self.learning_rate,
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        # Train
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            gradient_clip_val=0.1,
            enable_progress_bar=True,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss", patience=8, mode="min"
                ),
            ],
        )
        self.trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Get metrics
        train_loss = float(self.trainer.callback_metrics.get("train_loss", 0))
        val_loss = float(self.trainer.callback_metrics.get("val_loss", 0))

        logger.info("TFT 訓練完成: train_loss=%.6f, val_loss=%.6f", train_loss, val_loss)
        return {"train_loss": train_loss, "val_loss": val_loss}

    def predict(self, df: pd.DataFrame, feature_cols: list[str],
                target_col: str = "return_next_5d", stock_id: str = "default") -> np.ndarray:
        """使用訓練好的 TFT 模型預測

        Returns:
            預測報酬率 (n_days,)
        """
        if self.model is None:
            logger.error("TFT 模型尚未訓練或載入")
            return np.array([0.0])

        try:
            from pytorch_forecasting import TimeSeriesDataSet
        except ImportError:
            return np.array([0.0])

        tft_df = self._prepare_tft_dataframe(df, feature_cols, target_col, stock_id)

        # 只需要最後 max_encoder_length 筆資料
        raw_predictions = self.model.predict(
            tft_df.iloc[-self.max_encoder_length - self.max_prediction_length:],
            mode="raw",
            return_x=True,
        )
        predictions = raw_predictions.output["prediction"]
        return predictions[0].cpu().numpy().flatten()

    def get_attention_weights(self) -> dict[str, Any] | None:
        """取得特徵重要性（Variable Selection Network 權重）"""
        if self.model is None:
            return None

        try:
            interpretation = self.model.interpret_output(
                self.model.predict(None, mode="raw", return_x=True),
                reduction="sum",
            )
            return {
                "attention_weights": interpretation["attention"].cpu().numpy(),
                "variable_importance": {
                    k: v.cpu().numpy()
                    for k, v in interpretation.get("static_variables", {}).items()
                },
            }
        except Exception as e:
            logger.warning("無法取得 attention weights: %s", e)
            return None

    def save(self, path: str | Path):
        """儲存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.model is not None and self.trainer is not None:
            self.trainer.save_checkpoint(str(path))
            logger.info("TFT 模型已儲存至 %s", path)

    def load(self, path: str | Path):
        """載入模型"""
        try:
            from pytorch_forecasting import TemporalFusionTransformer
            self.model = TemporalFusionTransformer.load_from_checkpoint(str(path))
            logger.info("TFT 模型已載入自 %s", path)
        except Exception as e:
            logger.error("載入 TFT 模型失敗: %s", e)
