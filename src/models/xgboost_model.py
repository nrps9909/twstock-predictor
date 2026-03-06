"""XGBoost 特徵模型 — 使用所有特徵預測未來報酬率

優勢：
- 對特徵重要性解釋能力強
- 訓練速度快
- 對缺失值有內建處理
"""

import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.models.ensemble import direction_accuracy

logger = logging.getLogger(__name__)


class StockXGBoost:
    """XGBoost 回歸模型"""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        min_child_weight: int = 5,
        gamma: float = 0.1,
    ):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            tree_method="hist",
            random_state=42,
        )
        self.feature_names: list[str] = []

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> dict:
        """訓練模型

        Args:
            sample_weight: 樣本唯一性權重（來自 Triple Barrier 的 average uniqueness）

        Returns:
            {"train_mse": float, "val_mse": float | None}
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        # Adaptive hyperparameters based on sample size
        n = len(X_train)
        if n < 800:
            self.model.set_params(
                max_depth=5, min_child_weight=3, reg_lambda=0.5, gamma=0.05
            )
            logger.info(
                "XGBoost adaptive params: small dataset (%d), relaxed regularization", n
            )
        elif n < 2000:
            self.model.set_params(
                max_depth=5, min_child_weight=4, reg_lambda=0.8, gamma=0.08
            )
            logger.info("XGBoost adaptive params: medium dataset (%d)", n)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # XGBoost 3.x: early_stopping_rounds via set_params
        if X_val is not None and y_val is not None:
            self.model.set_params(early_stopping_rounds=30)

        # XGBoost handles NaN natively but crashes on inf
        if np.isinf(X_train).any():
            logger.warning(
                "XGBoost: %d inf values in X_train, replacing with NaN",
                np.isinf(X_train).sum(),
            )
            X_train = np.where(np.isinf(X_train), np.nan, X_train)
        if X_val is not None and np.isinf(X_val).any():
            X_val = np.where(np.isinf(X_val), np.nan, X_val)
            # Rebuild eval_set after sanitization
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
            sample_weight=sample_weight,
        )

        if sample_weight is not None:
            logger.info(
                "XGBoost 使用樣本權重: mean=%.3f, std=%.3f",
                np.mean(sample_weight),
                np.std(sample_weight),
            )

        if X_val is not None and y_val is not None:
            best_iter = self.model.best_iteration
            if best_iter > 0:
                logger.info("XGBoost early stopped at iteration %d", best_iter)

        # 評估
        train_pred = self.model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_pred)
        train_dir_acc = direction_accuracy(train_pred, y_train)

        result = {"train_mse": train_mse, "train_direction_acc": train_dir_acc}

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            result["val_mse"] = mean_squared_error(y_val, val_pred)
            result["val_mae"] = mean_absolute_error(y_val, val_pred)
            result["val_direction_acc"] = direction_accuracy(val_pred, y_val)

        # Naive baseline MSE (predict 0 = hold)
        result["naive_mse"] = float(np.mean(y_train**2))

        logger.info("XGBoost 訓練完成: %s", result)
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測"""
        return self.model.predict(X)

    def get_feature_importance(self, top_n: int = 20) -> dict[str, float]:
        """取得特徵重要性排名

        Returns:
            {feature_name: importance_score} 由高到低排序
        """
        importance = self.model.feature_importances_
        pairs = sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True,
        )
        return dict(pairs[:top_n])

    def save(self, path: str | Path):
        """儲存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        # 同時保存特徵名稱
        import json

        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({"feature_names": self.feature_names}, f)
        logger.info("XGBoost 模型已儲存至 %s", path)

    def load(self, path: str | Path):
        """載入模型"""
        path = Path(path)
        self.model.load_model(str(path))
        import json

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self.feature_names = meta.get("feature_names", [])
        logger.info("XGBoost 模型已載入自 %s", path)
