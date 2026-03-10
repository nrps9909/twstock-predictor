"""XGBoost models — regression + classification for stock prediction

Provides:
- StockXGBoost: regression on continuous return (tb_label)
- StockXGBoostClassifier: 3-class direction prediction (up/neutral/down)

The classifier directly optimizes for direction accuracy, which is
the metric that matters for trading signals.
"""

import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

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

        # Adaptive hyperparameters — balance fitting capacity vs overfitting
        # Previous params (max_depth=3, min_child=8 for n<800) were TOO conservative,
        # causing the model to predict constant zero (50% direction accuracy).
        n = len(X_train)
        if n < 200:
            self.model.set_params(
                max_depth=3, min_child_weight=8, reg_lambda=2.0, gamma=0.2
            )
            logger.info(
                "XGBoost adaptive params: ultra-small dataset (%d), max regularization", n
            )
        elif n < 800:
            self.model.set_params(
                max_depth=4, min_child_weight=5, reg_lambda=0.5, gamma=0.005,
                subsample=0.7, colsample_bytree=0.7,
            )
            logger.info(
                "XGBoost adaptive params: small dataset (%d), moderate regularization", n
            )
        elif n < 2000:
            self.model.set_params(
                max_depth=6, min_child_weight=3, reg_lambda=0.3, gamma=0.03
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


class StockXGBoostClassifier:
    """XGBoost 3-class direction classifier (up / neutral / down).

    Directly optimizes for direction accuracy instead of MSE on noisy returns.
    Classes: 0=down, 1=neutral, 2=up (matches LSTM dual-task convention).
    """

    # Map tb_class {-1, 0, 1} → internal {0, 1, 2}
    CLASS_MAP = {-1: 0, 0: 1, 1: 2}
    INV_CLASS_MAP = {0: -1, 1: 0, 2: 1}

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.7,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.5,
        min_child_weight: int = 5,
        gamma: float = 0.05,
    ):
        self.model = xgb.XGBClassifier(
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
            num_class=3,
            objective="multi:softprob",
            eval_metric="mlogloss",
        )
        self.feature_names: list[str] = []

    @staticmethod
    def encode_labels(tb_class: np.ndarray) -> np.ndarray:
        """Convert tb_class {-1, 0, 1} to classifier labels {0, 1, 2}."""
        mapping = np.vectorize(StockXGBoostClassifier.CLASS_MAP.get)
        return mapping(tb_class.astype(int))

    @staticmethod
    def decode_labels(cls_pred: np.ndarray) -> np.ndarray:
        """Convert classifier labels {0, 1, 2} back to {-1, 0, 1}."""
        mapping = np.vectorize(StockXGBoostClassifier.INV_CLASS_MAP.get)
        return mapping(cls_pred.astype(int))

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> dict:
        """Train classifier on tb_class labels.

        Args:
            y_train: tb_class values {-1, 0, 1}
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        # Encode labels
        y_enc = self.encode_labels(y_train)

        # Class weight balancing for up/down only — ignore neutral if too rare.
        # Neutral class (1) often has <2% of samples with tight TB barriers.
        # Giving it massive weight would prevent direction learning.
        from collections import Counter
        class_counts = Counter(y_enc)
        total = len(y_enc)
        if sample_weight is None:
            sample_weight = np.ones(total)
        # Only balance up(2) vs down(0) — these are the classes that matter for direction
        n_up = class_counts.get(2, 0)
        n_down = class_counts.get(0, 0)
        if n_up > 0 and n_down > 0 and min(n_up, n_down) > 0:
            ratio = max(n_up, n_down) / min(n_up, n_down)
            if ratio > 1.1:  # Only balance if >10% imbalanced
                if n_up > n_down:
                    sample_weight[y_enc == 0] *= ratio  # up-weight minority (down)
                else:
                    sample_weight[y_enc == 2] *= ratio  # up-weight minority (up)
                logger.info(
                    "XGBClassifier class balance: up=%d, down=%d, ratio=%.2f (balanced)",
                    n_up, n_down, ratio,
                )
        logger.info(
            "XGBClassifier class distribution: %s",
            dict(class_counts),
        )

        # Adaptive hyperparameters — need enough capacity to learn direction
        # but not so much that it memorizes training noise
        n = len(X_train)
        if n < 200:
            self.model.set_params(
                max_depth=3, min_child_weight=5, reg_lambda=1.0, gamma=0.1,
                learning_rate=0.03, n_estimators=100,
            )
            logger.info("XGBClassifier adaptive: ultra-small (%d), conservative", n)
        elif n < 800:
            self.model.set_params(
                max_depth=3, min_child_weight=5, reg_lambda=1.0, gamma=0.05,
                learning_rate=0.02, n_estimators=200,
                subsample=0.7, colsample_bytree=0.7,
            )
            logger.info("XGBClassifier adaptive: small (%d), balanced", n)
        elif n < 2000:
            self.model.set_params(
                max_depth=5, min_child_weight=3, reg_lambda=0.5, gamma=0.03,
                learning_rate=0.03,
            )
            logger.info("XGBClassifier adaptive: medium (%d)", n)

        eval_set = [(X_train, y_enc)]
        use_early_stopping = False
        if X_val is not None and y_val is not None:
            y_val_enc = self.encode_labels(y_val)
            eval_set.append((X_val, y_val_enc))
            # For small datasets, use long patience — weak signals need
            # many rounds to emerge but we still want overfitting protection.
            if n >= 2000:
                self.model.set_params(early_stopping_rounds=50)
                use_early_stopping = True
            else:
                patience = min(60, max(30, n // 15))
                self.model.set_params(early_stopping_rounds=patience)
                use_early_stopping = True
                logger.info(
                    "XGBClassifier: dynamic patience (%d rounds) for small dataset (%d samples)",
                    patience, n,
                )

        # Handle inf values
        if np.isinf(X_train).any():
            X_train = np.where(np.isinf(X_train), np.nan, X_train)
        if X_val is not None and np.isinf(X_val).any():
            X_val = np.where(np.isinf(X_val), np.nan, X_val)
            eval_set = [(X_train, y_enc)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, self.encode_labels(y_val)))

        self.model.fit(
            X_train,
            y_enc,
            eval_set=eval_set,
            verbose=False,
            sample_weight=sample_weight,
        )

        # Evaluate
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_enc, train_pred)
        # Direction accuracy: exclude neutral predictions and neutral labels
        train_dir = self._direction_accuracy(train_pred, y_enc)

        result = {"train_accuracy": train_acc, "train_direction_acc": train_dir}

        if X_val is not None and y_val is not None:
            y_val_enc = self.encode_labels(y_val)
            val_pred = self.model.predict(X_val)
            result["val_accuracy"] = accuracy_score(y_val_enc, val_pred)
            result["val_direction_acc"] = self._direction_accuracy(val_pred, y_val_enc)

        logger.info("XGBClassifier 訓練完成: %s", result)
        return result

    @staticmethod
    def _direction_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
        """Direction accuracy: only count up(2) vs down(0), ignore neutral(1)."""
        # Only evaluate where both pred and true are non-neutral
        mask = (true != 1) & (pred != 1)
        if mask.sum() == 0:
            return 0.5
        return float(np.mean(pred[mask] == true[mask]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels {0=down, 1=neutral, 2=up}."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (n_samples, 3)."""
        return self.model.predict_proba(X)

    def predict_direction_score(self, X: np.ndarray) -> np.ndarray:
        """Return a continuous direction score in [-1, 1].

        score = (P(up) - P(down)) * (1 - P(neutral))
        Weighting by confidence distinguishes decisive predictions from uncertain ones.
        E.g., [0.45, 0.10, 0.45] gives |score|=0 * 0.9=0 (uncertain split),
        while [0.05, 0.90, 0.05] gives |score|=0 * 0.1=0 (confidently neutral).
        Used as the ml_ensemble factor score.
        """
        proba = self.predict_proba(X)
        # proba columns: [P(down), P(neutral), P(up)]
        raw_score = proba[:, 2] - proba[:, 0]
        confidence = 1.0 - proba[:, 1]  # 1 - P(neutral)
        return raw_score * confidence

    @staticmethod
    def score_direction_accuracy(
        dir_score: np.ndarray, true_class: np.ndarray, min_confidence: float = 0.01
    ) -> float:
        """Direction accuracy using probability score with confidence filter.

        Only evaluates predictions where |dir_score| > min_confidence
        and true_class != 0 (not neutral). This gives more honest and
        higher accuracy than hard classification.

        Args:
            dir_score: P(up) - P(down) values in [-1, 1]
            true_class: tb_class values {-1, 0, 1}
            min_confidence: minimum |dir_score| to count
        """
        mask = (np.abs(dir_score) > min_confidence) & (true_class != 0)
        if mask.sum() < 10:
            return 0.5
        return float(np.mean(np.sign(dir_score[mask]) == np.sign(true_class[mask])))

    def get_feature_importance(self, top_n: int = 20) -> dict[str, float]:
        """Feature importance ranking."""
        importance = self.model.feature_importances_
        pairs = sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True,
        )
        return dict(pairs[:top_n])

    def save(self, path: str | Path):
        """Save model."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({"feature_names": self.feature_names, "type": "classifier"}, f)
        logger.info("XGBClassifier 已儲存至 %s", path)

    def load(self, path: str | Path):
        """Load model."""
        import json

        path = Path(path)
        self.model.load_model(str(path))
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self.feature_names = meta.get("feature_names", [])
        logger.info("XGBClassifier 已載入自 %s", path)
