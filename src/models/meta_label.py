"""Meta-Labeling 二階段模型

Stage 1 (主模型): LSTM/XGBoost 預測方向
Stage 2 (Meta-Labeler): 預測主模型「是否正確」→ 校準機率 → 調整倉位

參考: Marcos López de Prado, AFML Chapter 3
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class MetaLabeler:
    """Meta-Labeling 倉位校準器

    輸入：主模型預測 + 輔助特徵
    輸出：校準後的機率（主模型方向是否正確）→ 動態倉位大小
    """

    def __init__(self, max_position: float = 0.20):
        self.max_position = max_position
        self.model = None
        self.is_fitted = False

    def prepare_meta_features(
        self,
        primary_pred: np.ndarray,
        signal_strength: np.ndarray,
        hmm_probs: np.ndarray | None = None,
        volatility: np.ndarray | None = None,
    ) -> np.ndarray:
        """準備 meta-labeling 特徵

        Args:
            primary_pred: 主模型預測值（回歸或分類）
            signal_strength: 信號強度 (0-1)
            hmm_probs: HMM 狀態機率 (n, 3)
            volatility: 波動率序列

        Returns:
            meta_features (n, d)
        """
        n = len(primary_pred)
        features = [
            primary_pred.reshape(-1, 1),
            np.abs(primary_pred).reshape(-1, 1),  # 信號絕對值
            signal_strength.reshape(-1, 1),
        ]

        if hmm_probs is not None and len(hmm_probs) == n:
            features.append(
                np.atleast_2d(hmm_probs)
                if hmm_probs.ndim == 2
                else hmm_probs.reshape(-1, 1)
            )

        if volatility is not None and len(volatility) == n:
            features.append(volatility.reshape(-1, 1))

        return np.hstack(features).astype(np.float32)

    def prepare_meta_labels(
        self,
        primary_pred: np.ndarray,
        actual_returns: np.ndarray,
    ) -> np.ndarray:
        """準備 meta 標籤

        Meta label = 1 if 主模型方向正確, else 0

        Args:
            primary_pred: 主模型預測方向
            actual_returns: 實際報酬率

        Returns:
            binary labels (n,)
        """
        pred_direction = np.sign(primary_pred)
        actual_direction = np.sign(actual_returns)
        return (pred_direction == actual_direction).astype(np.float32)

    def fit(
        self,
        meta_features: np.ndarray,
        meta_labels: np.ndarray,
    ):
        """訓練 meta-labeler

        使用 CalibratedClassifierCV(GBM, isotonic) 產生校準機率
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV

        base_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )

        self.model = CalibratedClassifierCV(
            base_model,
            method="isotonic",
            cv=3,
        )

        # 確保標籤為整數
        labels = meta_labels.astype(int)

        # 需要至少兩個類別
        if len(np.unique(labels)) < 2:
            logger.warning("Meta labels 只有一個類別，跳過訓練")
            self.is_fitted = False
            return

        self.model.fit(meta_features, labels)
        self.is_fitted = True
        logger.info("Meta-labeler 訓練完成: %d 樣本", len(meta_labels))

    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        """預測主模型正確的機率

        Returns:
            calibrated probability (n,) — P(主模型方向正確)
        """
        if not self.is_fitted:
            return np.full(len(meta_features), 0.5)

        proba = self.model.predict_proba(meta_features)
        # 取正類（方向正確）的機率
        return proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]

    def bet_size(self, probabilities: np.ndarray) -> np.ndarray:
        """根據校準機率計算倉位大小

        Kelly-inspired sizing: size = max_pos * (2p - 1) if p > 0.5 else 0

        Args:
            probabilities: P(主模型正確) (n,)

        Returns:
            position sizes (n,) in [0, max_position]
        """
        sizes = np.where(
            probabilities > 0.5,
            self.max_position * (2 * probabilities - 1),
            0.0,
        )
        return np.clip(sizes, 0, self.max_position)

    def save(self, path: str | Path):
        """儲存 meta-labeler"""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "is_fitted": self.is_fitted,
                "max_position": self.max_position,
            },
            path,
        )
        logger.info("Meta-labeler 已儲存至 %s", path)

    def load(self, path: str | Path):
        """載入 meta-labeler"""
        import joblib

        path = Path(path)
        if not path.exists():
            logger.warning("Meta-labeler 檔案不存在: %s", path)
            return
        data = joblib.load(path)
        self.model = data["model"]
        self.is_fitted = data["is_fitted"]
        self.max_position = data["max_position"]
        logger.info("Meta-labeler 已載入自 %s", path)
