"""特徵重要性漂移偵測

PSI (Population Stability Index) 偵測特徵分佈漂移，
追蹤特徵重要性時序變化。
"""

import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureDriftDetector:
    """特徵漂移偵測器"""

    PSI_THRESHOLD_LOW = 0.10    # < 0.10: 穩定
    PSI_THRESHOLD_HIGH = 0.25   # > 0.25: 顯著漂移

    def __init__(self, session_factory=None):
        self.session_factory = session_factory

    def compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """計算 Population Stability Index

        PSI = Σ (actual% - expected%) * ln(actual% / expected%)

        Args:
            expected: 歷史（參考）分佈
            actual: 當前分佈
            n_bins: 分箱數

        Returns:
            PSI 值（0 = 無漂移，> 0.25 = 顯著漂移）
        """
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(actual, dtype=float)

        # 移除 NaN
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) < n_bins or len(actual) < n_bins:
            return 0.0

        # 使用 expected 的分位數作為 bin edges
        bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        expected_counts = np.histogram(expected, bins=bins)[0]
        actual_counts = np.histogram(actual, bins=bins)[0]

        # 轉為比例，避免零除
        eps = 1e-6
        expected_pct = expected_counts / len(expected) + eps
        actual_pct = actual_counts / len(actual) + eps

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def compute_drift(
        self,
        current_features: pd.DataFrame,
        historical_features: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> dict[str, float]:
        """計算每個特徵的 PSI 漂移分數

        Args:
            current_features: 當前視窗的特徵 DataFrame
            historical_features: 歷史參考期的特徵 DataFrame
            feature_cols: 要檢查的特徵欄位

        Returns:
            {feature_name: psi_score}
        """
        if feature_cols is None:
            feature_cols = [
                c for c in current_features.columns
                if c in historical_features.columns
                and current_features[c].dtype in [np.float64, np.float32, np.int64]
            ]

        drift_scores = {}
        for col in feature_cols:
            if col not in current_features.columns or col not in historical_features.columns:
                continue
            psi = self.compute_psi(
                historical_features[col].values,
                current_features[col].values,
            )
            drift_scores[col] = psi

        # 排序：漂移最嚴重的在前
        drift_scores = dict(sorted(drift_scores.items(), key=lambda x: x[1], reverse=True))

        drifted = {k: v for k, v in drift_scores.items() if v > self.PSI_THRESHOLD_HIGH}
        if drifted:
            logger.warning("特徵漂移偵測: %d 個特徵 PSI > %.2f: %s",
                           len(drifted), self.PSI_THRESHOLD_HIGH, drifted)

        return drift_scores

    def get_importance_history(
        self,
        stock_id: str,
        feature_name: str,
        limit: int = 30,
    ) -> list[dict]:
        """取得單一特徵的重要性歷史時序

        Args:
            stock_id: 股票代碼
            feature_name: 特徵名稱
            limit: 最多回傳幾筆

        Returns:
            list of {run_date, importance_score, method, rank}
        """
        if self.session_factory is None:
            logger.warning("session_factory 未設定，無法查詢歷史")
            return []

        from sqlalchemy import select
        from src.db.models import FeatureImportanceRecord

        session = self.session_factory()
        try:
            stmt = (
                select(FeatureImportanceRecord)
                .where(
                    FeatureImportanceRecord.stock_id == stock_id,
                    FeatureImportanceRecord.feature_name == feature_name,
                )
                .order_by(FeatureImportanceRecord.run_date.desc())
                .limit(limit)
            )
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "run_date": r.run_date,
                    "importance_score": r.importance_score,
                    "method": r.method,
                    "rank": r.rank,
                }
                for r in reversed(rows)
            ]
        finally:
            session.close()

    def get_max_psi(self, drift_scores: dict[str, float]) -> float:
        """取得最大 PSI 值"""
        if not drift_scores:
            return 0.0
        return max(drift_scores.values())

    def is_drifted(self, drift_scores: dict[str, float]) -> bool:
        """是否有任何特徵顯著漂移"""
        return self.get_max_psi(drift_scores) > self.PSI_THRESHOLD_HIGH
