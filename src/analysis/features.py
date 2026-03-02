"""特徵工程模組 — 合併所有資料為模型訓練特徵矩陣

Tier 2 增強：波動率、微結構、日曆、跨資產特徵
支援 Triple Barrier 標籤 + SHAP/MI 特徵篩選
"""

import logging
from datetime import date as dt_date

import numpy as np
import pandas as pd

from src.analysis.technical import TechnicalAnalyzer
from src.analysis.labels import triple_barrier_label, compute_sample_weights
from src.db.database import get_stock_prices, get_sentiment
from src.utils.config import settings

logger = logging.getLogger(__name__)

# 完整特徵欄位清單（含 Tier 2 新增）
FEATURE_COLUMNS = [
    # 價格特徵
    "close", "open", "high", "low", "volume",
    "return_1d", "return_5d", "return_20d",

    # 技術指標
    "sma_5", "sma_20", "sma_60",
    "rsi_14", "kd_k", "kd_d",
    "macd", "macd_signal", "macd_hist",
    "bias_5", "bias_10", "bias_20",
    "bb_upper", "bb_lower", "bb_width",
    "obv", "adx",

    # 情緒特徵
    "sentiment_score", "sentiment_ma5",
    "sentiment_change", "post_volume", "bullish_ratio",

    # 籌碼特徵
    "foreign_buy_sell", "trust_buy_sell", "dealer_buy_sell",
    "margin_balance", "short_balance",

    # Tier 2: 波動率特徵
    "realized_vol_5d", "realized_vol_20d", "parkinson_vol",

    # Tier 2: 微結構特徵
    "volume_ratio_5d", "spread_proxy",

    # Tier 2: 日曆特徵
    "day_of_week", "month", "is_settlement",
]

# Legacy target (保留向後相容)
TARGET_COLUMN = "return_next_5d"

# Triple Barrier target
TB_TARGET_COLUMN = "tb_label"
TB_WEIGHT_COLUMN = "sample_weight"

# 台灣期貨結算日（每月第三個星期三）
def _is_settlement_day(d: dt_date) -> bool:
    """判斷是否為期貨結算日（每月第三個星期三）"""
    if d.weekday() != 2:  # 不是星期三
        return False
    # 第三個星期三 = 日期在 15-21 之間
    return 15 <= d.day <= 21


class FeatureEngineer:
    """特徵工程處理器"""

    def __init__(self):
        self.ta = TechnicalAnalyzer()

    def build_features(
        self,
        stock_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """從 DB 讀取資料，計算所有特徵，合併為訓練用矩陣

        Returns:
            DataFrame，含所有特徵欄位 + target
        """
        # 讀取股價
        price_df = get_stock_prices(
            stock_id,
            dt_date.fromisoformat(start_date),
            dt_date.fromisoformat(end_date),
        )
        if price_df.empty:
            logger.error("無法取得 %s 股價資料", stock_id)
            return pd.DataFrame()

        # 計算技術指標
        df = self.ta.compute_all(price_df)

        # 讀取情緒資料
        sentiment_df = get_sentiment(
            stock_id,
            dt_date.fromisoformat(start_date),
            dt_date.fromisoformat(end_date),
        )
        df = self._merge_sentiment(df, sentiment_df)

        # Tier 2: 計算增強特徵
        df = self._add_volatility_features(df)
        df = self._add_microstructure_features(df)
        df = self._add_calendar_features(df)

        # 計算 target: 未來 5 日報酬率（legacy，保留向後相容）
        df[TARGET_COLUMN] = df["close"].pct_change(5).shift(-5)

        # Triple Barrier 標籤（主要 target）
        df[TB_TARGET_COLUMN] = triple_barrier_label(
            df,
            upper_multiplier=2.0,
            lower_multiplier=2.0,
            max_holding=10,
            atr_window=14,
        )

        # 樣本唯一性權重
        df[TB_WEIGHT_COLUMN] = compute_sample_weights(
            df, label_col=TB_TARGET_COLUMN, max_holding=10
        )

        # 填充缺失值
        df = self._fill_missing(df)

        return df

    # ── Tier 2: 新增特徵 ─────────────────────────────────

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """波動率特徵"""
        if "return_1d" not in df.columns:
            df["return_1d"] = df["close"].pct_change()

        # Realized volatility（歷史波動率）
        df["realized_vol_5d"] = df["return_1d"].rolling(5).std()
        df["realized_vol_20d"] = df["return_1d"].rolling(20).std()

        # Parkinson high-low volatility estimator
        if "high" in df.columns and "low" in df.columns:
            log_hl = np.log(df["high"] / df["low"])
            df["parkinson_vol"] = (log_hl ** 2 / (4 * np.log(2))).rolling(20).mean().apply(np.sqrt)
        else:
            df["parkinson_vol"] = 0.0

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """微結構特徵"""
        if "volume" in df.columns:
            vol_ma5 = df["volume"].rolling(5).mean()
            df["volume_ratio_5d"] = df["volume"] / vol_ma5.replace(0, np.nan)
        else:
            df["volume_ratio_5d"] = 1.0

        # Spread proxy（用 high-low 近似 bid-ask spread）
        if "high" in df.columns and "low" in df.columns and "close" in df.columns:
            df["spread_proxy"] = (df["high"] - df["low"]) / df["close"]
        else:
            df["spread_proxy"] = 0.0

        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """日曆特徵"""
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            df["day_of_week"] = dates.dt.dayofweek
            df["month"] = dates.dt.month
            df["is_settlement"] = dates.apply(
                lambda d: 1.0 if _is_settlement_day(d.date()) else 0.0
            )
        else:
            df["day_of_week"] = 0
            df["month"] = 1
            df["is_settlement"] = 0.0

        return df

    # ── 合併與填充 ──────────────────────────────────────

    def _merge_sentiment(
        self, price_df: pd.DataFrame, sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """合併情緒資料到價格 DataFrame"""
        if sentiment_df.empty:
            # 無情緒資料時填 0
            price_df["sentiment_score"] = 0.0
            price_df["sentiment_ma5"] = 0.0
            price_df["sentiment_change"] = 0.0
            price_df["post_volume"] = 0
            price_df["bullish_ratio"] = 0.0
            return price_df

        # 每日聚合情緒
        daily_sent = sentiment_df.groupby("date").agg(
            sentiment_score=("sentiment_score", "mean"),
            post_volume=("sentiment_score", "count"),
            bullish_ratio=(
                "sentiment_label",
                lambda x: (x == "bullish").mean(),
            ),
        ).reset_index()

        daily_sent["sentiment_ma5"] = (
            daily_sent["sentiment_score"].rolling(5, min_periods=1).mean()
        )
        daily_sent["sentiment_change"] = daily_sent["sentiment_score"].diff()

        # Merge
        df = price_df.merge(daily_sent, on="date", how="left")
        return df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """處理缺失值"""
        # 情緒欄位填 0
        for col in ["sentiment_score", "sentiment_ma5", "sentiment_change",
                     "post_volume", "bullish_ratio"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 籌碼欄位前填
        for col in ["foreign_buy_sell", "trust_buy_sell", "dealer_buy_sell",
                     "margin_balance", "short_balance"]:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(0)

        # 新增特徵填 0
        for col in ["realized_vol_5d", "realized_vol_20d", "parkinson_vol",
                     "volume_ratio_5d", "spread_proxy",
                     "day_of_week", "month", "is_settlement"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 技術指標 NaN（前面暖身期）→ 刪除
        df = df.dropna(subset=["sma_60"]).reset_index(drop=True)

        return df

    def select_features(
        self,
        df: pd.DataFrame,
        target_col: str = TB_TARGET_COLUMN,
        max_features: int = 20,
        method: str = "mutual_info",
        session_factory=None,
        stock_id: str | None = None,
    ) -> list[str]:
        """特徵篩選 — 從全部特徵中選出最重要的 max_features 個

        Args:
            df: 含特徵 + target 的 DataFrame
            target_col: 目標欄位
            max_features: 最多保留幾個特徵
            method: 'mutual_info' (互信息) 或 'shap' (SHAP 重要性)
            session_factory: DB session 工廠（可選，用於持久化重要性分數）
            stock_id: 股票代碼（可選，搭配 session_factory 使用）

        Returns:
            篩選後的特徵欄位清單
        """
        available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        valid = df.dropna(subset=[target_col])

        if len(valid) < 50 or len(available_cols) <= max_features:
            logger.info("特徵數 (%d) <= max (%d)，跳過篩選", len(available_cols), max_features)
            return available_cols

        X = valid[available_cols].values.astype(np.float32)
        y = valid[target_col].values.astype(np.float32)

        # 替換 NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if method == "shap":
            selected = self._select_by_shap(X, y, available_cols, max_features)
        else:
            selected = self._select_by_mi(X, y, available_cols, max_features)

        # 持久化重要性分數
        if session_factory is not None and stock_id is not None:
            self._persist_importance(
                X, y, available_cols, method, session_factory, stock_id,
            )

        # 移除高度共線特徵
        selected = self._remove_collinear(df, selected, threshold=0.95)

        logger.info(
            "特徵篩選: %d → %d (%s): %s",
            len(available_cols), len(selected), method, selected,
        )
        return selected

    def _persist_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        method: str,
        session_factory,
        stock_id: str,
    ):
        """持久化特徵重要性分數到 DB

        先刪除同 (stock_id, run_date, method) 的舊紀錄，再批次 INSERT。
        """
        try:
            from sqlalchemy import delete
            from src.db.models import FeatureImportanceRecord
            from datetime import date as dt_date_cls

            if method == "shap":
                scores = self._compute_shap_scores(X, y)
            else:
                from sklearn.feature_selection import mutual_info_regression
                scores = mutual_info_regression(X, y, random_state=42)

            ranked_idx = np.argsort(scores)[::-1]
            run_date = dt_date_cls.today()

            session = session_factory()
            try:
                # 先刪除同日同方法的舊紀錄，避免重複
                session.execute(
                    delete(FeatureImportanceRecord).where(
                        FeatureImportanceRecord.stock_id == stock_id,
                        FeatureImportanceRecord.run_date == run_date,
                        FeatureImportanceRecord.method == method,
                    )
                )
                for rank, idx in enumerate(ranked_idx, 1):
                    record = FeatureImportanceRecord(
                        stock_id=stock_id,
                        run_date=run_date,
                        feature_name=feature_names[idx],
                        importance_score=float(scores[idx]),
                        method=method,
                        rank=rank,
                    )
                    session.add(record)
                session.commit()
                logger.info("已持久化 %d 個特徵重要性分數", len(feature_names))
            except Exception as e:
                session.rollback()
                logger.warning("持久化特徵重要性失敗: %s", e)
            finally:
                session.close()
        except Exception as e:
            logger.warning("持久化特徵重要性時發生錯誤: %s", e)

    def _compute_shap_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """計算 SHAP 重要性分數"""
        try:
            import xgboost as xgb
            import shap
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42, verbosity=0,
            )
            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[:min(500, len(X))])
            return np.abs(shap_values).mean(axis=0)
        except ImportError:
            from sklearn.feature_selection import mutual_info_regression
            return mutual_info_regression(X, y, random_state=42)

    def _select_by_mi(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        max_features: int,
    ) -> list[str]:
        """用 Mutual Information 篩選特徵"""
        from sklearn.feature_selection import mutual_info_regression

        mi_scores = mutual_info_regression(X, y, random_state=42)
        ranked_idx = np.argsort(mi_scores)[::-1][:max_features]
        return [feature_names[i] for i in ranked_idx]

    def _select_by_shap(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        max_features: int,
    ) -> list[str]:
        """用 SHAP 重要性篩選特徵（以輕量 XGBoost 為基礎模型）"""
        try:
            import xgboost as xgb
            import shap

            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42, verbosity=0,
            )
            model.fit(X, y)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[:min(500, len(X))])
            importance = np.abs(shap_values).mean(axis=0)
            ranked_idx = np.argsort(importance)[::-1][:max_features]
            return [feature_names[i] for i in ranked_idx]
        except ImportError:
            logger.warning("shap 未安裝，fallback 到 mutual_info")
            return self._select_by_mi(X, y, feature_names, max_features)

    def _remove_collinear(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        threshold: float = 0.95,
    ) -> list[str]:
        """移除高度共線的特徵（保留 MI/SHAP 排名較前的）"""
        if len(feature_cols) <= 2:
            return feature_cols

        valid = df[feature_cols].dropna()
        if len(valid) < 10:
            return feature_cols

        corr_matrix = valid.corr().abs()
        to_drop = set()
        for i in range(len(feature_cols)):
            if feature_cols[i] in to_drop:
                continue
            for j in range(i + 1, len(feature_cols)):
                if feature_cols[j] in to_drop:
                    continue
                if corr_matrix.iloc[i, j] > threshold:
                    # 移除排名靠後的（index 大的 = MI/SHAP 排名低）
                    to_drop.add(feature_cols[j])

        result = [c for c in feature_cols if c not in to_drop]
        if to_drop:
            logger.info("移除共線特徵: %s", to_drop)
        return result

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        seq_len: int = 60,
        feature_cols: list[str] | None = None,
        target_col: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """為 LSTM 準備序列資料

        Args:
            df: 含特徵 + target 的 DataFrame
            seq_len: 序列長度（回看天數）
            feature_cols: 使用的特徵欄位
            target_col: 目標欄位（預設使用 TB_TARGET_COLUMN，fallback TARGET_COLUMN）

        Returns:
            (X, y) where X.shape = (samples, seq_len, features), y.shape = (samples,)
        """
        if feature_cols is None:
            feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

        if target_col is None:
            target_col = TB_TARGET_COLUMN if TB_TARGET_COLUMN in df.columns else TARGET_COLUMN

        # 正規化特徵
        feature_data = df[feature_cols].values
        target_data = df[target_col].values

        X, y = [], []
        for i in range(seq_len, len(df)):
            if np.isnan(target_data[i]):
                continue
            X.append(feature_data[i - seq_len : i])
            y.append(target_data[i])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def prepare_tabular(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        target_col: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """為 XGBoost 準備表格資料

        Returns:
            (X, y) where X.shape = (samples, features), y.shape = (samples,)
        """
        if feature_cols is None:
            feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

        if target_col is None:
            target_col = TB_TARGET_COLUMN if TB_TARGET_COLUMN in df.columns else TARGET_COLUMN

        valid = df.dropna(subset=[target_col])
        X = valid[feature_cols].values.astype(np.float32)
        y = valid[target_col].values.astype(np.float32)

        return X, y
