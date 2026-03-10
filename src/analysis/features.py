"""特徵工程模組 — 合併所有資料為模型訓練特徵矩陣

Tier 2 增強：波動率、微結構、日曆、跨資產特徵
支援 Triple Barrier 標籤 + SHAP/MI 特徵篩選
"""

import logging
from datetime import date as dt_date

import numpy as np
import pandas as pd

from src.analysis.technical import TechnicalAnalyzer
from src.analysis.labels import (
    triple_barrier_label,
    triple_barrier_classify,
    compute_sample_weights,
)
from src.db.database import get_stock_prices, get_sentiment

logger = logging.getLogger(__name__)

# 完整特徵欄位清單（含 Tier 2 新增）
FEATURE_COLUMNS = [
    # 價格特徵
    "close",
    "open",
    "high",
    "low",
    "volume",
    "return_1d",
    "return_5d",
    "return_20d",
    # 技術指標
    "sma_5",
    "sma_20",
    "sma_60",
    "rsi_14",
    "kd_k",
    "kd_d",
    "macd",
    "macd_signal",
    "macd_hist",
    "bias_5",
    "bias_10",
    "bias_20",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "obv",
    "adx",
    # 情緒特徵
    "sentiment_score",
    "sentiment_ma5",
    "sentiment_change",
    "post_volume",
    "bullish_ratio",
    # 籌碼特徵
    "foreign_buy_sell",
    "trust_buy_sell",
    "dealer_buy_sell",
    "margin_balance",
    "short_balance",
    # Tier 2: 波動率特徵
    "realized_vol_5d",
    "realized_vol_20d",
    "parkinson_vol",
    # Tier 2: 微結構特徵
    "volume_ratio_5d",
    "spread_proxy",
    # Tier 2: 日曆特徵
    "day_of_week",
    "month",
    "is_settlement",
    # Tier 3: 相對特徵（穩態化，取代絕對價格）
    "close_to_sma5",
    "close_to_sma20",
    "close_to_sma60",
    "bb_position",
    "volume_to_ma20",
    "margin_change_5d",
    "short_change_5d",
    "institutional_net_ratio",
    "momentum_12_1",
    "return_vol_ratio",
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

    def _sanitize_array(
        self, X: np.ndarray, clip_percentile: float = 99.5
    ) -> np.ndarray:
        """Replace inf/NaN and winsorize extreme values per feature column."""
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            if len(col) < 10:
                continue
            lo = np.percentile(col, 100 - clip_percentile)
            hi = np.percentile(col, clip_percentile)
            if hi > lo:
                X[:, col_idx] = np.clip(col, lo, hi)
        return X

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

        # Tier 3: 相對特徵（穩態化）
        df = self._add_relative_features(df)

        # 計算 target: 未來 5 日報酬率（legacy，保留向後相容）
        df[TARGET_COLUMN] = df["close"].pct_change(5).shift(-5)

        # Triple Barrier 標籤（主要 target）
        # ATR×1.0 + 7-day holding: shorter horizon = more barrier touches,
        # less noise accumulation, better match for technical feature predictability.
        df[TB_TARGET_COLUMN] = triple_barrier_label(
            df,
            upper_multiplier=1.0,
            lower_multiplier=1.0,
            max_holding=7,
            atr_window=14,
        )

        # Triple Barrier classification labels (for direction accuracy eval)
        df["tb_class"] = triple_barrier_classify(
            df,
            upper_multiplier=1.0,
            lower_multiplier=1.0,
            max_holding=7,
            atr_window=14,
        )

        # 樣本唯一性權重
        df[TB_WEIGHT_COLUMN] = compute_sample_weights(
            df, label_col=TB_TARGET_COLUMN, max_holding=7
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
            hl_ratio = (df["high"] / df["low"].replace(0, np.nan)).clip(lower=1.0)
            log_hl = np.log(hl_ratio)
            df["parkinson_vol"] = (
                (log_hl**2 / (4 * np.log(2))).rolling(20).mean().apply(np.sqrt)
            )
        else:
            df["parkinson_vol"] = 0.0

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """微結構特徵"""
        if "volume" in df.columns:
            vol_ma5 = df["volume"].rolling(5).mean()
            df["volume_ratio_5d"] = (df["volume"] / vol_ma5.replace(0, np.nan)).clip(
                upper=10.0
            )
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

    def _add_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 3: 相對特徵 — 穩態化取代絕對價格

        These are stationary transformations of price-level features,
        far more predictive for ML models than raw prices.
        """
        close = df["close"]

        # Price-to-SMA ratios (mean-reversion signals)
        if "sma_5" in df.columns:
            sma5 = df["sma_5"].replace(0, np.nan)
            df["close_to_sma5"] = (close / sma5 - 1).clip(-0.2, 0.2)
        else:
            df["close_to_sma5"] = 0.0

        if "sma_20" in df.columns:
            sma20 = df["sma_20"].replace(0, np.nan)
            df["close_to_sma20"] = (close / sma20 - 1).clip(-0.3, 0.3)
        else:
            df["close_to_sma20"] = 0.0

        if "sma_60" in df.columns:
            sma60 = df["sma_60"].replace(0, np.nan)
            df["close_to_sma60"] = (close / sma60 - 1).clip(-0.5, 0.5)
        else:
            df["close_to_sma60"] = 0.0

        # Bollinger band position (0=lower band, 1=upper band)
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
            df["bb_position"] = ((close - df["bb_lower"]) / bb_range).clip(-0.5, 1.5)
        else:
            df["bb_position"] = 0.5

        # Volume relative to 20-day average
        if "volume" in df.columns:
            vol_ma20 = df["volume"].rolling(20).mean().replace(0, np.nan)
            df["volume_to_ma20"] = (df["volume"] / vol_ma20).clip(0, 5.0)
        else:
            df["volume_to_ma20"] = 1.0

        # Margin balance change (leverage direction signal)
        if "margin_balance" in df.columns:
            df["margin_change_5d"] = (
                df["margin_balance"].pct_change(5, fill_method=None).clip(-1, 1)
            )
        else:
            df["margin_change_5d"] = 0.0

        # Short balance change (short selling pressure)
        if "short_balance" in df.columns:
            df["short_change_5d"] = (
                df["short_balance"].pct_change(5, fill_method=None).clip(-1, 1)
            )
        else:
            df["short_change_5d"] = 0.0

        # Institutional net buy ratio (most powerful signal for TWSE)
        if "foreign_buy_sell" in df.columns and "volume" in df.columns:
            trust = df.get("trust_buy_sell", pd.Series(0, index=df.index))
            if not isinstance(trust, pd.Series):
                trust = pd.Series(0, index=df.index)
            vol_safe = df["volume"].replace(0, np.nan)
            net_inst = df["foreign_buy_sell"] + trust
            df["institutional_net_ratio"] = (net_inst / vol_safe).clip(-1, 1)
        else:
            df["institutional_net_ratio"] = 0.0

        # Momentum factor: 12-month return minus 1-month return (classic Jegadeesh-Titman)
        if "return_1d" in df.columns:
            ret_240d = close.pct_change(240)
            ret_20d = close.pct_change(20)
            df["momentum_12_1"] = (ret_240d - ret_20d).clip(-1, 1)
        else:
            df["momentum_12_1"] = 0.0

        # Return-to-volatility ratio (Sharpe-like signal)
        if "return_5d" in df.columns and "realized_vol_5d" in df.columns:
            vol_safe = df["realized_vol_5d"].replace(0, np.nan)
            df["return_vol_ratio"] = (df["return_5d"] / vol_safe).clip(-5, 5)
        else:
            df["return_vol_ratio"] = 0.0

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
        daily_sent = (
            sentiment_df.groupby("date")
            .agg(
                sentiment_score=("sentiment_score", "mean"),
                post_volume=("sentiment_score", "count"),
                bullish_ratio=(
                    "sentiment_label",
                    lambda x: (x == "bullish").mean(),
                ),
            )
            .reset_index()
        )

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
        for col in [
            "sentiment_score",
            "sentiment_ma5",
            "sentiment_change",
            "post_volume",
            "bullish_ratio",
        ]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 籌碼欄位前填
        for col in [
            "foreign_buy_sell",
            "trust_buy_sell",
            "dealer_buy_sell",
            "margin_balance",
            "short_balance",
        ]:
            if col in df.columns:
                df[col] = df[col].ffill(limit=5).fillna(0)

        # 新增特徵填 0
        for col in [
            "realized_vol_5d",
            "realized_vol_20d",
            "parkinson_vol",
            "volume_ratio_5d",
            "spread_proxy",
            "day_of_week",
            "month",
            "is_settlement",
            # Tier 3 relative features
            "close_to_sma5",
            "close_to_sma20",
            "close_to_sma60",
            "bb_position",
            "volume_to_ma20",
            "margin_change_5d",
            "short_change_5d",
            "institutional_net_ratio",
            "momentum_12_1",
            "return_vol_ratio",
        ]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # OBV normalization: scale to unit variance to prevent dominating tree splits
        if "obv" in df.columns:
            obv_std = df["obv"].std()
            if obv_std > 0:
                df["obv"] = df["obv"] / obv_std

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
        # Exclude non-stationary absolute price features from selection
        # (they create spurious correlations across different price regimes)
        _NON_STATIONARY = {"close", "open", "high", "low", "sma_5", "sma_20", "sma_60",
                           "bb_upper", "bb_lower"}
        # Force-include return features — they are the strongest predictors
        # for direction in walk-forward validation (0.55+ direction accuracy)
        _FORCE_INCLUDE = {"return_1d", "return_5d", "return_20d"}
        available_cols = [
            c for c in FEATURE_COLUMNS
            if c in df.columns and c not in _NON_STATIONARY
        ]
        valid = df.dropna(subset=[target_col])

        if len(valid) < 50 or len(available_cols) <= max_features:
            logger.info(
                "特徵數 (%d) <= max (%d)，跳過篩選", len(available_cols), max_features
            )
            return available_cols

        X = valid[available_cols].values.astype(np.float32)
        y = valid[target_col].values.astype(np.float32)

        # 替換 NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if method == "shap":
            selected = self._select_by_shap(X, y, available_cols, max_features)
        else:
            selected = self._select_by_mi(X, y, available_cols, max_features)

        # Force-include critical features even if MI/SHAP didn't rank them high
        for force_col in _FORCE_INCLUDE:
            if force_col in available_cols and force_col not in selected:
                selected.append(force_col)

        # 持久化重要性分數
        if session_factory is not None and stock_id is not None:
            self._persist_importance(
                X,
                y,
                available_cols,
                method,
                session_factory,
                stock_id,
            )

        # 移除高度共線特徵 (0.85: catches close~sma, bb_upper~close, etc.)
        selected = self._remove_collinear(df, selected, threshold=0.85)

        logger.info(
            "特徵篩選: %d → %d (%s): %s",
            len(available_cols),
            len(selected),
            method,
            selected,
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
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
            )
            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[: min(500, len(X))])
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
        """Feature selection using f_regression (more robust than MI on small samples).

        MI needs ~500+ samples to be reliable; f_regression works with ~200+.
        Falls back to MI for larger datasets.
        """
        n_samples = len(y)
        if n_samples < 500:
            from sklearn.feature_selection import f_regression

            f_scores, p_values = f_regression(X, y)
            # Replace NaN scores (constant features) with 0
            f_scores = np.nan_to_num(f_scores, nan=0.0)
            ranked_idx = np.argsort(f_scores)[::-1][:max_features]
            logger.info(
                "Feature selection: f_regression (n=%d < 500), top p-values: %s",
                n_samples,
                [f"{p_values[i]:.4f}" for i in ranked_idx[:5]],
            )
        else:
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
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
            )
            model.fit(X, y)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[: min(500, len(X))])
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
        seq_len: int = 30,
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
            target_col = (
                TB_TARGET_COLUMN if TB_TARGET_COLUMN in df.columns else TARGET_COLUMN
            )

        # Sanitize features: replace inf/NaN before sequence construction
        feature_data = np.nan_to_num(
            df[feature_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
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
        weight_col: str | None = None,
    ) -> (
        tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray | None]
    ):
        """為 XGBoost 準備表格資料

        Args:
            weight_col: if provided, return aligned sample weights as 3rd element

        Returns:
            (X, y) or (X, y, weights) where X.shape = (samples, features), y.shape = (samples,)
        """
        if feature_cols is None:
            feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

        if target_col is None:
            target_col = (
                TB_TARGET_COLUMN if TB_TARGET_COLUMN in df.columns else TARGET_COLUMN
            )

        valid = df.dropna(subset=[target_col])
        X = valid[feature_cols].values.astype(np.float32)
        X = self._sanitize_array(X)
        y = valid[target_col].values.astype(np.float32)

        if weight_col is not None:
            w = valid[weight_col].values if weight_col in valid.columns else None
            # Re-normalize weights to mean=1.0 after dropna removed NaN-label rows
            # (which had weight=0). Without this, mean≈0.68 breaks XGBoost
            # min_child_weight thresholds.
            if w is not None and len(w) > 0:
                w_pos = w[w > 0]
                if len(w_pos) > 0:
                    w_mean = w_pos.mean()
                    if w_mean > 0 and abs(w_mean - 1.0) > 0.01:
                        w = w * (1.0 / w_mean)
            return X, y, w

        return X, y
