"""模型訓練 / 評估 Pipeline

支援：
- Triple Barrier 標籤 (tb_label)
- Purged Walk-Forward CV（purging + embargo）
- 樣本唯一性權重
- 特徵篩選
"""

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from src.analysis.features import (
    FeatureEngineer, FEATURE_COLUMNS, TARGET_COLUMN,
    TB_TARGET_COLUMN, TB_WEIGHT_COLUMN,
)
from src.models.lstm_model import LSTMPredictor
from src.models.xgboost_model import StockXGBoost
from src.models.ensemble import EnsemblePredictor, StackingEnsemble, PredictionResult
from src.utils.config import settings

# Lazy imports for optional models
def _import_tft():
    from src.models.tft_model import TFTPredictor
    return TFTPredictor

def _import_meta_labeler():
    from src.models.meta_label import MetaLabeler
    return MetaLabeler

logger = logging.getLogger(__name__)

MODEL_DIR = settings.PROJECT_ROOT / "models"


class PurgedTimeSeriesSplit:
    """Purged Walk-Forward Cross-Validation

    在 TimeSeriesSplit 基礎上加入：
    1. Purging: 從訓練集末尾移除與測試集有標籤重疊的樣本
    2. Embargo: 在 purge 區之後再移除額外的 embargo 天數

    這避免了因為標籤前視（如 Triple Barrier 的持有期）
    導致的訓練/測試資料洩漏。

    Args:
        n_splits: 折數
        purge_days: 從訓練集末尾移除的天數（應 >= max_holding）
        embargo_days: purge 之後額外的安全間隔天數
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 10,
        embargo_days: int = 5,
    ):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, X):
        """產生 purged + embargoed 的 train/test index 對

        Yields:
            (train_indices, test_indices) for each fold
        """
        n = len(X) if hasattr(X, '__len__') else X.shape[0]
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        for train_idx, test_idx in tscv.split(range(n)):
            # 測試集的起始位置
            test_start = test_idx[0]

            # Purge: 從訓練集末尾移除可能有標籤重疊的樣本
            purge_start = max(0, test_start - self.purge_days)

            # Embargo: 在 purge 之後再多移除幾天
            effective_train_end = max(0, purge_start - self.embargo_days)

            # 新的訓練集 index
            purged_train_idx = train_idx[train_idx < effective_train_end]

            if len(purged_train_idx) < 20:
                # 訓練集太小時跳過這折
                logger.warning(
                    "Fold skipped: train set too small after purging (%d samples)",
                    len(purged_train_idx),
                )
                continue

            logger.debug(
                "Purged CV: train[0:%d] (removed %d), test[%d:%d]",
                effective_train_end,
                len(train_idx) - len(purged_train_idx),
                test_idx[0], test_idx[-1],
            )

            yield purged_train_idx, test_idx


class ModelTrainer:
    """模型訓練管線"""

    def __init__(self, stock_id: str, session_factory=None):
        self.stock_id = stock_id
        self.session_factory = session_factory
        self.feature_eng = FeatureEngineer()
        self.lstm: LSTMPredictor | None = None
        self.xgb: StockXGBoost | None = None
        self.tft = None  # TFTPredictor (optional)
        self.stacking: StackingEnsemble | None = None
        self.meta_labeler = None  # MetaLabeler (optional)
        self.ensemble = EnsemblePredictor()
        self.feature_cols: list[str] = []

    def train(
        self,
        start_date: str,
        end_date: str,
        epochs: int = 50,
        seq_len: int = 60,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        use_triple_barrier: bool = True,
        max_features: int = 20,
        feature_selection_method: str = "mutual_info",
        use_tft: bool = False,
        use_meta_label: bool = False,
    ) -> dict:
        """完整訓練流程（3-way split: train/val/test）

        Args:
            start_date, end_date: 訓練資料範圍
            epochs: LSTM 訓練輪數
            seq_len: LSTM 序列長度
            val_ratio: 驗證集比例（用於 early stopping + 權重校準）
            test_ratio: 測試集比例（僅用於最終評估，不參與任何訓練決策）
            use_triple_barrier: 使用 Triple Barrier 標籤（預設 True）
            max_features: 特徵篩選最大數量
            feature_selection_method: 特徵篩選方法 ('mutual_info' 或 'shap')
            use_tft: 啟用 TFT 訓練 + StackingEnsemble
            use_meta_label: 啟用 Meta-Labeling 二階段

        Returns:
            {"lstm": {metrics}, "xgboost": {metrics}, ...}
        """
        logger.info("開始訓練 %s 模型...", self.stock_id)

        # 1. 建立特徵（含 Triple Barrier 標籤）
        df = self.feature_eng.build_features(self.stock_id, start_date, end_date)
        if df.empty or len(df) < seq_len + 20:
            raise ValueError(f"資料不足：只有 {len(df)} 筆，需要至少 {seq_len + 20} 筆")

        # 選擇 target
        target_col = TB_TARGET_COLUMN if use_triple_barrier else TARGET_COLUMN
        logger.info("使用 target: %s", target_col)

        # 2. 特徵篩選（含持久化重要性分數）
        self.feature_cols = self.feature_eng.select_features(
            df,
            target_col=target_col,
            max_features=max_features,
            method=feature_selection_method,
            session_factory=self.session_factory,
            stock_id=self.stock_id,
        )
        logger.info("使用 %d 個特徵, %d 筆資料", len(self.feature_cols), len(df))

        # 3. 3-way split with purging (train 60% / val 20% / test 20%)
        n = len(df)
        train_end = int(n * (1 - val_ratio - test_ratio))
        val_end = int(n * (1 - test_ratio))

        # Purge: 移除訓練集末尾與驗證集標籤重疊的樣本
        purge_gap = 15  # max_holding(10) + embargo(5)
        effective_train_end = max(0, train_end - purge_gap)
        effective_val_end = max(0, val_end - purge_gap)

        df_train = df.iloc[:effective_train_end]
        df_val = df.iloc[train_end:effective_val_end]
        df_test = df.iloc[val_end:]

        logger.info(
            "資料切分 (purged): train=%d, val=%d, test=%d (purge_gap=%d)",
            len(df_train), len(df_val), len(df_test), purge_gap,
        )

        # 取得樣本權重
        train_weights = None
        if use_triple_barrier and TB_WEIGHT_COLUMN in df_train.columns:
            valid_train = df_train.dropna(subset=[target_col])
            train_weights = valid_train[TB_WEIGHT_COLUMN].values

        results = {}

        # 4. LSTM 訓練
        X_train_seq, y_train_seq = self.feature_eng.prepare_sequences(
            df_train, seq_len, self.feature_cols, target_col=target_col,
        )
        X_val_seq, y_val_seq = self.feature_eng.prepare_sequences(
            df_val, seq_len, self.feature_cols, target_col=target_col,
        )
        X_test_seq, y_test_seq = self.feature_eng.prepare_sequences(
            df_test, seq_len, self.feature_cols, target_col=target_col,
        )

        if len(X_train_seq) > 0:
            self.lstm = LSTMPredictor(
                input_size=len(self.feature_cols),
                output_size=1,
            )
            lstm_history = self.lstm.train(
                X_train_seq, y_train_seq,
                X_val_seq if len(X_val_seq) > 0 else None,
                y_val_seq if len(y_val_seq) > 0 else None,
                epochs=epochs,
                patience=10,
            )
            results["lstm"] = {
                "final_train_loss": lstm_history["train_loss"][-1],
                "final_val_loss": lstm_history["val_loss"][-1] if lstm_history["val_loss"] else None,
            }

            # 測試集評估（僅報告，不用於任何決策）
            if len(X_test_seq) > 0:
                test_loss = self.lstm.evaluate(X_test_seq, y_test_seq)
                results["lstm"]["test_loss"] = test_loss

            # 儲存
            self.lstm.save(MODEL_DIR / f"{self.stock_id}_lstm.pt")
        else:
            logger.warning("LSTM 訓練資料不足")

        # 5. XGBoost 訓練（含樣本權重）
        X_train_tab, y_train_tab = self.feature_eng.prepare_tabular(
            df_train, self.feature_cols, target_col=target_col,
        )
        X_val_tab, y_val_tab = self.feature_eng.prepare_tabular(
            df_val, self.feature_cols, target_col=target_col,
        )
        X_test_tab, y_test_tab = self.feature_eng.prepare_tabular(
            df_test, self.feature_cols, target_col=target_col,
        )

        if len(X_train_tab) > 0:
            self.xgb = StockXGBoost()

            # 準備樣本權重（與 tabular data 對齊）
            xgb_sample_weight = None
            if train_weights is not None and len(train_weights) == len(X_train_tab):
                xgb_sample_weight = train_weights

            xgb_results = self.xgb.train(
                X_train_tab, y_train_tab,
                X_val_tab if len(X_val_tab) > 0 else None,
                y_val_tab if len(y_val_tab) > 0 else None,
                feature_names=self.feature_cols,
                sample_weight=xgb_sample_weight,
            )
            results["xgboost"] = xgb_results

            # 儲存
            self.xgb.save(MODEL_DIR / f"{self.stock_id}_xgb.json")
        else:
            logger.warning("XGBoost 訓練資料不足")

        # 6. 用 validation set（非 test set）校準集成權重
        if self.lstm and self.xgb and len(X_val_seq) > 0 and len(X_val_tab) > 0:
            lstm_val_errors = self.lstm.predict(X_val_seq).flatten() - y_val_seq
            xgb_val_errors = self.xgb.predict(X_val_tab) - y_val_tab
            self.ensemble.update_weights(lstm_val_errors, xgb_val_errors)

            results["ensemble"] = {
                "lstm_weight": self.ensemble.lstm_weight,
                "xgb_weight": self.ensemble.xgb_weight,
            }

        # 7. 訓練 HMM 市場狀態偵測器
        if "return_1d" in df.columns:
            returns = df["return_1d"].dropna().values
            volatility = df["realized_vol_20d"] if "realized_vol_20d" in df.columns else None
            vol_values = volatility.dropna().values if volatility is not None else None
            self.ensemble.fit_hmm(returns, vol_values)
            results["hmm"] = {"fitted": self.ensemble.hmm is not None and self.ensemble.hmm.is_fitted}

        # 8. TFT 訓練（gated by use_tft）
        if use_tft:
            try:
                TFTPredictor = _import_tft()
                self.tft = TFTPredictor()
                tft_result = self.tft.train(
                    df_train, self.feature_cols,
                    target_col=target_col,
                    stock_id=self.stock_id,
                    max_epochs=epochs,
                )
                results["tft"] = tft_result
                if "error" not in tft_result:
                    self.tft.save(MODEL_DIR / f"{self.stock_id}_tft.ckpt")
            except Exception as e:
                logger.warning("TFT 訓練失敗: %s", e)
                self.tft = None

        # 9. StackingEnsemble（用 Ridge meta-learner 取代固定權重）
        if use_tft and self.tft is not None and self.lstm and self.xgb:
            try:
                val_preds = {}
                # LSTM val predictions
                if len(X_val_seq) > 0:
                    val_preds["lstm"] = self.lstm.predict(X_val_seq).flatten()
                # XGBoost val predictions
                if len(X_val_tab) > 0:
                    val_preds["xgboost"] = self.xgb.predict(X_val_tab)
                # TFT val predictions (DataFrame interface)
                tft_val_pred = self.tft.predict(
                    df_val, self.feature_cols, target_col=target_col,
                    stock_id=self.stock_id,
                )
                if len(tft_val_pred) > 0:
                    val_preds["tft"] = tft_val_pred

                # 對齊長度
                if len(val_preds) >= 2:
                    min_len = min(len(v) for v in val_preds.values())
                    val_preds = {k: v[:min_len] for k, v in val_preds.items()}
                    y_val_aligned = y_val_tab[:min_len] if len(y_val_tab) >= min_len else y_val_seq[:min_len]

                    self.stacking = StackingEnsemble(alpha=1.0)
                    self.stacking.fit(val_preds, y_val_aligned)
                    self.stacking.save(MODEL_DIR / f"{self.stock_id}_stacking.pkl")
                    results["stacking"] = {"fitted": True, "models": list(val_preds.keys())}
            except Exception as e:
                logger.warning("StackingEnsemble 訓練失敗: %s", e)

        # 10. Meta-Labeling 二階段（gated by use_meta_label）
        if use_meta_label and self.xgb is not None:
            try:
                MetaLabeler = _import_meta_labeler()
                self.meta_labeler = MetaLabeler()

                # 使用驗證集建構 meta features
                if len(X_val_tab) > 0:
                    xgb_val_pred = self.xgb.predict(X_val_tab)
                    signal_strength = np.abs(xgb_val_pred)
                    signal_strength = signal_strength / (signal_strength.max() + 1e-8)

                    meta_features = self.meta_labeler.prepare_meta_features(
                        primary_pred=xgb_val_pred,
                        signal_strength=signal_strength,
                    )
                    meta_labels = self.meta_labeler.prepare_meta_labels(
                        primary_pred=xgb_val_pred,
                        actual_returns=y_val_tab,
                    )

                    self.meta_labeler.fit(meta_features, meta_labels)
                    self.meta_labeler.save(MODEL_DIR / f"{self.stock_id}_meta.pkl")
                    results["meta_labeler"] = {"fitted": self.meta_labeler.is_fitted}
            except Exception as e:
                logger.warning("Meta-Labeler 訓練失敗: %s", e)

        results["config"] = {
            "target": target_col,
            "n_features": len(self.feature_cols),
            "features": self.feature_cols,
            "use_triple_barrier": use_triple_barrier,
            "use_tft": use_tft,
            "use_meta_label": use_meta_label,
            "purge_gap": purge_gap,
        }

        logger.info("訓練完成: %s", results)
        return results

    def cpcv_validate(
        self,
        start_date: str,
        end_date: str,
        n_blocks: int = 6,
        k_test: int = 2,
        seq_len: int = 60,
        use_triple_barrier: bool = True,
    ) -> dict:
        """CPCV 過擬合檢測

        Args:
            n_blocks: 資料分塊數
            k_test: 每次測試用幾個 block

        Returns:
            CPCV + PBO 分析結果
        """
        from src.models.cpcv import CPCVAnalyzer

        df = self.feature_eng.build_features(self.stock_id, start_date, end_date)
        if df.empty:
            raise ValueError("無法建立特徵")

        target_col = TB_TARGET_COLUMN if use_triple_barrier else TARGET_COLUMN
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

        def train_and_evaluate(train_idx, test_idx) -> float:
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]

            X_train_tab, y_train_tab = self.feature_eng.prepare_tabular(
                df_train, feature_cols, target_col=target_col,
            )
            X_test_tab, y_test_tab = self.feature_eng.prepare_tabular(
                df_test, feature_cols, target_col=target_col,
            )

            if len(X_train_tab) < 10 or len(X_test_tab) < 5:
                return 0.0

            xgb = StockXGBoost()
            xgb.train(X_train_tab, y_train_tab, feature_names=feature_cols)
            pred = xgb.predict(X_test_tab)

            # 方向準確率作為績效指標
            direction_acc = float(np.mean(np.sign(pred) == np.sign(y_test_tab)))
            return direction_acc

        analyzer = CPCVAnalyzer(n_blocks=n_blocks, k_test=k_test, purge_days=10)
        return analyzer.run_cpcv_analysis(len(df), train_and_evaluate)

    def walk_forward_validate(
        self,
        start_date: str,
        end_date: str,
        n_splits: int = 5,
        seq_len: int = 60,
        epochs: int = 30,
        use_triple_barrier: bool = True,
        purge_days: int = 10,
        embargo_days: int = 5,
    ) -> list[dict]:
        """Purged Walk-Forward 交叉驗證

        在 TimeSeriesSplit 基礎上加入 purging + embargo，
        避免因標籤前視導致的訓練/測試資料洩漏。

        Args:
            n_splits: 折數
            epochs: 每折 LSTM 訓練輪數（較少以加速）
            use_triple_barrier: 使用 Triple Barrier 標籤
            purge_days: Purge 天數（>= max_holding）
            embargo_days: Embargo 天數

        Returns:
            list of per-fold metrics dicts
        """
        logger.info(
            "開始 Purged Walk-Forward 驗證 (%d folds, purge=%d, embargo=%d)...",
            n_splits, purge_days, embargo_days,
        )

        df = self.feature_eng.build_features(self.stock_id, start_date, end_date)
        if df.empty:
            raise ValueError("無法建立特徵")

        target_col = TB_TARGET_COLUMN if use_triple_barrier else TARGET_COLUMN
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

        # 使用 Purged CV
        ptscv = PurgedTimeSeriesSplit(
            n_splits=n_splits,
            purge_days=purge_days,
            embargo_days=embargo_days,
        )
        fold_results = []
        fold_idx = 0

        for train_idx, test_idx in ptscv.split(df):
            fold_idx += 1
            logger.info("=== Fold %d/%d ===", fold_idx, n_splits)
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]

            fold_metric = {
                "fold": fold_idx,
                "train_size": len(df_train),
                "test_size": len(df_test),
                "purged": True,
            }

            # LSTM
            X_train_seq, y_train_seq = self.feature_eng.prepare_sequences(
                df_train, seq_len, feature_cols, target_col=target_col,
            )
            X_test_seq, y_test_seq = self.feature_eng.prepare_sequences(
                df_test, seq_len, feature_cols, target_col=target_col,
            )

            if len(X_train_seq) > 0 and len(X_test_seq) > 0:
                lstm = LSTMPredictor(input_size=len(feature_cols), output_size=1)
                lstm.train(X_train_seq, y_train_seq, epochs=epochs)

                lstm_pred = lstm.predict(X_test_seq).flatten()
                lstm_mse = float(np.mean((lstm_pred - y_test_seq) ** 2))
                lstm_direction_acc = float(
                    np.mean(np.sign(lstm_pred) == np.sign(y_test_seq))
                )
                fold_metric["lstm_mse"] = lstm_mse
                fold_metric["lstm_direction_acc"] = lstm_direction_acc

            # XGBoost（含樣本權重）
            X_train_tab, y_train_tab = self.feature_eng.prepare_tabular(
                df_train, feature_cols, target_col=target_col,
            )
            X_test_tab, y_test_tab = self.feature_eng.prepare_tabular(
                df_test, feature_cols, target_col=target_col,
            )

            if len(X_train_tab) > 0 and len(X_test_tab) > 0:
                xgb = StockXGBoost()

                # 樣本權重
                sample_weight = None
                if use_triple_barrier and TB_WEIGHT_COLUMN in df_train.columns:
                    valid_train = df_train.dropna(subset=[target_col])
                    if len(valid_train) == len(X_train_tab):
                        sample_weight = valid_train[TB_WEIGHT_COLUMN].values

                xgb.train(
                    X_train_tab, y_train_tab,
                    feature_names=feature_cols,
                    sample_weight=sample_weight,
                )

                xgb_pred = xgb.predict(X_test_tab)
                xgb_mse = float(np.mean((xgb_pred - y_test_tab) ** 2))
                xgb_direction_acc = float(
                    np.mean(np.sign(xgb_pred) == np.sign(y_test_tab))
                )
                fold_metric["xgb_mse"] = xgb_mse
                fold_metric["xgb_direction_acc"] = xgb_direction_acc

            fold_results.append(fold_metric)
            logger.info("Fold %d results: %s", fold_idx, fold_metric)

        # Summary
        if fold_results:
            avg_lstm_mse = np.mean([f["lstm_mse"] for f in fold_results if "lstm_mse" in f])
            avg_xgb_mse = np.mean([f["xgb_mse"] for f in fold_results if "xgb_mse" in f])
            logger.info(
                "Purged Walk-Forward 完成: avg LSTM MSE=%.6f, avg XGB MSE=%.6f",
                avg_lstm_mse, avg_xgb_mse,
            )
        else:
            logger.warning("所有 fold 都因資料不足被跳過")

        return fold_results

    def predict(
        self,
        start_date: str,
        end_date: str,
        seq_len: int = 60,
        recent_returns_std: float | None = None,
    ) -> PredictionResult | None:
        """使用訓練好的模型進行預測

        Args:
            start_date, end_date: 特徵計算的資料範圍（應包含最近的交易日）
            recent_returns_std: 近 20 日報酬率標準差（用於波動率校準閾值）
        """
        # 建立特徵
        df = self.feature_eng.build_features(self.stock_id, start_date, end_date)
        if df.empty:
            logger.error("無法建立特徵")
            return None

        feature_cols = self.feature_cols or [c for c in FEATURE_COLUMNS if c in df.columns]
        current_price = df["close"].iloc[-1]

        # 計算近 20 日報酬率標準差（若未提供）
        if recent_returns_std is None and "return_1d" in df.columns:
            recent_returns_std = df["return_1d"].iloc[-20:].std()

        # 取得近期報酬率和波動率（用於 HMM 狀態偵測）
        recent_returns = None
        recent_volatility = None
        if "return_1d" in df.columns:
            recent_returns = df["return_1d"].iloc[-60:].values
        if "realized_vol_20d" in df.columns:
            recent_volatility = df["realized_vol_20d"].iloc[-60:].values

        # LSTM 預測
        lstm_pred = np.array([0.0])
        if self.lstm:
            X_seq, _ = self.feature_eng.prepare_sequences(df, seq_len, feature_cols)
            if len(X_seq) > 0:
                lstm_pred = self.lstm.predict(X_seq[-1:]).flatten()

        # XGBoost 預測
        xgb_pred = np.array([0.0])
        if self.xgb:
            X_tab, _ = self.feature_eng.prepare_tabular(df, feature_cols)
            if len(X_tab) > 0:
                xgb_pred = np.atleast_1d(self.xgb.predict(X_tab[-1:]))

        # 嘗試 StackingEnsemble（優先），fallback 加權平均
        if self.stacking is not None and self.stacking.is_fitted:
            model_preds = {"lstm": lstm_pred, "xgboost": xgb_pred}
            # TFT 預測（若可用）
            if self.tft is not None:
                tft_pred = self.tft.predict(df, self.feature_cols, stock_id=self.stock_id)
                if len(tft_pred) > 0:
                    model_preds["tft"] = tft_pred[:len(lstm_pred)]
            # 只保留 stacking 認識的模型
            valid_preds = {k: v for k, v in model_preds.items() if k in self.stacking.model_names}
            if len(valid_preds) == len(self.stacking.model_names):
                market_state = None
                if recent_returns is not None and self.ensemble.hmm is not None:
                    market_state = self.ensemble.detect_market_state(recent_returns, recent_volatility)
                result = self.stacking.predict_with_signal(
                    valid_preds, current_price,
                    recent_returns_std=recent_returns_std,
                    market_state=market_state,
                )
                return result

        # Fallback: 加權平均集成（含 HMM 狀態偵測）
        result = self.ensemble.predict(
            lstm_pred, xgb_pred, current_price,
            recent_returns_std=recent_returns_std,
            recent_returns=recent_returns,
            recent_volatility=recent_volatility,
        )

        return result

    def load_models(self):
        """載入已訓練的模型"""
        lstm_path = MODEL_DIR / f"{self.stock_id}_lstm.pt"
        xgb_path = MODEL_DIR / f"{self.stock_id}_xgb.json"
        tft_path = MODEL_DIR / f"{self.stock_id}_tft.ckpt"
        stacking_path = MODEL_DIR / f"{self.stock_id}_stacking.pkl"
        meta_path = MODEL_DIR / f"{self.stock_id}_meta.pkl"

        if lstm_path.exists():
            # 需要知道 input_size，從 meta 或預設值推斷
            import json
            meta_json_path = xgb_path.with_suffix(".meta.json")
            n_features = len(FEATURE_COLUMNS)
            if meta_json_path.exists():
                with open(meta_json_path) as f:
                    meta_data = json.load(f)
                    n_features = len(meta_data.get("feature_names", FEATURE_COLUMNS))
                    self.feature_cols = meta_data["feature_names"]

            self.lstm = LSTMPredictor(input_size=n_features, output_size=1)
            self.lstm.load(lstm_path)
            logger.info("已載入 LSTM 模型")

        if xgb_path.exists():
            self.xgb = StockXGBoost()
            self.xgb.load(xgb_path)
            self.feature_cols = self.xgb.feature_names
            logger.info("已載入 XGBoost 模型")

        # TFT（可選）
        if tft_path.exists():
            try:
                TFTPredictor = _import_tft()
                self.tft = TFTPredictor()
                self.tft.load(tft_path)
                logger.info("已載入 TFT 模型")
            except Exception as e:
                logger.debug("TFT 載入失敗: %s", e)

        # StackingEnsemble（可選）
        if stacking_path.exists():
            try:
                self.stacking = StackingEnsemble()
                self.stacking.load(stacking_path)
                logger.info("已載入 StackingEnsemble")
            except Exception as e:
                logger.debug("StackingEnsemble 載入失敗: %s", e)

        # Meta-Labeler（可選）
        if meta_path.exists():
            try:
                MetaLabeler = _import_meta_labeler()
                self.meta_labeler = MetaLabeler()
                self.meta_labeler.load(meta_path)
                logger.info("已載入 Meta-Labeler")
            except Exception as e:
                logger.debug("Meta-Labeler 載入失敗: %s", e)
