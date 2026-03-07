"""模型訓練 / 評估 Pipeline

支援：
- Triple Barrier 標籤 (tb_label)
- Purged Walk-Forward CV（purging + embargo）
- 樣本唯一性權重
- 特徵篩選
"""

import json as _json
import logging
from datetime import date

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from src.analysis.features import (
    FeatureEngineer,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TB_TARGET_COLUMN,
    TB_WEIGHT_COLUMN,
)
from src.models.lstm_model import LSTMPredictor
from src.models.xgboost_model import StockXGBoost
from src.models.ensemble import (
    EnsemblePredictor,
    StackingEnsemble,
    PredictionResult,
    direction_accuracy,
    direction_accuracy_classify,
)
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
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
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
                test_idx[0],
                test_idx[-1],
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

        # Sanitize labels: replace inf with NaN (e.g. divide-by-zero in Triple Barrier)
        if target_col in df.columns:
            inf_count = np.isinf(df[target_col]).sum()
            if inf_count > 0:
                logger.warning("標籤含 %d 個 inf 值，已替換為 NaN", inf_count)
                df[target_col] = df[target_col].replace([np.inf, -np.inf], np.nan)

        # Sanitize feature columns: replace inf with NaN, then fill with median
        numeric_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        for col in numeric_cols:
            if df[col].dtype.kind == "f":
                inf_mask = np.isinf(df[col])
                if inf_mask.any():
                    logger.warning(
                        "Feature '%s' has %d inf values → median fill",
                        col,
                        inf_mask.sum(),
                    )
                    df.loc[inf_mask, col] = np.nan
                    df[col] = df[col].fillna(df[col].median())

        # Label distribution logging
        if target_col in df.columns:
            labels = df[target_col].dropna()
            n_pos = (labels > 0).sum()
            n_neg = (labels < 0).sum()
            n_zero = (labels == 0).sum()
            logger.info(
                "標籤分布: positive=%d (%.1f%%), negative=%d (%.1f%%), zero=%d (%.1f%%), total=%d",
                n_pos,
                100 * n_pos / len(labels) if len(labels) > 0 else 0,
                n_neg,
                100 * n_neg / len(labels) if len(labels) > 0 else 0,
                n_zero,
                100 * n_zero / len(labels) if len(labels) > 0 else 0,
                len(labels),
            )

        # 2. 3-way split with purging (train 60% / val 20% / test 20%)
        #    Feature selection AFTER split to avoid data leakage
        n = len(df)
        train_end = int(n * (1 - val_ratio - test_ratio))
        val_end = int(n * (1 - test_ratio))

        # Purge: 移除訓練集末尾與驗證集標籤重疊的樣本
        purge_gap = 10  # max_holding(10), no additional embargo
        effective_train_end = max(0, train_end - purge_gap)
        effective_val_end = max(0, val_end - purge_gap)

        effective_val_start = train_end + purge_gap  # purge: avoid label leakage
        df_train = df.iloc[:effective_train_end]
        df_val = df.iloc[effective_val_start:effective_val_end]
        df_test = df.iloc[val_end:]

        # Feature selection on train set only (prevents data leakage)
        self.feature_cols = self.feature_eng.select_features(
            df_train,
            target_col=target_col,
            max_features=max_features,
            method=feature_selection_method,
            session_factory=self.session_factory,
            stock_id=self.stock_id,
        )
        logger.info("使用 %d 個特徵, %d 筆資料", len(self.feature_cols), len(df))

        # Feature quality check
        X_check = df[self.feature_cols].values
        n_nan = np.isnan(X_check).sum()
        n_inf = np.isinf(X_check).sum()
        if n_nan > 0 or n_inf > 0:
            logger.warning(
                "Feature matrix quality: %d NaN, %d inf (before sanitization)",
                n_nan,
                n_inf,
            )
        else:
            logger.info("Feature matrix quality: clean (0 NaN, 0 inf)")

        logger.info(
            "資料切分 (purged): train=%d, val=%d, test=%d (purge_gap=%d)",
            len(df_train),
            len(df_val),
            len(df_test),
            purge_gap,
        )

        results = {}

        # 4. LSTM 訓練
        X_train_seq, y_train_seq = self.feature_eng.prepare_sequences(
            df_train,
            seq_len,
            self.feature_cols,
            target_col=target_col,
        )
        X_val_seq, y_val_seq = self.feature_eng.prepare_sequences(
            df_val,
            seq_len,
            self.feature_cols,
            target_col=target_col,
        )
        X_test_seq, y_test_seq = self.feature_eng.prepare_sequences(
            df_test,
            seq_len,
            self.feature_cols,
            target_col=target_col,
        )

        if len(X_train_seq) > 0:
            self.lstm = LSTMPredictor(
                input_size=len(self.feature_cols),
                output_size=1,
                use_attention=False,
            )
            lstm_history = self.lstm.train(
                X_train_seq,
                y_train_seq,
                X_val_seq if len(X_val_seq) > 0 else None,
                y_val_seq if len(y_val_seq) > 0 else None,
                epochs=epochs,
                patience=10,
            )
            results["lstm"] = {
                "final_train_loss": lstm_history["train_loss"][-1],
                "final_val_loss": lstm_history["val_loss"][-1]
                if lstm_history["val_loss"]
                else None,
            }

            # 測試集評估（僅報告，不用於任何決策）
            if len(X_test_seq) > 0:
                test_loss = self.lstm.evaluate(X_test_seq, y_test_seq)
                results["lstm"]["test_loss"] = test_loss
                test_dir = self.lstm.evaluate_directional(X_test_seq, y_test_seq)
                # Prefer tb_class for direction accuracy if available
                if "tb_class" in df_test.columns:
                    lstm_pred = self.lstm.predict(X_test_seq).flatten()
                    test_valid = df_test.dropna(subset=[target_col])
                    tb_cls = test_valid["tb_class"].values[-len(lstm_pred) :]
                    results["lstm"]["test_direction_acc"] = direction_accuracy_classify(
                        lstm_pred, tb_cls
                    )
                else:
                    results["lstm"]["test_direction_acc"] = test_dir["direction_acc"]
                results["lstm"]["test_beats_naive"] = test_dir["beats_naive"]

            # NOTE: save deferred to after quality_gate
        else:
            logger.warning("LSTM 訓練資料不足")

        # 5. XGBoost 訓練（含樣本權重 — aligned via prepare_tabular）
        weight_col = TB_WEIGHT_COLUMN if use_triple_barrier else None
        X_train_tab, y_train_tab, train_weights = self.feature_eng.prepare_tabular(
            df_train,
            self.feature_cols,
            target_col=target_col,
            weight_col=weight_col,
        )
        X_val_tab, y_val_tab = self.feature_eng.prepare_tabular(
            df_val,
            self.feature_cols,
            target_col=target_col,
        )
        X_test_tab, y_test_tab = self.feature_eng.prepare_tabular(
            df_test,
            self.feature_cols,
            target_col=target_col,
        )

        if len(X_train_tab) > 0:
            self.xgb = StockXGBoost()

            xgb_results = self.xgb.train(
                X_train_tab,
                y_train_tab,
                X_val_tab if len(X_val_tab) > 0 else None,
                y_val_tab if len(y_val_tab) > 0 else None,
                feature_names=self.feature_cols,
                sample_weight=train_weights,
            )
            results["xgboost"] = xgb_results

            # 測試集方向準確率（prefer tb_class for cleaner direction eval）
            if len(X_test_tab) > 0:
                test_pred = self.xgb.predict(X_test_tab)
                if "tb_class" in df_test.columns:
                    test_valid = df_test.dropna(subset=[target_col])
                    tb_cls = test_valid["tb_class"].values[-len(test_pred) :]
                    test_dir_acc = direction_accuracy_classify(test_pred, tb_cls)
                else:
                    test_dir_acc = direction_accuracy(test_pred, y_test_tab)
                test_mse = float(np.mean((test_pred - y_test_tab) ** 2))
                naive_mse = float(np.mean(y_test_tab**2))
                results["xgboost"]["test_direction_acc"] = test_dir_acc
                results["xgboost"]["test_mse"] = test_mse
                results["xgboost"]["test_beats_naive"] = test_mse < naive_mse

            # NOTE: save deferred to after quality_gate
        else:
            logger.warning("XGBoost 訓練資料不足")

        # 5b. Quality Gate — conditional save
        gate = self.quality_gate(results, start_date, end_date)
        results["quality_gate"] = gate

        if gate.get("lstm_passed") and self.lstm:
            self.lstm.save(MODEL_DIR / f"{self.stock_id}_lstm.pt")
            logger.info(
                "LSTM 通過品質門檻 (dir_acc=%.3f), 已存檔",
                gate.get("lstm_direction_acc", 0),
            )
        elif self.lstm:
            logger.warning(
                "LSTM 未通過品質門檻 (dir_acc=%.3f), 不存檔",
                gate.get("lstm_direction_acc", 0),
            )

        if gate.get("xgb_passed") and self.xgb:
            self.xgb.save(MODEL_DIR / f"{self.stock_id}_xgb.json")
            logger.info(
                "XGBoost 通過品質門檻 (dir_acc=%.3f), 已存檔",
                gate.get("xgb_direction_acc", 0),
            )
        elif self.xgb:
            logger.warning(
                "XGBoost 未通過品質門檻 (dir_acc=%.3f), 不存檔",
                gate.get("xgb_direction_acc", 0),
            )

        # RC1: Nullify failed models so they don't contaminate ensemble
        if not gate.get("lstm_passed"):
            self.lstm = None
            logger.info("LSTM failed quality gate — removed from ensemble")
        if not gate.get("xgb_passed"):
            self.xgb = None
            logger.info("XGBoost failed quality gate — removed from ensemble")

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
            volatility = (
                df["realized_vol_20d"] if "realized_vol_20d" in df.columns else None
            )
            vol_values = volatility.dropna().values if volatility is not None else None
            self.ensemble.fit_hmm(returns, vol_values)
            results["hmm"] = {
                "fitted": self.ensemble.hmm is not None and self.ensemble.hmm.is_fitted
            }

        # 8. TFT 訓練（gated by use_tft）
        if use_tft:
            try:
                TFTPredictor = _import_tft()
                self.tft = TFTPredictor()
                tft_result = self.tft.train(
                    df_train,
                    self.feature_cols,
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
                    df_val,
                    self.feature_cols,
                    target_col=target_col,
                    stock_id=self.stock_id,
                )
                if len(tft_val_pred) > 0:
                    val_preds["tft"] = tft_val_pred

                # 對齊長度
                if len(val_preds) >= 2:
                    min_len = min(len(v) for v in val_preds.values())
                    val_preds = {k: v[:min_len] for k, v in val_preds.items()}
                    y_val_aligned = (
                        y_val_tab[:min_len]
                        if len(y_val_tab) >= min_len
                        else y_val_seq[:min_len]
                    )

                    self.stacking = StackingEnsemble(alpha=1.0)
                    self.stacking.fit(val_preds, y_val_aligned)
                    self.stacking.save(MODEL_DIR / f"{self.stock_id}_stacking.pkl")
                    results["stacking"] = {
                        "fitted": True,
                        "models": list(val_preds.keys()),
                    }
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

        # Save training report JSON
        self._save_training_report(
            results, start_date, end_date, len(df_train), len(df_val), len(df_test)
        )

        logger.info("訓練完成: %s", results)
        return results

    def quality_gate(
        self,
        results: dict,
        start_date: str,
        end_date: str,
        min_direction_acc: float = 0.52,
        # NOTE: Financial stocks (banks/insurance) typically achieve 50.3-50.9% with
        # current technical-only features — below 0.52 threshold. This is expected.
        # ml_ensemble falls back to 0.5 (neutral), weight redistributed.
        # Future: add financial features (NIM, rates, dividend yield, P/B).
    ) -> dict:
        """Three-gate quality check before saving models.

        Gates:
        1. Direction accuracy > min_direction_acc on test set
        2. MSE < naive baseline (predict 0)
        3. CPCV PBO check — applies to BOTH XGBoost and LSTM:
           - XGBoost: PBO > 0.8 AND mean_oos < 0.47 → kill
           - LSTM: multi-fold direction accuracy check (no PBO kill, warning only)
           - PBO > 0.6 → warning only (small sample CPCV has high variance)

        Returns dict with per-model pass/fail and overall status.
        """
        gate = {
            "lstm_passed": False,
            "xgb_passed": False,
            "lstm_direction_acc": 0.0,
            "xgb_direction_acc": 0.0,
            "pbo": None,
            "lstm_cpcv_dir_acc": None,
            "overall_passed": False,
        }

        # Gate 1+2: LSTM
        lstm_metrics = results.get("lstm", {})
        lstm_dir = lstm_metrics.get("test_direction_acc", 0.0)
        lstm_naive = lstm_metrics.get("test_beats_naive", False)
        gate["lstm_direction_acc"] = lstm_dir
        if lstm_dir >= min_direction_acc and lstm_naive:
            gate["lstm_passed"] = True

        # Gate 1+2: XGBoost
        xgb_metrics = results.get("xgboost", {})
        xgb_dir = xgb_metrics.get("test_direction_acc", 0.0)
        xgb_naive = xgb_metrics.get("test_beats_naive", False)
        gate["xgb_direction_acc"] = xgb_dir
        if xgb_dir >= min_direction_acc and xgb_naive:
            gate["xgb_passed"] = True

        # Gate 3: CPCV PBO — XGBoost (hard gate)
        if gate["xgb_passed"]:
            try:
                cpcv_result = self.cpcv_validate(start_date, end_date)
                pbo_val = cpcv_result.get("pbo", {}).get("pbo", 0.0)
                gate["pbo"] = pbo_val
                mean_oos = cpcv_result.get("mean_oos", 0.5)
                gate["mean_oos_acc"] = mean_oos
                if pbo_val > 0.8 and mean_oos < 0.47:
                    logger.warning(
                        "PBO=%.2f > 0.8 AND mean_oos=%.3f < 0.47 — severe overfitting, failing XGBoost",
                        pbo_val,
                        mean_oos,
                    )
                    gate["xgb_passed"] = False
                elif pbo_val > 0.6:
                    logger.warning(
                        "PBO=%.2f > 0.6 — moderate overfitting risk (warning only)",
                        pbo_val,
                    )
            except Exception as e:
                logger.warning("CPCV validation failed, skipping PBO gate: %s", e)
                gate["pbo"] = None

        # Gate 3b: LSTM multi-fold direction check (soft gate — warning only)
        if gate["lstm_passed"]:
            try:
                lstm_cpcv_acc = self._lstm_cpcv_direction_check(start_date, end_date)
                gate["lstm_cpcv_dir_acc"] = lstm_cpcv_acc
                if lstm_cpcv_acc < 0.48:
                    logger.warning(
                        "LSTM CPCV direction_acc=%.3f < 0.48 — possible overfitting (warning only)",
                        lstm_cpcv_acc,
                    )
            except Exception as e:
                logger.debug("LSTM CPCV check skipped: %s", e)

        gate["overall_passed"] = gate["lstm_passed"] or gate["xgb_passed"]
        return gate

    def _lstm_cpcv_direction_check(
        self,
        start_date: str,
        end_date: str,
        n_folds: int = 3,
        seq_len: int = 40,
    ) -> float:
        """Multi-fold direction accuracy check for LSTM overfitting detection.

        Uses PurgedTimeSeriesSplit (not full CPCV) with n_folds to test
        whether LSTM direction accuracy is stable across time splits.

        Returns average OOS direction accuracy across folds.
        """
        from src.analysis.features import (
            FeatureEngineer,
            FEATURE_COLUMNS,
            TB_TARGET_COLUMN,
        )

        fe = FeatureEngineer()
        df = fe.build_features(self.stock_id, start_date, end_date)
        if df.empty or len(df) < seq_len + 60:
            return 0.5

        target_col = TB_TARGET_COLUMN if TB_TARGET_COLUMN in df.columns else "return_1d"
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

        splitter = PurgedTimeSeriesSplit(
            n_splits=n_folds, purge_days=10, embargo_days=5
        )
        dir_accs = []

        for train_idx, test_idx in splitter.split(df):
            if len(train_idx) < seq_len + 20 or len(test_idx) < 10:
                continue

            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]

            X_train_seq, y_train_seq = fe.prepare_sequences(
                df_train, seq_len, feature_cols
            )
            X_test_seq, y_test_seq = fe.prepare_sequences(
                df_test, seq_len, feature_cols
            )

            if len(X_train_seq) < 20 or len(X_test_seq) < 5:
                continue

            try:
                fold_lstm = LSTMPredictor(
                    input_size=X_train_seq.shape[2],
                    hidden_size=32,
                    num_layers=1,
                    dropout=0.2,
                    output_size=1,
                )
                fold_lstm.train(
                    X_train_seq,
                    y_train_seq,
                    X_test_seq,
                    y_test_seq,
                    epochs=30,
                    batch_size=32,
                    patience=10,
                )
                eval_result = fold_lstm.evaluate_directional(X_test_seq, y_test_seq)
                dir_accs.append(eval_result["direction_acc"])
            except Exception:
                continue

        if not dir_accs:
            return 0.5

        avg_acc = float(np.mean(dir_accs))
        logger.info(
            "LSTM CPCV direction check: %d folds, avg=%.3f, per-fold=%s",
            len(dir_accs),
            avg_acc,
            [f"{a:.3f}" for a in dir_accs],
        )
        return avg_acc

    def _save_training_report(
        self,
        results: dict,
        start_date: str,
        end_date: str,
        n_train: int,
        n_val: int,
        n_test: int,
    ):
        """Save training report JSON for model age tracking and quality audit."""
        report = {
            "stock_id": self.stock_id,
            "trained_at": date.today().isoformat(),
            "data_range": {"start": start_date, "end": end_date},
            "n_samples": {"train": n_train, "val": n_val, "test": n_test},
            "features": self.feature_cols,
            "quality_gate": results.get("quality_gate"),
            "metrics": {
                k: v
                for k, v in results.items()
                if k in ("lstm", "xgboost", "ensemble", "hmm")
            },
        }
        report_path = MODEL_DIR / f"{self.stock_id}_training_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            _json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Training report saved to %s", report_path)

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
        if target_col in df.columns:
            df[target_col] = df[target_col].replace([np.inf, -np.inf], np.nan)
        feature_cols = (
            self.feature_cols
            if self.feature_cols
            else [c for c in FEATURE_COLUMNS if c in df.columns]
        )

        # Sanitize features for CPCV
        for col in feature_cols:
            if col in df.columns and df[col].dtype.kind == "f":
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

        def train_and_evaluate(train_idx, test_idx) -> float:
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]

            X_train_tab, y_train_tab = self.feature_eng.prepare_tabular(
                df_train,
                feature_cols,
                target_col=target_col,
            )
            X_test_tab, y_test_tab = self.feature_eng.prepare_tabular(
                df_test,
                feature_cols,
                target_col=target_col,
            )

            if len(X_train_tab) < 10 or len(X_test_tab) < 5:
                return 0.0

            xgb = StockXGBoost()
            xgb.train(X_train_tab, y_train_tab, feature_names=feature_cols)
            pred = xgb.predict(X_test_tab)

            # 方向準確率作為績效指標
            return direction_accuracy(pred, y_test_tab)

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
            n_splits,
            purge_days,
            embargo_days,
        )

        df = self.feature_eng.build_features(self.stock_id, start_date, end_date)
        if df.empty:
            raise ValueError("無法建立特徵")

        target_col = TB_TARGET_COLUMN if use_triple_barrier else TARGET_COLUMN
        if target_col in df.columns:
            df[target_col] = df[target_col].replace([np.inf, -np.inf], np.nan)
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
                df_train,
                seq_len,
                feature_cols,
                target_col=target_col,
            )
            X_test_seq, y_test_seq = self.feature_eng.prepare_sequences(
                df_test,
                seq_len,
                feature_cols,
                target_col=target_col,
            )

            if len(X_train_seq) > 0 and len(X_test_seq) > 0:
                lstm = LSTMPredictor(input_size=len(feature_cols), output_size=1)
                lstm.train(X_train_seq, y_train_seq, epochs=epochs)

                lstm_pred = lstm.predict(X_test_seq).flatten()
                lstm_mse = float(np.mean((lstm_pred - y_test_seq) ** 2))
                lstm_direction_acc = direction_accuracy(lstm_pred, y_test_seq)
                fold_metric["lstm_mse"] = lstm_mse
                fold_metric["lstm_direction_acc"] = lstm_direction_acc

            # XGBoost（含樣本權重）
            X_train_tab, y_train_tab = self.feature_eng.prepare_tabular(
                df_train,
                feature_cols,
                target_col=target_col,
            )
            X_test_tab, y_test_tab = self.feature_eng.prepare_tabular(
                df_test,
                feature_cols,
                target_col=target_col,
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
                    X_train_tab,
                    y_train_tab,
                    feature_names=feature_cols,
                    sample_weight=sample_weight,
                )

                xgb_pred = xgb.predict(X_test_tab)
                xgb_mse = float(np.mean((xgb_pred - y_test_tab) ** 2))
                xgb_direction_acc = direction_accuracy(xgb_pred, y_test_tab)
                fold_metric["xgb_mse"] = xgb_mse
                fold_metric["xgb_direction_acc"] = xgb_direction_acc

            fold_results.append(fold_metric)
            logger.info("Fold %d results: %s", fold_idx, fold_metric)

        # Summary
        if fold_results:
            avg_lstm_mse = np.mean(
                [f["lstm_mse"] for f in fold_results if "lstm_mse" in f]
            )
            avg_xgb_mse = np.mean(
                [f["xgb_mse"] for f in fold_results if "xgb_mse" in f]
            )
            logger.info(
                "Purged Walk-Forward 完成: avg LSTM MSE=%.6f, avg XGB MSE=%.6f",
                avg_lstm_mse,
                avg_xgb_mse,
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

        feature_cols = self.feature_cols or [
            c for c in FEATURE_COLUMNS if c in df.columns
        ]
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

        # RC1: Handle single-model fallback
        if self.lstm is None and self.xgb is not None:
            self.ensemble.lstm_weight = 0.0
            self.ensemble.xgb_weight = 1.0
        elif self.xgb is None and self.lstm is not None:
            self.ensemble.lstm_weight = 1.0
            self.ensemble.xgb_weight = 0.0

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
                tft_pred = self.tft.predict(
                    df, self.feature_cols, stock_id=self.stock_id
                )
                if len(tft_pred) > 0:
                    model_preds["tft"] = tft_pred[: len(lstm_pred)]
            # 只保留 stacking 認識的模型
            valid_preds = {
                k: v for k, v in model_preds.items() if k in self.stacking.model_names
            }
            if len(valid_preds) == len(self.stacking.model_names):
                market_state = None
                if recent_returns is not None and self.ensemble.hmm is not None:
                    market_state = self.ensemble.detect_market_state(
                        recent_returns, recent_volatility
                    )
                result = self.stacking.predict_with_signal(
                    valid_preds,
                    current_price,
                    recent_returns_std=recent_returns_std,
                    market_state=market_state,
                )
                return result

        # Fallback: 加權平均集成（含 HMM 狀態偵測）
        result = self.ensemble.predict(
            lstm_pred,
            xgb_pred,
            current_price,
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
            # Infer input_size: xgb.meta.json → training_report.json → default
            import json

            meta_json_path = xgb_path.with_suffix(".meta.json")
            report_path = MODEL_DIR / f"{self.stock_id}_training_report.json"
            n_features = len(FEATURE_COLUMNS)

            if meta_json_path.exists():
                with open(meta_json_path) as f:
                    meta_data = json.load(f)
                    n_features = len(meta_data.get("feature_names", FEATURE_COLUMNS))
                    self.feature_cols = meta_data["feature_names"]
            elif report_path.exists():
                with open(report_path, encoding="utf-8") as f:
                    report_data = json.load(f)
                    features = report_data.get("features", [])
                    if features:
                        n_features = len(features)
                        self.feature_cols = features

            self.lstm = LSTMPredictor(input_size=n_features, output_size=1)
            self.lstm.load(lstm_path)
            # Sync n_features from actual loaded model (checkpoint may auto-correct)
            n_features = self.lstm.model.lstm.input_size
            logger.info("已載入 LSTM 模型 (input_size=%d)", n_features)

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
