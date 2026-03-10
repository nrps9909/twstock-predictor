"""集成模型 — 加權結合 LSTM + XGBoost 預測結果

Output:
- predicted_returns: 未來 N 日預測報酬率
- predicted_prices: 未來 N 日預測收盤價（幾何複利）
- confidence_interval: 95% 信心區間（log-space）
- signal: '買進' | '賣出' | '持有'
- signal_strength: 0.0 ~ 1.0

支援 HMM 市場狀態偵測 → 動態權重分配
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def direction_accuracy(
    pred: np.ndarray, target: np.ndarray, epsilon: float = 0.003
) -> float:
    """Direction accuracy excluding near-zero targets (|target| < epsilon).

    epsilon=0.003 filters out ±0.3% returns where direction is ambiguous
    (e.g. Triple Barrier time-expiry with tiny drift).
    """
    mask = np.abs(target) > epsilon
    if mask.sum() == 0:
        return 0.5
    return float(np.mean(np.sign(pred[mask]) == np.sign(target[mask])))


def direction_accuracy_classify(
    pred: np.ndarray, tb_class: np.ndarray, epsilon: float = 0.003
) -> float:
    """Direction accuracy using Triple Barrier class labels {1, 0, -1}.

    Excludes class=0 (time-expired with ambiguous direction) AND
    near-zero predictions (|pred| < epsilon) where LSTM outputs
    conservative values that np.sign maps unreliably.
    """
    mask = (tb_class != 0) & (np.abs(pred) > epsilon)
    if mask.sum() == 0:
        return 0.5
    pred_sign = np.sign(pred[mask])
    true_sign = np.sign(tb_class[mask])
    return float(np.mean(pred_sign == true_sign))


@dataclass
class MarketState:
    """HMM 市場狀態"""

    state: int  # 0=bull, 1=bear, 2=sideways
    state_name: str
    probabilities: np.ndarray  # [p_bull, p_bear, p_sideways]
    volatility: float
    mean_return: float


@dataclass
class PredictionResult:
    """預測結果"""

    predicted_returns: np.ndarray  # 預測報酬率
    predicted_prices: np.ndarray  # 預測收盤價
    confidence_lower: np.ndarray  # 95% CI 下界
    confidence_upper: np.ndarray  # 95% CI 上界
    signal: str  # "buy", "sell", "hold"
    signal_strength: float  # 0.0 ~ 1.0
    lstm_weight: float
    xgb_weight: float
    market_state: MarketState | None = None


@dataclass
class RegimeTransition:
    """行情轉場事件"""

    prev_state: str  # "bull", "bear", "sideways"
    curr_state: str
    severity: float  # 0.0 ~ 1.0
    action: str  # "reduce_50%", "close_all", "no_action", "increase_exposure"

    # 轉場矩陣：(prev, curr) → (severity, action)
    TRANSITION_MAP = {
        ("bull", "bear"): (1.0, "reduce_50%"),
        ("bull", "sideways"): (0.3, "no_action"),
        ("bear", "bull"): (0.5, "increase_exposure"),
        ("bear", "sideways"): (0.2, "no_action"),
        ("sideways", "bull"): (0.2, "no_action"),
        ("sideways", "bear"): (0.7, "reduce_50%"),
    }

    @classmethod
    def from_states(cls, prev: str, curr: str) -> "RegimeTransition":
        if prev == curr:
            return cls(prev, curr, 0.0, "no_action")
        severity, action = cls.TRANSITION_MAP.get((prev, curr), (0.3, "no_action"))
        return cls(prev, curr, severity, action)


class HMMStateDetector:
    """HMM 市場狀態偵測器

    使用 3-state Gaussian HMM 偵測市場狀態：
    - State 0: Bull（牛市 — 正報酬、低波動）
    - State 1: Bear（熊市 — 負報酬、高波動）
    - State 2: Sideways（盤整 — 低報酬、中波動）

    觀測值 = [daily_return, realized_volatility]
    """

    STATE_NAMES = {0: "bull", 1: "bear", 2: "sideways"}

    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.model = None
        self.is_fitted = False
        self._state_mapping: dict[int, str] = {}
        self._state_history: list[str] = []
        self._scale: float = 1.0  # 觀測值縮放因子（改善數值穩定性）

    def fit(
        self, returns: np.ndarray, volatility: np.ndarray | None = None
    ) -> "HMMStateDetector":
        """訓練 HMM

        Args:
            returns: 日報酬率序列
            volatility: 已實現波動率（可選，若無則用 rolling std 計算）
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning(
                "hmmlearn 未安裝，HMM 狀態偵測不可用。"
                "執行 pip install hmmlearn 來啟用。"
            )
            return self

        # 準備觀測值
        returns = np.asarray(returns).flatten()
        valid_mask = ~np.isnan(returns)

        if volatility is None:
            vol = pd.Series(returns).rolling(20).std().values
        else:
            vol = np.asarray(volatility).flatten()

        valid_mask &= ~np.isnan(vol)
        valid_mask &= np.isfinite(returns)
        valid_mask &= np.isfinite(vol)
        returns_clean = returns[valid_mask]
        vol_clean = vol[valid_mask]

        # Clip extreme values to prevent numerical instability
        returns_clean = np.clip(returns_clean, -0.2, 0.2)
        vol_clean = np.clip(vol_clean, 0, 0.15)

        if len(returns_clean) < 60:
            logger.warning("HMM 訓練資料不足 (%d < 60)，跳過", len(returns_clean))
            return self

        # 縮放觀測值以改善數值穩定性
        # （decimal returns ~1e-3 會導致 covariance 退化）
        self._scale = 100.0
        X = np.column_stack([returns_clean * self._scale, vol_clean * self._scale])

        # 嘗試 full covariance，失敗時退回 diag
        for cov_type in ("full", "diag"):
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=cov_type,
                n_iter=200,
                random_state=42,
                tol=0.01,
            )
            try:
                self.model.fit(X)
                self.model.predict(X[:5])  # 驗證 predict 不會因退化 covariance 爆掉
                self.is_fitted = True
                break
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.warning(
                    "HMM fit failed (covariance_type='%s'): %s", cov_type, exc
                )
                self.model = None
                self.is_fitted = False

        if not self.is_fitted:
            logger.warning("HMM 無法擬合，跳過狀態偵測")
            return self

        # 根據模型學到的 mean return 自動對應狀態名稱
        self._assign_state_names()

        logger.info(
            "HMM 訓練完成: states=%s, means=%s, cov_type=%s",
            self._state_mapping,
            {
                k: f"{self.model.means_[k][0] / self._scale:.4f}"
                for k in range(self.n_states)
            },
            self.model.covariance_type,
        )
        return self

    def _assign_state_names(self):
        """根據各狀態的平均報酬率自動分配名稱"""
        if not self.is_fitted:
            return

        mean_returns = self.model.means_[:, 0]  # 每個狀態的平均報酬率
        sorted_indices = np.argsort(mean_returns)

        # 最低報酬 = bear, 最高 = bull, 中間 = sideways
        self._state_mapping = {
            sorted_indices[0]: "bear",
            sorted_indices[1]: "sideways",
            sorted_indices[2]: "bull",
        }

    def predict_state(
        self, returns: np.ndarray, volatility: np.ndarray | None = None
    ) -> MarketState:
        """預測當前市場狀態

        Args:
            returns: 最近 N 天的日報酬率
            volatility: 最近 N 天的已實現波動率

        Returns:
            MarketState 物件
        """
        if not self.is_fitted:
            # 未訓練 → 返回預設狀態
            return MarketState(
                state=2,
                state_name="sideways",
                probabilities=np.array([0.33, 0.33, 0.34]),
                volatility=0.0,
                mean_return=0.0,
            )

        returns = np.asarray(returns).flatten()

        if volatility is None:
            vol = pd.Series(returns).rolling(20).std().values
        else:
            vol = np.asarray(volatility).flatten()

        # 使用最後一個有效的觀測值
        valid_mask = ~(np.isnan(returns) | np.isnan(vol))
        valid_mask &= np.isfinite(returns) & np.isfinite(vol)
        if not valid_mask.any():
            return MarketState(
                state=2,
                state_name="sideways",
                probabilities=np.array([0.33, 0.33, 0.34]),
                volatility=0.0,
                mean_return=0.0,
            )

        X = np.column_stack(
            [
                returns[valid_mask] * self._scale,
                vol[valid_mask] * self._scale,
            ]
        )

        # 預測
        state_seq = self.model.predict(X)
        state_probs = self.model.predict_proba(X)

        current_state = int(state_seq[-1])
        current_probs = state_probs[-1]
        state_name = self._state_mapping.get(current_state, "unknown")

        state_obj = MarketState(
            state=current_state,
            state_name=state_name,
            probabilities=current_probs,
            volatility=float(vol[valid_mask][-1]),
            mean_return=float(self.model.means_[current_state][0] / self._scale),
        )

        # 追蹤狀態歷史
        self._state_history.append(state_name)

        return state_obj

    def detect_transition(
        self,
        prev_state: str | None = None,
        curr_state: str | None = None,
    ) -> RegimeTransition | None:
        """偵測行情轉場

        Args:
            prev_state: 前一個狀態名稱（若不指定則從歷史推斷）
            curr_state: 當前狀態名稱

        Returns:
            RegimeTransition 或 None（無轉場）
        """
        if prev_state is None:
            if len(self._state_history) < 2:
                return None
            prev_state = self._state_history[-2]

        if curr_state is None:
            if not self._state_history:
                return None
            curr_state = self._state_history[-1]

        if prev_state == curr_state:
            return None

        transition = RegimeTransition.from_states(prev_state, curr_state)
        logger.info(
            "行情轉場: %s → %s (severity=%.1f, action=%s)",
            prev_state,
            curr_state,
            transition.severity,
            transition.action,
        )
        return transition


class EnsemblePredictor:
    """加權集成預測器（支援 HMM 動態權重）"""

    def __init__(
        self,
        lstm_weight: float = 0.6,
        xgb_weight: float = 0.4,
        volatility_multiplier: float = 0.5,
    ):
        self.lstm_weight = lstm_weight
        self.xgb_weight = xgb_weight
        self.volatility_multiplier = volatility_multiplier
        self.hmm: HMMStateDetector | None = None
        self.current_market_state: MarketState | None = None

        # HMM 狀態 → 權重調整策略
        # bear 市場時信號強度降低，sideways 時更保守
        self._state_signal_scale = {
            "bull": 1.0,  # 正常信號
            "sideways": 0.5,  # 減半信號強度
            "bear": 0.3,  # 大幅降低信號強度（主要價值：何時不交易）
        }

    def fit_hmm(self, returns: np.ndarray, volatility: np.ndarray | None = None):
        """訓練 HMM 市場狀態偵測器

        Args:
            returns: 日報酬率序列（至少 60 天）
            volatility: 已實現波動率（可選）
        """
        self.hmm = HMMStateDetector(n_states=3)
        self.hmm.fit(returns, volatility)

    def detect_market_state(
        self,
        returns: np.ndarray,
        volatility: np.ndarray | None = None,
    ) -> MarketState:
        """偵測當前市場狀態（需先 fit_hmm）"""
        if self.hmm is None or not self.hmm.is_fitted:
            return MarketState(
                state=2,
                state_name="sideways",
                probabilities=np.array([0.33, 0.33, 0.34]),
                volatility=0.0,
                mean_return=0.0,
            )
        state = self.hmm.predict_state(returns, volatility)
        self.current_market_state = state
        return state

    def predict(
        self,
        lstm_pred: np.ndarray,
        xgb_pred: np.ndarray,
        current_price: float,
        lstm_history_error: np.ndarray | None = None,
        xgb_history_error: np.ndarray | None = None,
        recent_returns_std: float | None = None,
        recent_returns: np.ndarray | None = None,
        recent_volatility: np.ndarray | None = None,
    ) -> PredictionResult:
        """加權集成預測

        Args:
            lstm_pred: LSTM 預測報酬率 (n_days,)
            xgb_pred: XGBoost 預測報酬率（標量或陣列）
            current_price: 當前收盤價
            lstm_history_error: LSTM 歷史預測誤差（用於信心區間）
            xgb_history_error: XGBoost 歷史預測誤差
            recent_returns_std: 近 20 日報酬率標準差（用於波動率校準閾值）
            recent_returns: 近期日報酬率序列（用於 HMM 狀態偵測）
            recent_volatility: 近期已實現波動率（用於 HMM 狀態偵測）

        Returns:
            PredictionResult
        """
        # 確保維度一致
        lstm_pred = np.atleast_1d(lstm_pred).flatten()
        xgb_pred = np.atleast_1d(xgb_pred).flatten()

        # 若 XGBoost 只輸出一個值，擴展為與 LSTM 相同長度
        if len(xgb_pred) == 1 and len(lstm_pred) > 1:
            xgb_pred = np.full_like(lstm_pred, xgb_pred[0])
        elif len(lstm_pred) == 1 and len(xgb_pred) > 1:
            lstm_pred = np.full_like(xgb_pred, lstm_pred[0])

        n_days = max(len(lstm_pred), len(xgb_pred))

        # 加權平均
        ensemble_returns = (
            self.lstm_weight * lstm_pred[:n_days] + self.xgb_weight * xgb_pred[:n_days]
        )

        # Bug 8 fix: 幾何複利計算預測價格（非算術 cumsum）
        predicted_prices = current_price * np.cumprod(1 + ensemble_returns)

        # Bug 8 fix: Log-space CI（對數常態分佈）
        if lstm_history_error is not None and xgb_history_error is not None:
            combined_error = self.lstm_weight * np.std(
                lstm_history_error
            ) + self.xgb_weight * np.std(xgb_history_error)
        else:
            combined_error = 0.02

        # CI in log-space: log-price uncertainty grows with sqrt(time)
        # 1.5x empirical factor: out-of-sample error is typically 1.5-2x in-sample
        log_std = combined_error * 1.5 * np.sqrt(np.arange(1, n_days + 1))
        log_prices = np.log(predicted_prices)
        confidence_lower = np.exp(log_prices - 1.96 * log_std)
        confidence_upper = np.exp(log_prices + 1.96 * log_std)

        # HMM 市場狀態偵測
        market_state = None
        if recent_returns is not None and self.hmm is not None and self.hmm.is_fitted:
            market_state = self.detect_market_state(recent_returns, recent_volatility)
            logger.info(
                "HMM 市場狀態: %s (prob=%.2f)",
                market_state.state_name,
                market_state.probabilities[market_state.state],
            )

        # Bug 7 fix: 波動率相對閾值
        total_return = ensemble_returns.sum()
        signal, strength = self._generate_signal(total_return, recent_returns_std)

        # HMM 動態調整：根據市場狀態縮放信號強度
        if market_state is not None:
            scale = self._state_signal_scale.get(market_state.state_name, 1.0)
            strength = min(strength * scale, 1.0)

            # 熊市中的 buy 信號降級為 hold（核心價值：何時不交易）
            if market_state.state_name == "bear" and signal == "buy" and strength < 0.5:
                signal = "hold"
                logger.info(
                    "HMM: 熊市中 buy 信號強度不足 (%.2f)，降級為 hold", strength
                )

        return PredictionResult(
            predicted_returns=ensemble_returns,
            predicted_prices=predicted_prices,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            signal=signal,
            signal_strength=strength,
            lstm_weight=self.lstm_weight,
            xgb_weight=self.xgb_weight,
            market_state=market_state,
        )

    def _generate_signal(
        self,
        total_return: float,
        recent_returns_std: float | None = None,
    ) -> tuple[str, float]:
        """根據預測報酬率產生買賣訊號（波動率校準閾值）

        Bug 7 fix: 閾值 = volatility_multiplier * 歷史 std
        而非固定 +/-2%

        Returns:
            (signal, strength)
        """
        if recent_returns_std is not None and recent_returns_std > 0:
            threshold = self.volatility_multiplier * recent_returns_std
        else:
            # Fallback: 預設 2%（低波動環境）
            threshold = 0.02

        # 確保最小閾值避免過度交易
        threshold = max(threshold, 0.005)

        if total_return > threshold:
            strength = min(total_return / (5 * threshold), 1.0)
            return "buy", round(strength, 2)
        elif total_return < -threshold:
            strength = min(abs(total_return) / (5 * threshold), 1.0)
            return "sell", round(strength, 2)
        else:
            strength = 1.0 - abs(total_return) / threshold
            return "hold", round(max(strength, 0), 2)

    def update_weights(
        self,
        lstm_errors: np.ndarray,
        xgb_errors: np.ndarray,
    ):
        """根據驗證集表現動態調整權重

        權重 ∝ 1 / MSE（誤差越小權重越大）
        """
        lstm_mse = np.mean(lstm_errors**2) + 1e-8
        xgb_mse = np.mean(xgb_errors**2) + 1e-8

        lstm_inv = 1 / lstm_mse
        xgb_inv = 1 / xgb_mse
        total_inv = lstm_inv + xgb_inv

        self.lstm_weight = lstm_inv / total_inv
        self.xgb_weight = xgb_inv / total_inv

        logger.info(
            "權重更新: LSTM=%.3f, XGBoost=%.3f",
            self.lstm_weight,
            self.xgb_weight,
        )


class StackingEnsemble:
    """Stacking ensemble — Ridge 回歸作為 meta-learner

    輸入三個模型的 validation predictions，學習最佳組合權重。
    """

    def __init__(self, alpha: float = 5.0):
        from sklearn.linear_model import Ridge

        self.alpha = alpha
        self.meta_learner = Ridge(alpha=alpha)
        self.is_fitted = False
        self.model_names: list[str] = []
        self._use_simple_avg = False

    def fit(
        self,
        model_predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
    ):
        """用 validation set 的模型預測值訓練 meta-learner

        Args:
            model_predictions: {"lstm": preds, "xgboost": preds, "tft": preds}
            y_true: 真實值
        """
        X_meta = np.column_stack(list(model_predictions.values()))
        self.model_names = list(model_predictions.keys())

        # With only 2 base models and < 60 val samples, Ridge weights are unstable
        # Fall back to simple equal-weight averaging
        if len(y_true) < 60 and X_meta.shape[1] <= 2:
            self._use_simple_avg = True
            self.is_fitted = True
            logger.info(
                "StackingEnsemble: val samples=%d < 60 with %d models, "
                "using simple average instead of Ridge",
                len(y_true), X_meta.shape[1],
            )
            return

        self._use_simple_avg = False
        self.meta_learner.fit(X_meta, y_true)
        self.is_fitted = True

        # Log learned weights
        weights = dict(zip(self.model_names, self.meta_learner.coef_))
        logger.info(
            "Stacking 權重: %s (intercept=%.6f)", weights, self.meta_learner.intercept_
        )

    def predict(self, model_predictions: dict[str, np.ndarray]) -> np.ndarray:
        """用 meta-learner 預測"""
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble 尚未 fit")
        X_meta = np.column_stack([model_predictions[name] for name in self.model_names])
        if self._use_simple_avg:
            return X_meta.mean(axis=1)
        return self.meta_learner.predict(X_meta)

    def predict_with_signal(
        self,
        model_predictions: dict[str, np.ndarray],
        current_price: float,
        recent_returns_std: float | None = None,
        market_state: MarketState | None = None,
    ) -> PredictionResult:
        """整合信號生成 + HMM 的預測

        Args:
            model_predictions: 各模型預測值
            current_price: 當前價格
            recent_returns_std: 近 20 日報酬率 std
            market_state: HMM 市場狀態

        Returns:
            PredictionResult
        """
        ensemble_returns = self.predict(model_predictions)
        predicted_prices = current_price * np.cumprod(1 + ensemble_returns)

        # CI (1.5x empirical factor for out-of-sample calibration)
        n_days = len(ensemble_returns)
        log_std = 0.02 * 1.5 * np.sqrt(np.arange(1, n_days + 1))
        log_prices = np.log(predicted_prices)
        confidence_lower = np.exp(log_prices - 1.96 * log_std)
        confidence_upper = np.exp(log_prices + 1.96 * log_std)

        # 信號生成（簡化版）
        total_return = float(ensemble_returns.sum())
        threshold = (
            recent_returns_std * 0.5
            if recent_returns_std and recent_returns_std > 0
            else 0.02
        )
        threshold = max(threshold, 0.005)

        if total_return > threshold:
            signal = "buy"
            strength = min(total_return / (5 * threshold), 1.0)
        elif total_return < -threshold:
            signal = "sell"
            strength = min(abs(total_return) / (5 * threshold), 1.0)
        else:
            signal = "hold"
            strength = 1.0 - abs(total_return) / threshold

        # HMM 調整
        state_scale = {"bull": 1.0, "sideways": 0.5, "bear": 0.3}
        if market_state:
            strength *= state_scale.get(market_state.state_name, 1.0)
            strength = min(strength, 1.0)

        return PredictionResult(
            predicted_returns=ensemble_returns,
            predicted_prices=predicted_prices,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            signal=signal,
            signal_strength=round(max(strength, 0), 2),
            lstm_weight=0.0,
            xgb_weight=0.0,
            market_state=market_state,
        )

    def save(self, path):
        """儲存 StackingEnsemble"""
        import joblib
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "meta_learner": self.meta_learner,
                "model_names": self.model_names,
                "is_fitted": self.is_fitted,
                "_use_simple_avg": self._use_simple_avg,
            },
            path,
        )
        logger.info("StackingEnsemble 已儲存至 %s", path)

    def load(self, path):
        """載入 StackingEnsemble"""
        import joblib
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            logger.warning("StackingEnsemble 檔案不存在: %s", path)
            return
        data = joblib.load(path)
        self.meta_learner = data["meta_learner"]
        self.model_names = data["model_names"]
        self.is_fitted = data["is_fitted"]
        self._use_simple_avg = data.get("_use_simple_avg", False)
        logger.info("StackingEnsemble 已載入自 %s", path)
