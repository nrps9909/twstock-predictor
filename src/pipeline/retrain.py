"""自動再訓練觸發器

4 個觸發條件：
1. 特徵漂移 PSI > 0.25
2. 滾動 Sharpe < 0
3. 距上次訓練 > 30 天
4. HMM 行情轉場（Bull→Bear 或反向）
"""

import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)


class RetrainTrigger:
    """自動再訓練觸發器"""

    def __init__(
        self,
        psi_threshold: float = 0.25,
        sharpe_threshold: float = 0.0,
        max_days_since_train: int = 30,
        last_train_date: date | None = None,
    ):
        self.psi_threshold = psi_threshold
        self.sharpe_threshold = sharpe_threshold
        self.max_days_since_train = max_days_since_train
        self.last_train_date = last_train_date

    def should_retrain(
        self,
        max_psi: float = 0.0,
        rolling_sharpe: float | None = None,
        regime_transition=None,
    ) -> tuple[bool, str]:
        """檢查是否需要再訓練

        Args:
            max_psi: 最大特徵漂移 PSI 值
            rolling_sharpe: 近期滾動 Sharpe ratio
            regime_transition: RegimeTransition 物件（可選）

        Returns:
            (should_retrain: bool, reason: str)
        """
        reasons = []

        # 1. 特徵漂移
        if max_psi > self.psi_threshold:
            reasons.append(f"特徵漂移 PSI={max_psi:.3f} > {self.psi_threshold}")

        # 2. 滾動 Sharpe
        if rolling_sharpe is not None and rolling_sharpe < self.sharpe_threshold:
            reasons.append(f"滾動 Sharpe={rolling_sharpe:.2f} < {self.sharpe_threshold}")

        # 3. 距上次訓練天數
        if self.last_train_date is not None:
            days_since = (date.today() - self.last_train_date).days
            if days_since > self.max_days_since_train:
                reasons.append(f"距上次訓練 {days_since} 天 > {self.max_days_since_train}")

        # 4. HMM 行情轉場
        if regime_transition is not None and regime_transition.severity > 0.7:
            reasons.append(
                f"行情轉場 {regime_transition.prev_state}→{regime_transition.curr_state} "
                f"(severity={regime_transition.severity:.1f})"
            )

        if reasons:
            reason = "; ".join(reasons)
            logger.info("再訓練觸發: %s", reason)
            return True, reason

        return False, "無需再訓練"

    def execute_retrain(
        self,
        stock_id: str,
        lookback_days: int = 365,
        **train_kwargs,
    ) -> dict:
        """執行再訓練

        Args:
            stock_id: 股票代碼
            lookback_days: 訓練資料回看天數

        Returns:
            訓練結果 dict
        """
        from src.models.trainer import ModelTrainer

        end_date = str(date.today())
        start_date = str(date.today() - timedelta(days=lookback_days))

        logger.info("開始再訓練 %s (%s ~ %s)...", stock_id, start_date, end_date)

        trainer = ModelTrainer(stock_id)
        try:
            results = trainer.train(
                start_date=start_date,
                end_date=end_date,
                **train_kwargs,
            )
            self.last_train_date = date.today()
            logger.info("再訓練完成: %s", stock_id)
            return results
        except Exception as e:
            logger.error("再訓練失敗: %s — %s", stock_id, e)
            return {"error": str(e)}
