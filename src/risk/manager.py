"""風險管理模組

Kelly criterion 倉位、ATR 停損/追蹤停損、回撤限制、最大回撤熔斷

硬性風控原則（LLM 無法覆蓋）：
1. 1/4 Kelly 倉位上限
2. ATR Trailing Stop（動態追蹤停損）
3. 最大回撤熔斷機制
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrailingStopState:
    """ATR Trailing Stop 追蹤狀態"""

    entry_price: float
    highest_price: float  # 持倉期間最高價
    atr: float
    multiplier: float
    current_stop: float  # 當前追蹤停損價
    triggered: bool = False

    @property
    def trailing_distance(self) -> float:
        return self.multiplier * self.atr

    def update(self, current_high: float) -> float:
        """更新追蹤停損（只能上移，不能下移）

        Args:
            current_high: 當日最高價

        Returns:
            新的追蹤停損價
        """
        if current_high > self.highest_price:
            self.highest_price = current_high
            new_stop = self.highest_price - self.trailing_distance
            if new_stop > self.current_stop:
                self.current_stop = new_stop
        return self.current_stop

    def check_trigger(self, current_low: float) -> bool:
        """檢查是否觸發停損

        Args:
            current_low: 當日最低價

        Returns:
            是否觸發
        """
        if current_low <= self.current_stop:
            self.triggered = True
        return self.triggered


class RiskManager:
    """風險管理器 — 硬性風控（LLM 無法覆蓋）"""

    def __init__(
        self,
        max_position_pct: float = 0.20,
        max_portfolio_risk: float = 0.02,  # 單筆交易最大組合風險 2%
        max_drawdown_limit: float = 0.15,  # 最大回撤熔斷線 15%
    ):
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk = max_portfolio_risk
        self.max_drawdown_limit = max_drawdown_limit

        # 追蹤停損狀態 {stock_id: TrailingStopState}
        self._trailing_stops: dict[str, TrailingStopState] = {}

        # 權益曲線追蹤（用於熔斷）
        self._equity_curve: list[float] = []
        self._circuit_breaker_active: bool = False

    def kelly_position(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.25,
    ) -> float:
        """Kelly criterion 倉位計算（使用 1/4 Kelly）"""
        if avg_loss == 0 or avg_win == 0:
            return 0.0
        b = avg_win / avg_loss
        kelly = (b * win_rate - (1 - win_rate)) / b
        return max(0, min(kelly * fraction, self.max_position_pct))

    # ── ATR 停損 ──────────────────────────────────────────

    def atr_stop_loss(
        self,
        current_price: float,
        atr: float,
        multiplier: float = 2.0,
    ) -> float:
        """ATR-based 靜態停損價"""
        return current_price - multiplier * atr

    def atr_take_profit(
        self,
        current_price: float,
        atr: float,
        multiplier: float = 4.0,
    ) -> float:
        """ATR-based 止盈價"""
        return current_price + multiplier * atr

    # ── ATR Trailing Stop ─────────────────────────────────

    def init_trailing_stop(
        self,
        stock_id: str,
        entry_price: float,
        atr: float,
        multiplier: float = 2.5,
    ) -> TrailingStopState:
        """初始化 ATR Trailing Stop

        Args:
            stock_id: 股票代碼
            entry_price: 進場價格
            atr: 當前 ATR
            multiplier: ATR 倍數（預設 2.5x，比靜態 2x 稍寬以避免噪音觸發）

        Returns:
            TrailingStopState
        """
        initial_stop = entry_price - multiplier * atr
        state = TrailingStopState(
            entry_price=entry_price,
            highest_price=entry_price,
            atr=atr,
            multiplier=multiplier,
            current_stop=initial_stop,
        )
        self._trailing_stops[stock_id] = state
        logger.info(
            "Trailing Stop 初始化 [%s]: entry=%.2f, stop=%.2f, ATR=%.2f, mult=%.1f",
            stock_id,
            entry_price,
            initial_stop,
            atr,
            multiplier,
        )
        return state

    def update_trailing_stop(
        self,
        stock_id: str,
        current_high: float,
        current_low: float,
    ) -> tuple[float, bool]:
        """更新 Trailing Stop 並檢查是否觸發

        Args:
            stock_id: 股票代碼
            current_high: 當日最高價
            current_low: 當日最低價

        Returns:
            (current_stop_price, triggered)
        """
        state = self._trailing_stops.get(stock_id)
        if state is None:
            logger.warning("Trailing Stop 未初始化: %s", stock_id)
            return 0.0, False

        old_stop = state.current_stop
        state.update(current_high)
        triggered = state.check_trigger(current_low)

        if state.current_stop > old_stop:
            logger.info(
                "Trailing Stop 上移 [%s]: %.2f → %.2f (highest=%.2f)",
                stock_id,
                old_stop,
                state.current_stop,
                state.highest_price,
            )

        if triggered:
            logger.info(
                "Trailing Stop 觸發 [%s]: stop=%.2f, low=%.2f",
                stock_id,
                state.current_stop,
                current_low,
            )

        return state.current_stop, triggered

    def close_trailing_stop(self, stock_id: str):
        """關閉 Trailing Stop（平倉後）"""
        if stock_id in self._trailing_stops:
            del self._trailing_stops[stock_id]

    def get_trailing_stop_state(self, stock_id: str) -> TrailingStopState | None:
        """取得 Trailing Stop 狀態"""
        return self._trailing_stops.get(stock_id)

    # ── 倉位管理 ─────────────────────────────────────────

    def position_size_by_risk(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> int:
        """根據固定風險比例計算持股數

        確保單筆虧損不超過 portfolio_value * max_portfolio_risk

        Returns:
            建議持股數（整張，最小 1000 股）
        """
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share <= 0:
            return 0

        max_loss = portfolio_value * self.max_portfolio_risk
        max_shares = max_loss / risk_per_share

        # 台股 1 張 = 1000 股
        lots = int(max_shares / 1000)
        # 不超過最大倉位比例
        max_lots_by_position = int(
            portfolio_value * self.max_position_pct / entry_price / 1000
        )
        return min(lots, max_lots_by_position) * 1000

    # ── 回撤追蹤 + 熔斷 ──────────────────────────────────

    def update_equity(self, current_value: float):
        """更新權益曲線"""
        self._equity_curve.append(current_value)
        drawdown_info = self.calculate_drawdown(self._equity_curve)

        # 熔斷檢查
        if drawdown_info["max_drawdown"] < -self.max_drawdown_limit:
            if not self._circuit_breaker_active:
                self._circuit_breaker_active = True
                logger.warning(
                    "最大回撤熔斷觸發! drawdown=%.1f%% > limit=%.1f%%",
                    drawdown_info["max_drawdown"] * 100,
                    self.max_drawdown_limit * 100,
                )

    def is_circuit_breaker_active(self) -> bool:
        """是否處於熔斷狀態"""
        return self._circuit_breaker_active

    def reset_circuit_breaker(self):
        """重置熔斷（手動操作，需要人類確認）"""
        self._circuit_breaker_active = False
        logger.info("最大回撤熔斷已重置")

    def calculate_drawdown(self, equity_curve: list[float]) -> dict:
        """計算回撤指標"""
        if not equity_curve:
            return {"max_drawdown": 0, "current_drawdown": 0}

        arr = np.array(equity_curve)
        peak = np.maximum.accumulate(arr)
        drawdowns = (arr - peak) / peak

        return {
            "max_drawdown": float(np.min(drawdowns)),
            "current_drawdown": float(drawdowns[-1]),
            "peak_value": float(peak[-1]),
            "current_value": float(arr[-1]),
        }

    # ── 行情轉場風控 ────────────────────────────────────

    def regime_transition_check(
        self,
        transition,
        positions: dict,
    ) -> list[dict]:
        """根據行情轉場產生減倉指令

        Args:
            transition: RegimeTransition 物件
            positions: {stock_id: Position} dict

        Returns:
            list of {"stock_id": str, "action": str, "reduce_pct": float}
        """
        if transition is None or transition.action == "no_action":
            return []

        orders = []
        if transition.action == "reduce_50%":
            for stock_id in positions:
                orders.append(
                    {
                        "stock_id": stock_id,
                        "action": "reduce",
                        "reduce_pct": 0.5,
                        "reason": f"行情轉場 {transition.prev_state}→{transition.curr_state}",
                    }
                )
            logger.warning(
                "行情轉場減倉: %s→%s, 減倉 50%% (%d 檔)",
                transition.prev_state,
                transition.curr_state,
                len(orders),
            )
        elif transition.action == "close_all":
            for stock_id in positions:
                orders.append(
                    {
                        "stock_id": stock_id,
                        "action": "close",
                        "reduce_pct": 1.0,
                        "reason": f"行情轉場 {transition.prev_state}→{transition.curr_state}",
                    }
                )
            logger.warning("行情轉場清倉: %d 檔", len(orders))

        return orders

    # ── 硬性風控檢查（LLM 無法覆蓋）────────────────────

    def hard_risk_check(
        self,
        action: str,
        stock_id: str,
        portfolio_value: float,
        position_size_pct: float,
    ) -> tuple[bool, str]:
        """硬性風控檢查 — 這些規則無論 LLM/Agent 怎麼說都不可違反

        Returns:
            (passed, reason)
        """
        # 1. 熔斷狀態 → 禁止一切買入
        if self._circuit_breaker_active and action == "buy":
            return False, f"熔斷中 (drawdown > {self.max_drawdown_limit:.0%})，禁止買入"

        # 2. 最大單一倉位限制
        if position_size_pct > self.max_position_pct:
            return (
                False,
                f"倉位超限: {position_size_pct:.0%} > {self.max_position_pct:.0%}",
            )

        # 3. Trailing Stop 已觸發 → 不可加倉
        ts = self._trailing_stops.get(stock_id)
        if ts and ts.triggered and action == "buy":
            return False, f"Trailing Stop 已觸發 ({stock_id})，禁止加倉"

        return True, "風控通過"
