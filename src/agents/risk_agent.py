"""風控 Agent — 風險管理（核准/否決交易）

結合規則引擎 + LLM 判斷，確保交易決策符合風險限制。
"""

import logging
from dataclasses import dataclass

from src.agents.base import (
    AgentMessage, AgentRole, BaseAgent, MarketContext, Signal, TradeDecision,
)
from src.utils.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """風險限制參數"""
    max_position_pct: float = 0.20  # 單一個股最大倉位
    max_total_positions: int = 5  # 最多持倉數
    max_daily_loss_pct: float = 0.03  # 單日最大虧損
    max_weekly_loss_pct: float = 0.05  # 單週最大虧損
    max_monthly_loss_pct: float = 0.10  # 單月最大虧損
    min_confidence: float = 0.4  # 最低信心度要求
    max_stop_loss_pct: float = 0.08  # 最大停損幅度
    min_risk_reward_ratio: float = 2.0  # 最低風險報酬比


class RiskAgent(BaseAgent):
    """風控 Agent — 規則引擎 + LLM 輔助"""

    def __init__(self, limits: RiskLimits | None = None):
        super().__init__(AgentRole.RISK)
        self.limits = limits or RiskLimits()
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.monthly_pnl: float = 0.0
        self.current_positions: int = 0

    async def analyze(self, context: MarketContext) -> AgentMessage:
        """基本風險評估（不含具體交易決策）"""
        return AgentMessage(
            sender=self.role,
            content={"status": "ready"},
            signal=Signal.HOLD,
            confidence=1.0,
            reasoning="風控系統就緒",
        )

    def evaluate_trade(
        self,
        decision: TradeDecision,
        context: MarketContext,
        portfolio_value: float = 1_000_000,
    ) -> tuple[bool, str, TradeDecision]:
        """評估交易決策是否符合風險限制

        Returns:
            (approved: bool, reason: str, adjusted_decision: TradeDecision)
        """
        violations = []
        adjusted = decision

        # 1. 信心度檢查
        if decision.confidence < self.limits.min_confidence:
            violations.append(
                f"信心度不足: {decision.confidence:.0%} < {self.limits.min_confidence:.0%}"
            )

        # 2. 倉位大小檢查
        if decision.position_size > self.limits.max_position_pct:
            adjusted.position_size = self.limits.max_position_pct
            violations.append(
                f"倉位調降: {decision.position_size:.0%} → {self.limits.max_position_pct:.0%}"
            )

        # 3. 持倉數量檢查
        if decision.action == "buy" and self.current_positions >= self.limits.max_total_positions:
            violations.append(
                f"持倉已滿: {self.current_positions}/{self.limits.max_total_positions}"
            )
            return False, "持倉數量已達上限", adjusted

        # 4. 停損幅度檢查
        if decision.stop_loss and context.current_price > 0:
            stop_loss_pct = abs(decision.stop_loss - context.current_price) / context.current_price
            if stop_loss_pct > self.limits.max_stop_loss_pct:
                # 強制調整停損
                adjusted.stop_loss = context.current_price * (1 - self.limits.max_stop_loss_pct)
                violations.append(
                    f"停損調整: {stop_loss_pct:.1%} → {self.limits.max_stop_loss_pct:.1%}"
                )

        # 5. 風險報酬比檢查
        if decision.stop_loss and decision.take_profit and context.current_price > 0:
            risk = abs(context.current_price - decision.stop_loss)
            reward = abs(decision.take_profit - context.current_price)
            if risk > 0:
                rr_ratio = reward / risk
                if rr_ratio < self.limits.min_risk_reward_ratio:
                    violations.append(
                        f"風險報酬比不足: {rr_ratio:.1f} < {self.limits.min_risk_reward_ratio:.1f}"
                    )

        # 6. 回撤限制
        if self.daily_pnl < -self.limits.max_daily_loss_pct:
            violations.append(f"當日虧損已達上限: {self.daily_pnl:.1%}")
            return False, "當日虧損已達上限，暫停交易", adjusted

        if self.weekly_pnl < -self.limits.max_weekly_loss_pct:
            violations.append(f"當週虧損已達上限: {self.weekly_pnl:.1%}")
            return False, "當週虧損已達上限，暫停交易", adjusted

        if self.monthly_pnl < -self.limits.max_monthly_loss_pct:
            violations.append(f"當月虧損已達上限: {self.monthly_pnl:.1%}")
            return False, "當月虧損已達上限，暫停交易", adjusted

        # 決策
        if decision.action == "hold":
            return True, "Hold 決策，無需風控", adjusted

        # 有 violation 但非致命的 → 核准但附帶警告
        critical_violations = [v for v in violations if "不足" in v or "已滿" in v]
        if critical_violations:
            reason = "風控否決: " + "; ".join(critical_violations)
            adjusted.approved_by_risk = False
            adjusted.risk_notes = reason
            return False, reason, adjusted

        # 核准（可能帶調整）
        adjusted.approved_by_risk = True
        if violations:
            adjusted.risk_notes = "核准（已調整）: " + "; ".join(violations)
            return True, adjusted.risk_notes, adjusted
        else:
            adjusted.risk_notes = "風控通過"
            return True, "風控通過", adjusted

    def update_pnl(self, daily: float = 0, weekly: float = 0, monthly: float = 0):
        """更新損益追蹤"""
        self.daily_pnl = daily
        self.weekly_pnl = weekly
        self.monthly_pnl = monthly

    def kelly_position_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.25,  # 用 1/4 Kelly 較保守
    ) -> float:
        """Kelly criterion 倉位計算

        Args:
            win_rate: 歷史勝率
            avg_win: 平均獲利幅度
            avg_loss: 平均虧損幅度
            kelly_fraction: Kelly 比例（0.25 = quarter Kelly）

        Returns:
            建議倉位比例 (0.0 ~ max_position_pct)
        """
        if avg_loss == 0 or avg_win == 0:
            return 0.0

        b = avg_win / avg_loss  # odds ratio
        p = win_rate
        q = 1 - p

        kelly = (b * p - q) / b
        position = max(0, kelly * kelly_fraction)
        return min(position, self.limits.max_position_pct)
