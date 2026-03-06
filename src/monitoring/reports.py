"""週報/月報生成器

- weekly_report(): 週 P&L、信號準確度、漂移摘要
- monthly_report(): 累積績效、模型準確度、交易日誌統計
"""

import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)


class ReportGenerator:
    """績效報告生成器"""

    def __init__(self, session_factory=None):
        self.session_factory = session_factory

    def weekly_report(
        self,
        stock_ids: list[str] | None = None,
        week_end: date | None = None,
    ) -> str:
        """生成週報

        Args:
            stock_ids: 關注的股票清單
            week_end: 週報截止日（預設為今天）

        Returns:
            格式化的週報文字
        """
        if week_end is None:
            week_end = date.today()
        week_start = week_end - timedelta(days=7)

        lines = [
            "=" * 60,
            f"週報 ({week_start} ~ {week_end})",
            "=" * 60,
        ]

        # 1. P&L 摘要
        lines.append("\n--- 損益摘要 ---")
        pnl_data = self._get_pnl_summary(week_start, week_end)
        if pnl_data:
            for item in pnl_data:
                lines.append(
                    f"  {item['stock_id']}: {item['pnl_pct']:+.2%} "
                    f"(trades={item['trades']})"
                )
        else:
            lines.append("  （本週無交易）")

        # 2. 信號準確度
        lines.append("\n--- 信號準確度 ---")
        accuracy = self._get_signal_accuracy(week_start, week_end)
        if accuracy is not None:
            lines.append(f"  方向準確率: {accuracy:.1%}")
        else:
            lines.append("  （資料不足）")

        # 3. 特徵漂移摘要
        lines.append("\n--- 特徵漂移 ---")
        drift = self._get_drift_summary()
        if drift:
            for feat, psi in drift.items():
                status = "OK" if psi < 0.1 else ("注意" if psi < 0.25 else "警告")
                lines.append(f"  {feat}: PSI={psi:.3f} [{status}]")
        else:
            lines.append("  （無漂移資料）")

        lines.append("=" * 60)
        report = "\n".join(lines)
        logger.info("週報已生成 (%d 行)", len(lines))
        return report

    def monthly_report(
        self,
        stock_ids: list[str] | None = None,
        month_end: date | None = None,
    ) -> str:
        """生成月報

        Args:
            stock_ids: 關注的股票清單
            month_end: 月報截止日

        Returns:
            格式化的月報文字
        """
        if month_end is None:
            month_end = date.today()
        month_start = month_end.replace(day=1)

        lines = [
            "=" * 60,
            f"月報 ({month_start} ~ {month_end})",
            "=" * 60,
        ]

        # 1. 累積績效
        lines.append("\n--- 累積績效 ---")
        monthly_pnl = self._get_pnl_summary(month_start, month_end)
        total_pnl = (
            sum(item.get("pnl_pct", 0) for item in monthly_pnl) if monthly_pnl else 0
        )
        total_trades = (
            sum(item.get("trades", 0) for item in monthly_pnl) if monthly_pnl else 0
        )
        lines.append(f"  月累積損益: {total_pnl:+.2%}")
        lines.append(f"  月交易次數: {total_trades}")

        # 2. 模型準確度
        lines.append("\n--- 模型準確度 ---")
        accuracy = self._get_signal_accuracy(month_start, month_end)
        if accuracy is not None:
            lines.append(f"  方向準確率: {accuracy:.1%}")
        else:
            lines.append("  （資料不足）")

        # 3. 交易日誌統計
        lines.append("\n--- 交易日誌統計 ---")
        if monthly_pnl:
            wins = [p for p in monthly_pnl if p.get("pnl_pct", 0) > 0]
            losses = [p for p in monthly_pnl if p.get("pnl_pct", 0) < 0]
            win_rate = len(wins) / len(monthly_pnl) if monthly_pnl else 0
            lines.append(f"  勝率: {win_rate:.1%} ({len(wins)}勝 {len(losses)}敗)")
        else:
            lines.append("  （無交易紀錄）")

        lines.append("=" * 60)
        report = "\n".join(lines)
        logger.info("月報已生成 (%d 行)", len(lines))
        return report

    # ── 私有查詢方法 ──

    def _get_pnl_summary(self, start: date, end: date) -> list[dict]:
        """查詢期間 P&L（從 TradeJournal 或 BacktestResult）"""
        if self.session_factory is None:
            return []

        try:
            from sqlalchemy import select
            from src.db.models import TradeJournal

            session = self.session_factory()
            try:
                stmt = (
                    select(TradeJournal)
                    .where(
                        TradeJournal.trade_date >= start,
                        TradeJournal.trade_date <= end,
                    )
                    .order_by(TradeJournal.trade_date)
                )
                rows = session.execute(stmt).scalars().all()

                # 按股票聚合
                stock_pnl: dict[str, dict] = {}
                for r in rows:
                    sid = r.stock_id
                    if sid not in stock_pnl:
                        stock_pnl[sid] = {"stock_id": sid, "pnl_pct": 0, "trades": 0}
                    if r.pnl_pct:
                        stock_pnl[sid]["pnl_pct"] += r.pnl_pct
                    stock_pnl[sid]["trades"] += 1

                return list(stock_pnl.values())
            finally:
                session.close()
        except Exception as e:
            logger.debug("查詢 P&L 失敗: %s", e)
            return []

    def _get_signal_accuracy(self, start: date, end: date) -> float | None:
        """查詢信號準確度"""
        if self.session_factory is None:
            return None

        try:
            from sqlalchemy import select
            from src.db.models import Prediction

            session = self.session_factory()
            try:
                stmt = select(Prediction).where(
                    Prediction.prediction_date >= start,
                    Prediction.prediction_date <= end,
                    Prediction.actual_price.isnot(None),
                )
                rows = session.execute(stmt).scalars().all()
                if not rows:
                    return None

                correct = sum(
                    1
                    for r in rows
                    if r.predicted_return
                    and r.actual_price
                    and r.predicted_price
                    and (r.actual_price > r.predicted_price) == (r.predicted_return > 0)
                )
                return correct / len(rows)
            finally:
                session.close()
        except Exception as e:
            logger.debug("查詢準確度失敗: %s", e)
            return None

    def _get_drift_summary(self) -> dict[str, float]:
        """取得最新特徵漂移摘要

        比較最近兩次訓練的特徵重要性，計算各特徵的 PSI 近似值。
        """
        if self.session_factory is None:
            return {}

        try:
            from sqlalchemy import select
            from src.db.models import FeatureImportanceRecord

            session = self.session_factory()
            try:
                # 取得最近兩個 run_date
                dates_stmt = (
                    select(FeatureImportanceRecord.run_date)
                    .distinct()
                    .order_by(FeatureImportanceRecord.run_date.desc())
                    .limit(2)
                )
                dates = [row[0] for row in session.execute(dates_stmt).all()]

                if len(dates) < 2:
                    return {}

                # 取得兩次的重要性分數
                recent = {}
                prev = {}
                for d in dates:
                    stmt = select(FeatureImportanceRecord).where(
                        FeatureImportanceRecord.run_date == d
                    )
                    rows = session.execute(stmt).scalars().all()
                    scores = {r.feature_name: r.importance_score for r in rows}
                    if d == dates[0]:
                        recent = scores
                    else:
                        prev = scores

                # 計算 PSI 近似（重要性分數分布偏移）
                result = {}
                for feat in recent:
                    if feat in prev and prev[feat] > 0 and recent[feat] > 0:
                        import math

                        r, p = recent[feat], prev[feat]
                        # 簡化 PSI: (r - p) * ln(r / p)
                        psi = (r - p) * math.log(r / p)
                        result[feat] = abs(psi)

                return result
            finally:
                session.close()
        except Exception as e:
            logger.debug("取得漂移摘要失敗: %s", e)
            return {}
