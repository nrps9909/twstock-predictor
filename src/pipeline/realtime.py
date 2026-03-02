"""即時管線 + 排程

APScheduler 排程：
- 盤前 (08:30): 資料更新、Agent 分析
- 盤中 (09:00-13:30): 每 30 分鐘監控
- 盤後 (14:00): 結算、記憶更新
- 晚間 (20:00): 情緒爬蟲、模型再訓練
"""

import asyncio
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)


class RealtimePipeline:
    """即時交易管線"""

    def __init__(
        self,
        stock_ids: list[str] | None = None,
        orchestrator=None,
        portfolio_manager=None,
    ):
        self.stock_ids = stock_ids or ["2330"]
        self.orchestrator = orchestrator
        self.portfolio = portfolio_manager
        self.scheduler = None
        self._running = False

    def setup_scheduler(self):
        """設定排程任務"""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            logger.error("需要 apscheduler: pip install apscheduler")
            return

        self.scheduler = AsyncIOScheduler()

        # 盤前分析 (08:30 Mon-Fri)
        self.scheduler.add_job(
            self.pre_market_analysis,
            CronTrigger(hour=8, minute=30, day_of_week="mon-fri"),
            id="pre_market",
        )

        # 盤中監控 (09:00-13:30, 每30分鐘)
        self.scheduler.add_job(
            self.intraday_monitor,
            CronTrigger(hour="9-13", minute="0,30", day_of_week="mon-fri"),
            id="intraday",
        )

        # 盤後結算 (14:00)
        self.scheduler.add_job(
            self.post_market_settle,
            CronTrigger(hour=14, minute=0, day_of_week="mon-fri"),
            id="post_market",
        )

        # 晚間更新 (20:00)
        self.scheduler.add_job(
            self.evening_update,
            CronTrigger(hour=20, minute=0, day_of_week="mon-fri"),
            id="evening",
        )

        # 週報 (週六 10:00)
        self.scheduler.add_job(
            self.generate_weekly_report,
            CronTrigger(hour=10, minute=0, day_of_week="sat"),
            id="weekly_report",
        )

        # 月報 (每月 1 日 10:00)
        self.scheduler.add_job(
            self.generate_monthly_report,
            CronTrigger(hour=10, minute=0, day=1),
            id="monthly_report",
        )

        logger.info("排程已設定完成")

    async def start(self):
        """啟動管線"""
        if self.scheduler is None:
            self.setup_scheduler()
        if self.scheduler:
            self.scheduler.start()
            self._running = True
            logger.info("即時管線已啟動")

    async def stop(self):
        """停止管線"""
        if self.scheduler:
            self.scheduler.shutdown()
        self._running = False
        logger.info("即時管線已停止")

    async def pre_market_analysis(self):
        """盤前分析"""
        logger.info("=== 盤前分析 (%s) ===", date.today())

        if not self.orchestrator:
            logger.warning("Orchestrator 未設定")
            return

        from src.agents.base import MarketContext
        from src.data.stock_fetcher import StockFetcher

        fetcher = StockFetcher()

        for stock_id in self.stock_ids:
            try:
                # 取得最新價格
                realtime = fetcher.fetch_realtime(stock_id)
                if not realtime:
                    continue

                context = MarketContext(
                    stock_id=stock_id,
                    current_price=realtime["price"],
                    date=str(date.today()),
                )

                decision = await self.orchestrator.run_analysis(context)
                logger.info(
                    "[%s] 分析結果: %s (信心 %.0f%%)",
                    stock_id, decision.action, decision.confidence * 100,
                )

                # 發送通知
                await self._notify(
                    f"盤前分析 {stock_id}: {decision.action} "
                    f"(信心 {decision.confidence:.0%})\n"
                    f"理由: {decision.reasoning}"
                )

            except Exception as e:
                logger.error("[%s] 盤前分析錯誤: %s", stock_id, e)

    async def intraday_monitor(self):
        """盤中監控 — 檢查停損/止盈"""
        if not self.portfolio:
            return

        from src.data.stock_fetcher import StockFetcher
        fetcher = StockFetcher()

        prices = {}
        for stock_id in self.portfolio.positions:
            realtime = fetcher.fetch_realtime(stock_id)
            if realtime:
                prices[stock_id] = realtime["price"]

        # 檢查停損/止盈
        to_close = self.portfolio.check_stop_loss_take_profit(prices)
        for stock_id in to_close:
            if stock_id in prices:
                result = self.portfolio.close_position(stock_id, prices[stock_id])
                if result:
                    await self._notify(
                        f"停損/止盈觸發 {stock_id}: "
                        f"PnL {result['pnl_pct']:.2%}"
                    )

    async def post_market_settle(self):
        """盤後結算"""
        logger.info("=== 盤後結算 (%s) ===", date.today())
        if self.portfolio:
            summary = self.portfolio.get_summary()
            logger.info("組合摘要: %s", summary)
            await self._notify(f"盤後結算: 總值 ${summary['total_value']:,.0f}")

    async def evening_update(self):
        """晚間更新 — 情緒爬蟲、資料更新、再訓練檢查"""
        logger.info("=== 晚間更新 (%s) ===", date.today())

        # 1. 更新情緒資料（placeholder — 實際爬蟲由外部觸發）
        logger.info("步驟 1: 情緒資料更新（略）")

        # 2. 檢查再訓練觸發
        logger.info("步驟 2: 檢查再訓練觸發...")
        try:
            from src.pipeline.retrain import RetrainTrigger
            from src.analysis.drift import FeatureDriftDetector

            trigger = RetrainTrigger()
            drift_detector = FeatureDriftDetector()

            for stock_id in self.stock_ids:
                try:
                    # 計算特徵漂移 PSI
                    max_psi = 0.0
                    try:
                        drift_result = drift_detector.compute_drift(stock_id)
                        if drift_result:
                            max_psi = max(drift_result.values()) if drift_result else 0.0
                    except Exception:
                        logger.debug("[%s] PSI 計算略過", stock_id)

                    # 計算滾動 Sharpe（從預測紀錄）
                    rolling_sharpe = None
                    try:
                        from src.db.database import get_predictions
                        preds = get_predictions(stock_id)
                        if not preds.empty and "predicted_return" in preds.columns:
                            returns = preds["predicted_return"].dropna().tail(30)
                            if len(returns) > 5:
                                rolling_sharpe = float(
                                    returns.mean() / (returns.std() + 1e-8) * (252 ** 0.5)
                                )
                    except Exception:
                        logger.debug("[%s] 滾動 Sharpe 計算略過", stock_id)

                    should_retrain, reason = trigger.should_retrain(
                        max_psi=max_psi,
                        rolling_sharpe=rolling_sharpe,
                    )
                    if should_retrain:
                        logger.info("[%s] 觸發再訓練: %s", stock_id, reason)
                        result = trigger.execute_retrain(stock_id, epochs=30)
                        await self._notify(
                            f"自動再訓練 {stock_id}: {reason}\n結果: {result.get('config', {}).get('n_features', 'N/A')} 特徵"
                        )
                    else:
                        logger.info("[%s] 無需再訓練", stock_id)
                except Exception as e:
                    logger.error("[%s] 再訓練檢查失敗: %s", stock_id, e)
        except ImportError as e:
            logger.debug("再訓練模組不可用: %s", e)

        # 3. 記憶評估
        if self.orchestrator and hasattr(self.orchestrator, 'memory'):
            logger.info("步驟 3: 記憶評估...")
            recent = self.orchestrator.memory.short_term.get_recent(5)
            logger.info("近 5 天記憶: %d 筆", len(recent))

    async def generate_weekly_report(self):
        """生成並發送週報"""
        logger.info("=== 生成週報 ===")
        try:
            from src.monitoring.reports import ReportGenerator
            reporter = ReportGenerator()
            report = reporter.weekly_report(self.stock_ids)
            logger.info("\n%s", report)
            await self._notify(f"週報已生成\n{report[:500]}")
        except Exception as e:
            logger.error("週報生成失敗: %s", e)

    async def generate_monthly_report(self):
        """生成並發送月報"""
        logger.info("=== 生成月報 ===")
        try:
            from src.monitoring.reports import ReportGenerator
            reporter = ReportGenerator()
            report = reporter.monthly_report(self.stock_ids)
            logger.info("\n%s", report)
            await self._notify(f"月報已生成\n{report[:500]}")
        except Exception as e:
            logger.error("月報生成失敗: %s", e)

    async def _notify(self, message: str):
        """發送通知（LINE / Telegram）"""
        logger.info("通知: %s", message)
        # 實際推播由 monitoring/alerts.py 處理
        try:
            from src.monitoring.alerts import AlertManager
            alert_mgr = AlertManager()
            await alert_mgr.send(message)
        except Exception as e:
            logger.debug("通知發送失敗（非致命）: %s", e)
