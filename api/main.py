"""FastAPI 主應用程式"""

# ── 最早期：移除 Claude Code 巢狀偵測變數 ─────────────
# 讓 server 的 claude -p subprocess 不會被擋
# 必須在所有 import 之前執行
import os as _os
for _k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
    _os.environ.pop(_k, None)

import logging
import warnings
from contextlib import asynccontextmanager

# Suppress ta library divide-by-zero warnings (harmless, clutters logs)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ta")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import stocks, technical, sentiment, prediction, agent, pipeline, system, market, alerts

logger = logging.getLogger(__name__)

# Background scheduler reference
_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle"""
    global _scheduler

    # Startup
    from src.db.database import init_db
    from src.utils.llm_client import check_claude_available
    init_db()
    check_claude_available()

    # Start background scheduler
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        _scheduler = AsyncIOScheduler()
        _scheduler.add_job(
            _scheduled_market_scan,
            "cron",
            hour=16, minute=30,  # 每日 16:30（TWSE T86 約 16:00 後才有資料）
            id="daily_market_scan",
            replace_existing=True,
        )
        _scheduler.add_job(
            _scheduled_daily_pipeline,
            "cron",
            hour=17, minute=0,  # 每日 17:00（等 market scan 完成後）
            id="daily_pipeline",
            replace_existing=True,
        )
        _scheduler.add_job(
            _scheduled_ic_backfill,
            "cron",
            hour=17, minute=30,  # 每日 17:30（回填因子遠期報酬）
            id="daily_ic_backfill",
            replace_existing=True,
        )
        _scheduler.start()
        logger.info("Background scheduler started (scan 16:30, pipeline 17:00, IC 17:30)")
    except ImportError:
        logger.warning("apscheduler not installed — auto market scan disabled")
    except Exception as e:
        logger.warning("Scheduler start failed: %s", e)

    yield

    # Shutdown
    if _scheduler:
        _scheduler.shutdown(wait=False)


async def _scheduled_market_scan():
    """排程執行的市場掃描"""
    from api.services.market_service import run_market_scan
    logger.info("Starting scheduled market scan...")
    try:
        async for event in run_market_scan(top_n=40):
            pass  # Consume the generator to execute the scan
        logger.info("Scheduled market scan completed")
    except Exception as e:
        logger.error("Scheduled market scan failed: %s", e)


async def _scheduled_daily_pipeline():
    """排程執行的每日深度分析 pipeline"""
    from api.services.auto_pipeline_service import run_daily_pipeline
    logger.info("Starting scheduled daily pipeline...")
    try:
        result = await run_daily_pipeline(top_n=50)
        logger.info("Scheduled daily pipeline completed: %s", result)
    except Exception as e:
        logger.error("Scheduled daily pipeline failed: %s", e)


async def _scheduled_ic_backfill():
    """排程回填因子 IC 遠期報酬"""
    import asyncio
    from src.db.database import backfill_forward_returns
    logger.info("Starting IC backfill...")
    try:
        await asyncio.to_thread(backfill_forward_returns, 5)
        logger.info("IC backfill completed")
    except Exception as e:
        logger.error("IC backfill failed: %s", e)


app = FastAPI(
    title="台股 AI 預測系統 API",
    version="2.0.0",
    description="台股走勢預測系統 REST API — 全市場掃描、技術分析、情緒分析、ML 預測、Multi-Agent 分析",
    lifespan=lifespan,
)

# CORS — 允許 Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛載 routers
app.include_router(market.router)
app.include_router(alerts.router)
app.include_router(stocks.router)
app.include_router(technical.router)
app.include_router(sentiment.router)
app.include_router(prediction.router)
app.include_router(agent.router)
app.include_router(pipeline.router)
app.include_router(system.router)
