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

# ── Logging: 確保 INFO 層級的訊息印在 terminal ──────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# Suppress ta library divide-by-zero warnings (harmless, clutters logs)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ta")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import (
    stocks,
    technical,
    sentiment,
    prediction,
    agent,
    pipeline,
    system,
    market,
    alerts,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle"""
    from src.db.database import init_db
    from src.utils.llm_client import check_claude_available

    init_db()
    check_claude_available()

    yield


app = FastAPI(
    title="台股 AI 預測系統 API",
    version="3.0.0",
    description="台股走勢預測系統 REST API — 統一 6 階段管線、全市場掃描、20 因子評分、LLM 敘事生成",
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
