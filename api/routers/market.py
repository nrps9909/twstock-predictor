"""市場掃描 API — 全市場掃描、推薦、情報、深度分析、統一個股分析"""

import asyncio
import logging
from datetime import date

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse

from api.services.market_service import run_market_scan, get_market_overview

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/market", tags=["market"])


@router.post("/scan")
async def scan_market(top_n: int = 40):
    """觸發全市場掃描 (SSE 串流)"""
    return StreamingResponse(
        run_market_scan(top_n=top_n),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/overview")
async def market_overview():
    """最新掃描結果"""
    return await get_market_overview()


@router.get("/recommendations")
async def market_recommendations():
    """買賣推薦"""
    overview = await get_market_overview()
    return {
        "scan_date": overview.get("scan_date"),
        "buy": overview.get("buy_recommendations", []),
        "sell": overview.get("sell_recommendations", []),
    }


@router.get("/intel")
async def market_intel():
    """市場情報（新聞 + 法人）— 此端點較慢，需抓取 TWSE"""
    from api.services.market_intel_service import get_market_intel
    try:
        return await asyncio.wait_for(get_market_intel(), timeout=30)
    except asyncio.TimeoutError:
        logger.warning("Market intel timed out")
        return {
            "global_news": [], "tw_news": [],
            "trust_top_buy": [], "trust_top_sell": [],
            "foreign_top_buy": [], "sync_buy": [],
            "institutional_total": {"date": "", "foreign": 0, "trust": 0, "dealer": 0, "total": 0},
        }
    except Exception as e:
        logger.error("Market intel error: %s", e)
        return {
            "global_news": [], "tw_news": [],
            "trust_top_buy": [], "trust_top_sell": [],
            "foreign_top_buy": [], "sync_buy": [],
            "institutional_total": {"date": "", "foreign": 0, "trust": 0, "dealer": 0, "total": 0},
        }


@router.get("/institutional")
async def institutional_overview():
    """三大法人整體概況（僅抓最新一日，較快）"""
    from src.data.twse_scanner import TWSEScanner
    try:
        scanner = TWSEScanner()
        summary = await asyncio.wait_for(
            asyncio.to_thread(scanner.get_institutional_summary),
            timeout=15,
        )
        return {"summary": summary}
    except asyncio.TimeoutError:
        logger.warning("Institutional summary timed out")
        return {"summary": {"date": "", "foreign_total": 0, "trust_total": 0, "dealer_total": 0, "total": 0}}
    except Exception as e:
        logger.error("Institutional summary error: %s", e)
        return {"summary": {"date": "", "foreign_total": 0, "trust_total": 0, "dealer_total": 0, "total": 0}}


# ── Pipeline endpoints ─────────────────────────────


@router.get("/pipeline/batch")
async def get_pipeline_batch(stock_ids: str = ""):
    """批次取得 pipeline 結果 (query: stock_ids=2330,2317,...)"""
    from src.db.database import get_pipeline_results_batch
    ids = [s.strip() for s in stock_ids.split(",") if s.strip()]
    if not ids:
        return []
    results = await asyncio.to_thread(get_pipeline_results_batch, ids)
    return results


@router.get("/pipeline/{stock_id}")
async def get_pipeline(stock_id: str):
    """取得個股最新 pipeline 分析結果"""
    from src.db.database import get_pipeline_result
    result = await asyncio.to_thread(get_pipeline_result, stock_id)
    if not result:
        return {"status": "not_found", "stock_id": stock_id}
    return result


@router.post("/pipeline/{stock_id}")
async def trigger_pipeline(stock_id: str, background_tasks: BackgroundTasks):
    """手動觸發單一個股 pipeline（背景執行，使用統一管線）"""
    from api.services.auto_pipeline_service import run_single_pipeline

    async def _run():
        await run_single_pipeline(stock_id)

    background_tasks.add_task(_run)
    return {"status": "started", "stock_id": stock_id}


@router.post("/analyze/{stock_id}")
async def analyze_stock(stock_id: str, stock_name: str = ""):
    """統一個股深度分析 (SSE 串流) — 6 階段管線"""
    logger.info("▶ POST /api/market/analyze/%s (name=%s)", stock_id, stock_name or "auto")
    from api.services.stock_analysis_service import analyze_stock as _analyze
    return StreamingResponse(
        _analyze(stock_id, stock_name),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/factor-ic")
async def factor_ic(factor: str = "", window: int = 60):
    """取得因子 IC 資料（滾動 Spearman IC）"""
    from src.db.database import get_factor_ic_rolling
    if not factor:
        return {"error": "factor parameter required"}
    result = await asyncio.to_thread(get_factor_ic_rolling, factor, window)
    return result
