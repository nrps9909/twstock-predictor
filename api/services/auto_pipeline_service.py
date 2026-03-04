"""每日自動深度分析 Pipeline — 使用統一管線

收盤後自動對法人交易量前 N 大個股跑統一 6 階段管線:
- 數據收集 → 特徵萃取 → 多因子評分 → LLM 敘事 → 風控 → 儲存
- 結果存入 MarketScanResult + PipelineResult 表
"""

import asyncio
import logging
from datetime import date

from src.db.database import (
    get_top_institutional_stocks,
    get_pipeline_results_batch,
)
from src.utils.constants import STOCK_LIST

logger = logging.getLogger(__name__)


async def run_single_pipeline(stock_id: str, target_date: date | None = None) -> dict:
    """跑單一個股的統一 6 階段管線

    委派給 StockAnalysisService.analyze_stock()，
    消費 SSE 串流取得最終結果。

    Returns:
        最終分析結果 dict (已由統一管線自動存入 DB)
    """
    import json

    target_date = target_date or date.today()
    stock_name = STOCK_LIST.get(stock_id, stock_id)

    logger.info("Running unified pipeline for %s %s ...", stock_id, stock_name)

    try:
        from api.services.stock_analysis_service import StockAnalysisService
        service = StockAnalysisService()

        final_result = None
        async for event_str in service.analyze_stock(stock_id, stock_name):
            # Parse SSE events to find the final result
            for line in event_str.strip().split("\n"):
                if line.startswith("data: "):
                    try:
                        payload = json.loads(line[6:])
                        if payload.get("phase") == "complete" and payload.get("status") == "done":
                            final_result = payload.get("data", {})
                    except (json.JSONDecodeError, KeyError):
                        pass

        if final_result:
            logger.info("Pipeline completed for %s: signal=%s, score=%.2f",
                        stock_id,
                        final_result.get("signal", "N/A"),
                        final_result.get("total_score", 0))
            return final_result

        logger.warning("Pipeline for %s produced no result", stock_id)
        return {
            "stock_id": stock_id,
            "analysis_date": target_date.isoformat(),
            "signal": "hold",
            "confidence": 0.0,
            "reasoning": "統一管線未產生結果",
            "pipeline_version": "3.0",
        }

    except Exception as e:
        logger.error("Pipeline failed for %s: %s", stock_id, e)
        return {
            "stock_id": stock_id,
            "analysis_date": target_date.isoformat(),
            "signal": "hold",
            "confidence": 0.0,
            "reasoning": f"管線錯誤: {e}",
            "pipeline_version": "3.0",
        }


async def run_daily_pipeline(target_date: date | None = None, top_n: int = 50) -> dict:
    """每日收盤後自動跑統一管線

    1. 從最新 MarketScanResult 取外資+投信交易量前 N 大
    2. 過濾掉今天已分析過的
    3. 逐一跑統一管線
    """
    target_date = target_date or date.today()
    logger.info("Starting daily pipeline for %s (top %d)...", target_date, top_n)

    # 1. Get top institutional stocks
    stock_ids = await asyncio.to_thread(get_top_institutional_stocks, top_n)
    if not stock_ids:
        logger.warning("No market scan results found, skipping pipeline")
        return {"total": 0, "completed": 0, "skipped": 0, "failed": 0}

    # 2. Filter out already analyzed
    existing = await asyncio.to_thread(get_pipeline_results_batch, stock_ids, target_date)
    existing_ids = {r["stock_id"] for r in existing}
    todo = [sid for sid in stock_ids if sid not in existing_ids]

    logger.info("Pipeline: %d total, %d already done, %d to process",
                len(stock_ids), len(existing_ids), len(todo))

    # 3. Run unified pipeline for each stock
    completed = 0
    failed = 0
    for stock_id in todo:
        try:
            await run_single_pipeline(stock_id, target_date)
            completed += 1
            logger.info("Pipeline completed for %s (%d/%d)", stock_id, completed, len(todo))
        except Exception as e:
            logger.error("Pipeline failed for %s: %s", stock_id, e)
            failed += 1

    logger.info("Daily pipeline finished: %d completed, %d failed, %d skipped",
                completed, failed, len(existing_ids))

    return {
        "total": len(stock_ids),
        "completed": completed,
        "skipped": len(existing_ids),
        "failed": failed,
    }


async def trigger_reanalysis(stock_ids: list[str]):
    """重大警報 → 立即重跑統一管線更新信心"""
    today = date.today()
    logger.info("Trigger reanalysis for %d stocks: %s", len(stock_ids), stock_ids)
    for sid in stock_ids:
        try:
            await run_single_pipeline(sid, today)
            logger.info("Reanalysis completed for %s", sid)
        except Exception as e:
            logger.error("Reanalysis failed for %s: %s", sid, e)
