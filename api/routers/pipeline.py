"""一鍵預測 Pipeline API (SSE) + 預測歷史 API"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.db.database import get_prediction_history, update_prediction_actuals
from api.schemas.pipeline import PipelineRequest
from api.services.pipeline_service import run_pipeline

router = APIRouter(prefix="/api", tags=["pipeline"])


@router.post("/stocks/{stock_id}/pipeline")
async def pipeline(stock_id: str, req: PipelineRequest | None = None):
    """一鍵預測 — SSE 串流回傳 9 步驟進度"""
    if not stock_id.isdigit() or len(stock_id) != 4:
        raise HTTPException(400, f"無效的股票代碼: {stock_id}")

    force_retrain = req.force_retrain if req else False
    epochs = req.epochs if req else 50

    return StreamingResponse(
        run_pipeline(stock_id, force_retrain=force_retrain, epochs=epochs),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/predictions/history")
async def prediction_history(
    stock_id: str | None = Query(None, description="股票代碼（不填則全部）"),
    limit: int = Query(50, ge=1, le=200),
):
    """取得預測歷史紀錄"""
    # Backfill actuals for requested stock
    if stock_id:
        try:
            update_prediction_actuals(stock_id)
        except Exception:
            pass  # non-critical

    records = get_prediction_history(stock_id=stock_id, limit=limit)
    return records


@router.get("/predictions/recent")
async def recent_predictions(
    limit: int = Query(20, ge=1, le=100),
):
    """取得最近的預測紀錄（所有股票）"""
    records = get_prediction_history(stock_id=None, limit=limit)
    return records
