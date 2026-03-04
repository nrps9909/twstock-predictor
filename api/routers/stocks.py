"""股票資料 API"""

from datetime import date, timedelta
from fastapi import APIRouter, HTTPException, Query

from src.utils.constants import STOCK_LIST
from src.db.database import get_stock_prices, upsert_stock_prices
from src.data.stock_fetcher import StockFetcher
from api.schemas.stock import StockInfo, StockPrice, FetchRequest

router = APIRouter(prefix="/api/stocks", tags=["stocks"])


@router.get("", response_model=list[StockInfo])
def list_stocks():
    """取得支援的股票清單"""
    return [StockInfo(stock_id=k, name=v) for k, v in STOCK_LIST.items()]


@router.get("/{stock_id}/prices")
def get_prices(
    stock_id: str,
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    limit: int = Query(250, ge=1, le=1000),
):
    """取得歷史價格"""
    if stock_id not in STOCK_LIST:
        raise HTTPException(404, f"股票 {stock_id} 不在支援清單中")

    sd = date.fromisoformat(start_date) if start_date else None
    ed = date.fromisoformat(end_date) if end_date else None

    df = get_stock_prices(stock_id, sd, ed)
    if df.empty:
        return []

    # 限制筆數（取最近的）
    df = df.tail(limit)

    records = df.to_dict("records")
    # 轉換 date 物件為字串
    for r in records:
        if isinstance(r.get("date"), date):
            r["date"] = r["date"].isoformat()
    return records


@router.post("/{stock_id}/fetch")
def fetch_data(stock_id: str, req: FetchRequest):
    """從 FinMind 抓取並存入 DB"""
    if stock_id not in STOCK_LIST:
        raise HTTPException(404, f"股票 {stock_id} 不在支援清單中")

    fetcher = StockFetcher()
    df = fetcher.fetch_all(stock_id, req.start_date, req.end_date)
    if df.empty:
        raise HTTPException(500, "無法從 FinMind 取得資料")

    upsert_stock_prices(df, stock_id)
    return {"status": "ok", "rows": len(df)}


@router.get("/{stock_id}/realtime")
def get_realtime(stock_id: str):
    """即時報價"""
    result = StockFetcher.fetch_realtime(stock_id)
    if result is None:
        raise HTTPException(503, "無法取得即時報價")
    return result
