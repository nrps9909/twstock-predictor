"""情緒分析 API"""

from datetime import date, timedelta
from fastapi import APIRouter, HTTPException, Query

from src.utils.constants import STOCK_LIST
from src.db.database import get_sentiment

router = APIRouter(prefix="/api/stocks", tags=["sentiment"])


@router.get("/{stock_id}/sentiment/summary")
def get_sentiment_summary(
    stock_id: str,
    days: int = Query(30, ge=1, le=365),
):
    """取得情緒聚合摘要"""
    if stock_id not in STOCK_LIST:
        raise HTTPException(404, f"股票 {stock_id} 不在支援清單中")

    end = date.today()
    start = end - timedelta(days=days)

    df = get_sentiment(stock_id, start, end)
    if df.empty:
        return {
            "total_records": 0,
            "avg_score": 0.0,
            "bullish_ratio": 0.0,
            "bearish_ratio": 0.0,
            "neutral_ratio": 0.0,
            "latest_date": None,
            "by_source": {},
            "records": [],
        }

    total = len(df)
    bullish = len(df[df["sentiment_label"] == "bullish"])
    bearish = len(df[df["sentiment_label"] == "bearish"])
    neutral = total - bullish - bearish

    # 按來源統計
    by_source = {}
    for source in df["source"].unique():
        sub = df[df["source"] == source]
        by_source[source] = {
            "count": len(sub),
            "avg_score": round(float(sub["sentiment_score"].mean()), 3),
        }

    # 最近 10 筆紀錄
    recent = df.tail(10).to_dict("records")
    for r in recent:
        if isinstance(r.get("date"), date):
            r["date"] = r["date"].isoformat()

    return {
        "total_records": total,
        "avg_score": round(float(df["sentiment_score"].mean()), 3),
        "bullish_ratio": round(bullish / total, 3),
        "bearish_ratio": round(bearish / total, 3),
        "neutral_ratio": round(neutral / total, 3),
        "latest_date": df["date"].max().isoformat() if not df.empty else None,
        "by_source": by_source,
        "records": recent,
    }
