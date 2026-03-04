"""技術分析 API"""

from datetime import date, timedelta
from fastapi import APIRouter, HTTPException, Query

from src.utils.constants import STOCK_LIST
from src.db.database import get_stock_prices
from src.analysis.technical import TechnicalAnalyzer

router = APIRouter(prefix="/api/stocks", tags=["technical"])


@router.get("/{stock_id}/technical")
def get_technical(
    stock_id: str,
    days: int = Query(120, ge=30, le=500),
):
    """計算技術指標 + 訊號"""
    if stock_id not in STOCK_LIST:
        raise HTTPException(404, f"股票 {stock_id} 不在支援清單中")

    end = date.today()
    start = end - timedelta(days=days + 60)  # 多抓 60 天供指標暖機

    df = get_stock_prices(stock_id, start, end)
    if df.empty or len(df) < 30:
        raise HTTPException(404, "資料不足，請先抓取資料")

    analyzer = TechnicalAnalyzer()
    df_tech = analyzer.compute_all(df)
    signals = analyzer.get_signals(df_tech)

    # 只回傳最近 days 筆 + 關鍵指標
    df_recent = df_tech.tail(days)

    latest = df_recent.iloc[-1]
    indicators = {}
    for col in ["rsi_14", "kd_k", "kd_d", "macd", "macd_signal", "macd_hist",
                 "sma_5", "sma_20", "sma_60", "bb_upper", "bb_lower", "bb_middle",
                 "adx", "bias_10", "bb_pband", "obv"]:
        if col in latest.index:
            val = latest[col]
            indicators[col] = None if (val != val) else round(float(val), 4)  # NaN check

    # 圖表資料
    chart_data = []
    for _, row in df_recent.iterrows():
        d = {
            "date": row["date"].isoformat() if isinstance(row["date"], date) else str(row["date"]),
            "open": row.get("open"),
            "high": row.get("high"),
            "low": row.get("low"),
            "close": row.get("close"),
            "volume": row.get("volume"),
        }
        for col in ["sma_5", "sma_20", "sma_60", "kd_k", "kd_d", "rsi_14",
                     "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower", "bb_middle"]:
            if col in row.index:
                val = row[col]
                d[col] = None if (val != val) else round(float(val), 4)
        chart_data.append(d)

    return {
        "signals": signals,
        "indicators": indicators,
        "latest_price": float(latest["close"]) if "close" in latest.index else None,
        "chart_data": chart_data,
    }
