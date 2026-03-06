"""系統狀態 API"""

from datetime import date
from fastapi import APIRouter, HTTPException

from src.utils.constants import STOCK_LIST
from src.utils.config import settings
from src.db.database import get_stock_prices

router = APIRouter(prefix="/api", tags=["system"])

MODEL_DIR = settings.PROJECT_ROOT / "models"


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/stocks/{stock_id}/status")
def stock_status(stock_id: str):
    """股票資料/模型狀態"""
    if stock_id not in STOCK_LIST:
        raise HTTPException(404, f"股票 {stock_id} 不在支援清單中")

    # 資料狀態
    df = get_stock_prices(stock_id)
    has_data = not df.empty
    data_count = len(df)
    latest_date = None
    if has_data:
        latest = df["date"].max()
        latest_date = latest.isoformat() if isinstance(latest, date) else str(latest)

    # 模型檔案
    model_files = []
    for suffix in ["_lstm.pt", "_xgb.json", "_tft.ckpt", "_stacking.pkl", "_meta.pkl"]:
        p = MODEL_DIR / f"{stock_id}{suffix}"
        if p.exists():
            model_files.append(p.name)

    return {
        "stock_id": stock_id,
        "name": STOCK_LIST[stock_id],
        "has_model": len(model_files) >= 2,  # 至少 LSTM + XGBoost
        "has_data": has_data,
        "data_count": data_count,
        "latest_date": latest_date,
        "model_files": model_files,
    }
