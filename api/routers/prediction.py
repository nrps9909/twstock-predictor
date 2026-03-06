"""模型訓練與預測 API"""

import asyncio
from fastapi import APIRouter, HTTPException

from src.utils.constants import STOCK_LIST
from src.models.trainer import ModelTrainer
from api.schemas.prediction import TrainRequest, PredictRequest, PredictionResponse

router = APIRouter(prefix="/api/stocks", tags=["prediction"])


@router.post("/{stock_id}/train")
async def train_model(stock_id: str, req: TrainRequest):
    """訓練模型"""
    if stock_id not in STOCK_LIST:
        raise HTTPException(404, f"股票 {stock_id} 不在支援清單中")

    trainer = ModelTrainer(stock_id)

    try:
        result = await asyncio.to_thread(
            trainer.train,
            start_date=req.start_date,
            end_date=req.end_date,
            epochs=req.epochs,
            use_triple_barrier=req.use_triple_barrier,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"訓練失敗: {e}")

    return {"status": "ok", "result": _sanitize(result)}


@router.post("/{stock_id}/predict", response_model=PredictionResponse)
async def predict(stock_id: str, req: PredictRequest):
    """執行預測"""
    if stock_id not in STOCK_LIST:
        raise HTTPException(404, f"股票 {stock_id} 不在支援清單中")

    trainer = ModelTrainer(stock_id)
    try:
        trainer.load_models()
    except Exception:
        raise HTTPException(404, "模型不存在，請先訓練")

    if trainer.lstm is None and trainer.xgb is None:
        raise HTTPException(404, "模型不存在，請先訓練")

    result = await asyncio.to_thread(
        trainer.predict,
        start_date=req.start_date,
        end_date=req.end_date,
    )

    if result is None:
        raise HTTPException(500, "預測失敗")

    market_state_dict = None
    if result.market_state is not None:
        market_state_dict = {
            "state": result.market_state.state,
            "state_name": result.market_state.state_name,
            "probabilities": result.market_state.probabilities.tolist(),
            "volatility": float(result.market_state.volatility),
            "mean_return": float(result.market_state.mean_return),
        }

    return PredictionResponse(
        predicted_returns=result.predicted_returns.tolist(),
        predicted_prices=result.predicted_prices.tolist(),
        confidence_lower=result.confidence_lower.tolist(),
        confidence_upper=result.confidence_upper.tolist(),
        signal=result.signal,
        signal_strength=float(result.signal_strength),
        lstm_weight=float(result.lstm_weight),
        xgb_weight=float(result.xgb_weight),
        market_state=market_state_dict,
    )


def _sanitize(obj):
    """遞迴清理結果中的 numpy 物件"""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
