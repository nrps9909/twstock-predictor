"""預測相關 Pydantic 模型"""

from pydantic import BaseModel


class TrainRequest(BaseModel):
    start_date: str
    end_date: str
    epochs: int = 50
    use_triple_barrier: bool = True


class PredictRequest(BaseModel):
    start_date: str
    end_date: str


class PredictionResponse(BaseModel):
    predicted_returns: list[float]
    predicted_prices: list[float]
    confidence_lower: list[float]
    confidence_upper: list[float]
    signal: str
    signal_strength: float
    lstm_weight: float
    xgb_weight: float
    market_state: dict | None = None
