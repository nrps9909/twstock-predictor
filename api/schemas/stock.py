"""股票相關 Pydantic 模型"""

from datetime import date
from pydantic import BaseModel


class StockInfo(BaseModel):
    stock_id: str
    name: str


class StockPrice(BaseModel):
    date: date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    foreign_buy_sell: float | None = None
    trust_buy_sell: float | None = None
    dealer_buy_sell: float | None = None
    margin_balance: float | None = None
    short_balance: float | None = None


class FetchRequest(BaseModel):
    start_date: str
    end_date: str


class StockStatus(BaseModel):
    stock_id: str
    name: str
    has_model: bool
    has_data: bool
    data_count: int
    latest_date: str | None = None
    model_files: list[str] = []


class TechnicalSignal(BaseModel):
    signal: str
    reason: str


class TechnicalResult(BaseModel):
    signals: dict[str, TechnicalSignal]
    summary: dict
    latest_price: float | None = None
    indicators: dict = {}


class SentimentSummary(BaseModel):
    total_records: int
    avg_score: float
    bullish_ratio: float
    bearish_ratio: float
    neutral_ratio: float
    latest_date: str | None = None
    by_source: dict = {}
