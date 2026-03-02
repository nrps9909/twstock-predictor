"""資料庫連線與 CRUD 操作"""

from datetime import date
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Base, StockPrice, SentimentRecord, Prediction
from src.utils.config import settings


engine = create_engine(settings.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """建立所有資料表"""
    Base.metadata.create_all(engine)


@contextmanager
def get_session():
    """取得 DB session（context manager）"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Stock Prices ─────────────────────────────────────────


def upsert_stock_prices(df: pd.DataFrame, stock_id: str):
    """批次寫入/更新日K線資料

    df 欄位: date, open, high, low, close, volume,
             foreign_buy_sell, trust_buy_sell, dealer_buy_sell,
             margin_balance, short_balance  (後五欄可選)
    """
    with get_session() as session:
        for _, row in df.iterrows():
            existing = session.execute(
                select(StockPrice).where(
                    StockPrice.stock_id == stock_id,
                    StockPrice.date == row["date"],
                )
            ).scalar_one_or_none()

            if existing:
                for col in [
                    "open", "high", "low", "close", "volume",
                    "foreign_buy_sell", "trust_buy_sell", "dealer_buy_sell",
                    "margin_balance", "short_balance",
                ]:
                    if col in row and pd.notna(row.get(col)):
                        setattr(existing, col, row[col])
            else:
                record = StockPrice(
                    stock_id=stock_id,
                    date=row["date"],
                    open=row.get("open"),
                    high=row.get("high"),
                    low=row.get("low"),
                    close=row.get("close"),
                    volume=row.get("volume"),
                    foreign_buy_sell=row.get("foreign_buy_sell"),
                    trust_buy_sell=row.get("trust_buy_sell"),
                    dealer_buy_sell=row.get("dealer_buy_sell"),
                    margin_balance=row.get("margin_balance"),
                    short_balance=row.get("short_balance"),
                    as_of_date=row.get("as_of_date", date.today()),
                )
                session.add(record)


def get_stock_prices(
    stock_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """讀取日K線資料，回傳 DataFrame"""
    with get_session() as session:
        stmt = select(StockPrice).where(StockPrice.stock_id == stock_id)
        if start_date:
            stmt = stmt.where(StockPrice.date >= start_date)
        if end_date:
            stmt = stmt.where(StockPrice.date <= end_date)
        stmt = stmt.order_by(StockPrice.date)

        rows = session.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()

        data = [
            {
                "date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "foreign_buy_sell": r.foreign_buy_sell,
                "trust_buy_sell": r.trust_buy_sell,
                "dealer_buy_sell": r.dealer_buy_sell,
                "margin_balance": r.margin_balance,
                "short_balance": r.short_balance,
            }
            for r in rows
        ]
        return pd.DataFrame(data)


def get_stock_prices_point_in_time(
    stock_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
    as_of_date: date | None = None,
) -> pd.DataFrame:
    """Point-in-time 查詢：只取得在 as_of_date 之前可取得的資料

    避免回測中的前視偏誤。

    Args:
        stock_id: 股票代碼
        start_date: 開始日期
        end_date: 結束日期
        as_of_date: 截止可取得日期（只取 as_of_date <= 此日期的資料）

    Returns:
        DataFrame
    """
    with get_session() as session:
        stmt = select(StockPrice).where(StockPrice.stock_id == stock_id)
        if start_date:
            stmt = stmt.where(StockPrice.date >= start_date)
        if end_date:
            stmt = stmt.where(StockPrice.date <= end_date)
        if as_of_date:
            stmt = stmt.where(
                (StockPrice.as_of_date <= as_of_date) | (StockPrice.as_of_date.is_(None))
            )
        stmt = stmt.order_by(StockPrice.date)

        rows = session.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()

        data = [
            {
                "date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "foreign_buy_sell": r.foreign_buy_sell,
                "trust_buy_sell": r.trust_buy_sell,
                "dealer_buy_sell": r.dealer_buy_sell,
                "margin_balance": r.margin_balance,
                "short_balance": r.short_balance,
                "as_of_date": r.as_of_date,
            }
            for r in rows
        ]
        return pd.DataFrame(data)


# ── Sentiment ────────────────────────────────────────────


def insert_sentiment(records: list[dict]):
    """批次寫入情緒紀錄"""
    with get_session() as session:
        for rec in records:
            if "as_of_date" not in rec:
                rec = {**rec, "as_of_date": date.today()}
            session.add(SentimentRecord(**rec))


def get_sentiment(
    stock_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """讀取情緒紀錄"""
    with get_session() as session:
        stmt = select(SentimentRecord).where(
            SentimentRecord.stock_id == stock_id
        )
        if start_date:
            stmt = stmt.where(SentimentRecord.date >= start_date)
        if end_date:
            stmt = stmt.where(SentimentRecord.date <= end_date)
        stmt = stmt.order_by(SentimentRecord.date)

        rows = session.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()

        data = [
            {
                "date": r.date,
                "source": r.source,
                "title": r.title,
                "sentiment_label": r.sentiment_label,
                "sentiment_score": r.sentiment_score,
                "keywords": r.keywords,
                "engagement": r.engagement,
                "url": r.url,
            }
            for r in rows
        ]
        return pd.DataFrame(data)


# ── Predictions ──────────────────────────────────────────


def insert_predictions(records: list[dict]):
    """批次寫入預測紀錄"""
    with get_session() as session:
        for rec in records:
            session.add(Prediction(**rec))


def get_predictions(
    stock_id: str,
    prediction_date: date | None = None,
    model_type: str = "ensemble",
) -> pd.DataFrame:
    """讀取預測紀錄"""
    with get_session() as session:
        stmt = select(Prediction).where(
            Prediction.stock_id == stock_id,
            Prediction.model_type == model_type,
        )
        if prediction_date:
            stmt = stmt.where(Prediction.prediction_date == prediction_date)
        stmt = stmt.order_by(Prediction.target_date)

        rows = session.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()

        data = [
            {
                "prediction_date": r.prediction_date,
                "target_date": r.target_date,
                "predicted_price": r.predicted_price,
                "predicted_return": r.predicted_return,
                "confidence_lower": r.confidence_lower,
                "confidence_upper": r.confidence_upper,
                "actual_price": r.actual_price,
                "signal": r.signal,
            }
            for r in rows
        ]
        return pd.DataFrame(data)
