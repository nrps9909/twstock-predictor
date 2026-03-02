"""共用 pytest fixtures — in-memory DB, sample DataFrames"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base


@pytest.fixture
def in_memory_db():
    """SQLite in-memory engine"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


# Alias for backward compatibility
@pytest.fixture
def in_memory_engine(in_memory_db):
    """Alias for in_memory_db"""
    return in_memory_db


@pytest.fixture
def session_factory(in_memory_engine):
    """Session factory bound to in-memory DB"""
    return sessionmaker(bind=in_memory_engine)


@pytest.fixture
def sample_price_df():
    """合成股價 DataFrame（100 天）

    包含 OHLCV + institutional + margin 欄位。
    價格模式：基礎 100 元 + 隨機遊走。
    """
    np.random.seed(42)
    n = 200
    dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(n)]

    # 隨機遊走價格
    returns = np.random.normal(0.001, 0.02, n)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_ = close * (1 + np.random.normal(0, 0.005, n))

    df = pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(5000, 50000, n).astype(float),
        "foreign_buy_sell": np.random.normal(0, 500, n),
        "trust_buy_sell": np.random.normal(0, 200, n),
        "dealer_buy_sell": np.random.normal(0, 100, n),
        "margin_balance": np.random.randint(10000, 50000, n).astype(float),
        "short_balance": np.random.randint(1000, 5000, n).astype(float),
    })
    return df


@pytest.fixture
def sample_features_df(sample_price_df):
    """合成特徵 DataFrame（含技術指標 + 目標欄位）

    在 sample_price_df 基礎上加入：
    - 技術指標 stub（SMA, RSI, MACD 等）
    - 情緒指標 stub
    - target 欄位（tb_label, return_next_5d）
    """
    df = sample_price_df.copy()
    n = len(df)
    np.random.seed(42)

    # 基本報酬率
    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)

    # 技術指標 stub
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_60"] = df["close"].rolling(60).mean()
    df["rsi_14"] = 50 + np.random.normal(0, 15, n)  # stub RSI
    df["kd_k"] = 50 + np.random.normal(0, 20, n)
    df["kd_d"] = 50 + np.random.normal(0, 15, n)
    df["macd"] = np.random.normal(0, 1, n)
    df["macd_signal"] = np.random.normal(0, 0.8, n)
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["bias_5"] = (df["close"] / df["sma_5"] - 1) * 100
    df["bias_10"] = np.random.normal(0, 2, n)
    df["bias_20"] = np.random.normal(0, 3, n)
    df["bb_upper"] = df["sma_20"] + 2 * df["close"].rolling(20).std()
    df["bb_lower"] = df["sma_20"] - 2 * df["close"].rolling(20).std()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["obv"] = np.cumsum(np.random.normal(0, 1000, n))
    df["adx"] = 20 + np.random.normal(0, 10, n)

    # 情緒指標 stub
    df["sentiment_score"] = np.random.uniform(-0.5, 0.5, n)
    df["sentiment_ma5"] = df["sentiment_score"].rolling(5).mean()
    df["sentiment_change"] = df["sentiment_score"].diff()
    df["post_volume"] = np.random.randint(10, 100, n)
    df["bullish_ratio"] = np.random.uniform(0.3, 0.7, n)

    # 波動率特徵
    df["realized_vol_5d"] = df["return_1d"].rolling(5).std()
    df["realized_vol_20d"] = df["return_1d"].rolling(20).std()
    log_hl = np.log(df["high"] / df["low"])
    df["parkinson_vol"] = (log_hl ** 2 / (4 * np.log(2))).rolling(20).mean().apply(np.sqrt)

    # 微結構
    df["volume_ratio_5d"] = df["volume"] / df["volume"].rolling(5).mean()
    df["spread_proxy"] = (df["high"] - df["low"]) / df["close"]

    # 日曆
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_settlement"] = 0.0

    # Targets
    df["return_next_5d"] = df["close"].pct_change(5).shift(-5)
    df["tb_label"] = np.random.normal(0, 0.03, n)
    df["sample_weight"] = np.random.uniform(0.5, 1.0, n)

    # 填充 NaN
    df = df.fillna(0)
    # 確保 sma_60 前面有值（丟掉暖身期）
    df = df.iloc[60:].reset_index(drop=True)

    return df
