"""Quick live scoring check for 5 stocks — compare margin_sentiment before/after fix."""

import sys
import logging
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, ".")

from src.db.database import get_stock_prices, get_data_cache, get_data_cache_latest
from src.data.stock_fetcher import StockFetcher
from src.analysis.technical import TechnicalAnalyzer
from api.services.market_service import (
    score_stock,
    _compute_margin_sentiment,
    _compute_sector_aggregates,
    STOCK_SECTOR,
    DEFAULT_SECTOR,
)

logging.basicConfig(level=logging.WARNING)

TARGETS = ["2330", "2317", "2454", "2881", "2603"]
fetcher = StockFetcher()
analyzer = TechnicalAnalyzer()
today = date.today()
start = (today - timedelta(days=120)).isoformat()
end = today.isoformat()


def fetch_df(sid: str) -> pd.DataFrame:
    df = get_stock_prices(sid)
    if df.empty or len(df) < 20:
        df = fetcher.fetch_all(sid, start, end)
    return df


print(f"=== Live Score Check — {today} ===\n")

# Fetch all data first
stock_dfs = {}
for sid in TARGETS:
    df = fetch_df(sid)
    if not df.empty:
        stock_dfs[sid] = df
        print(f"  {sid}: {len(df)} rows, latest={df['date'].max()}")
    else:
        print(f"  {sid}: NO DATA")

# ── Part 1: margin_sentiment comparison ──
print("\n--- margin_sentiment comparison ---")
print(f"{'Stock':>6}  {'Sector':>14}  {'Raw':>8}  {'Dampened':>8}  {'Delta':>8}  {'Dampened?':>10}")
print("-" * 70)

for sid, df in stock_dfs.items():
    if "margin_balance" not in df.columns:
        print(f"{sid:>6}  {'N/A':>14}  no margin data")
        continue

    raw_result = _compute_margin_sentiment(df)  # no stock_id → no dampening
    dampened_result = _compute_margin_sentiment(df, stock_id=sid)
    sector = STOCK_SECTOR.get(sid, DEFAULT_SECTOR)
    is_dampened = dampened_result.components.get("finance_dampened", False)
    delta = dampened_result.score - raw_result.score

    print(
        f"{sid:>6}  {sector:>14}  {raw_result.score:>8.4f}  "
        f"{dampened_result.score:>8.4f}  {delta:>+8.4f}  {str(is_dampened):>10}"
    )

# ── Part 2: Full score_stock ──
print("\n--- Full 15-factor scoring ---")

# Try to load cached sector data
import json

sector_data = None
cached = get_data_cache("sector_aggregates", today)
if not cached:
    cached = get_data_cache_latest("sector_aggregates")
if cached:
    sector_data = json.loads(cached)
    print(f"  Using cached sector_aggregates ({len(sector_data)} sectors)")
else:
    # Compute from available data
    trust_lookup = {}
    sector_data = _compute_sector_aggregates(stock_dfs, trust_lookup)
    print(f"  Computed sector_aggregates from {len(stock_dfs)} stocks")

print(
    f"\n{'Stock':>6}  {'Sector':>14}  {'Score':>7}  {'Signal':>10}  "
    f"{'Confidence':>10}  {'margin_sent':>12}  {'composite_inst':>14}"
)
print("-" * 90)

for sid, df in stock_dfs.items():
    sector = STOCK_SECTOR.get(sid, DEFAULT_SECTOR)

    # Technical analysis
    try:
        df_tech = analyzer.compute_all(df)
        signals = analyzer.get_signals(df_tech)
    except Exception as e:
        print(f"{sid:>6}  Error computing technicals: {e}")
        continue

    current_price = float(df["close"].iloc[-1])
    prev_price = float(df["close"].iloc[-2]) if len(df) >= 2 else current_price
    pct = ((current_price / prev_price) - 1) * 100

    stock_data = {
        "stock_id": sid,
        "stock_name": sid,
        "current_price": current_price,
        "price_change_pct": round(pct, 2),
    }

    result = score_stock(
        stock_data=stock_data,
        df=df,
        df_tech=df_tech,
        signals=signals,
        trust_info={},
        sentiment_scores={},
        sentiment_df=None,
        ml_scores={},
        regime="sideways",
        sector_data=sector_data,
    )

    total = result.get("total_score", 0)
    signal = result.get("signal", "?")
    conf = result.get("confidence", 0)

    # Extract individual factor scores
    details = result.get("factor_details", {})
    margin_s = details.get("margin_sentiment", {}).get("score", "N/A")
    comp_inst = details.get("composite_institutional", {}).get("score", "N/A")

    margin_str = f"{margin_s:.4f}" if isinstance(margin_s, float) else str(margin_s)
    comp_str = f"{comp_inst:.4f}" if isinstance(comp_inst, float) else str(comp_inst)

    print(
        f"{sid:>6}  {sector:>14}  {total:>7.4f}  {signal:>10}  "
        f"{conf:>10.1f}%  {margin_str:>12}  {comp_str:>14}"
    )

# ── Part 3: Backtest — score N days ago, compare with actual price change ──
print("\n--- Backtest: score from past vs actual price move ---")
LOOKBACK_DAYS = [5, 10, 20]

print(
    f"\n{'Stock':>6}  {'Sector':>14}  "
    + "  ".join(f"{'Score@-'+str(d)+'d':>10}  {'Actual%':>8}" for d in LOOKBACK_DAYS)
)
print("-" * (24 + len(LOOKBACK_DAYS) * 22))

for sid, df in stock_dfs.items():
    if len(df) < max(LOOKBACK_DAYS) + 60:
        print(f"{sid:>6}  insufficient history ({len(df)} rows)")
        continue

    sector = STOCK_SECTOR.get(sid, DEFAULT_SECTOR)
    row_parts = [f"{sid:>6}  {sector:>14}"]

    for lb in LOOKBACK_DAYS:
        # Score using data up to N days ago
        df_past = df.iloc[: -lb].copy()
        if len(df_past) < 60:
            row_parts.append(f"{'N/A':>10}  {'N/A':>8}")
            continue

        try:
            df_tech_past = analyzer.compute_all(df_past)
            signals_past = analyzer.get_signals(df_tech_past)
        except Exception:
            row_parts.append(f"{'ERR':>10}  {'ERR':>8}")
            continue

        price_then = float(df_past["close"].iloc[-1])
        price_now = float(df["close"].iloc[-1])
        actual_ret = ((price_now / price_then) - 1) * 100

        prev = float(df_past["close"].iloc[-2]) if len(df_past) >= 2 else price_then
        pct = ((price_then / prev) - 1) * 100

        past_stock_data = {
            "stock_id": sid,
            "stock_name": sid,
            "current_price": price_then,
            "price_change_pct": round(pct, 2),
        }

        result_past = score_stock(
            stock_data=past_stock_data,
            df=df_past,
            df_tech=df_tech_past,
            signals=signals_past,
            trust_info={},
            sentiment_scores={},
            sentiment_df=None,
            ml_scores={},
            regime="sideways",
            sector_data=sector_data,
        )
        score_past = result_past.get("total_score", 0)
        row_parts.append(f"{score_past:>10.4f}  {actual_ret:>+8.2f}%")

    print("  ".join(row_parts))

# ── Part 4: Score-return correlation summary ──
print("\n--- Score vs Return correlation ---")
for lb in LOOKBACK_DAYS:
    scores = []
    returns = []
    for sid, df in stock_dfs.items():
        if len(df) < lb + 60:
            continue
        df_past = df.iloc[:-lb].copy()
        if len(df_past) < 60:
            continue
        try:
            df_tech_past = analyzer.compute_all(df_past)
            signals_past = analyzer.get_signals(df_tech_past)
        except Exception:
            continue

        price_then = float(df_past["close"].iloc[-1])
        price_now = float(df["close"].iloc[-1])
        actual_ret = ((price_now / price_then) - 1) * 100

        prev = float(df_past["close"].iloc[-2]) if len(df_past) >= 2 else price_then
        pct = ((price_then / prev) - 1) * 100

        result_past = score_stock(
            stock_data={
                "stock_id": sid, "stock_name": sid,
                "current_price": price_then, "price_change_pct": round(pct, 2),
            },
            df=df_past, df_tech=df_tech_past, signals=signals_past,
            trust_info={}, sentiment_scores={}, sentiment_df=None,
            ml_scores={}, regime="sideways", sector_data=sector_data,
        )
        scores.append(result_past.get("total_score", 0.5))
        returns.append(actual_ret)

    if len(scores) >= 3:
        import numpy as _np
        corr = _np.corrcoef(scores, returns)[0, 1]
        print(f"  {lb}d lookback: corr={corr:+.4f}  (n={len(scores)})")
        for s, r, sid in zip(scores, returns, stock_dfs.keys()):
            direction_match = (s > 0.5 and r > 0) or (s < 0.5 and r < 0) or (s == 0.5)
            mark = "Y" if direction_match else "N"
            print(f"    {sid}: score={s:.4f} ret={r:+.2f}% {mark}")
    else:
        print(f"  {lb}d lookback: insufficient data")
