---
name: Data Sources and Fallback
description: >
  Data fetching priority, fallback ticker mappings, rate limits, retry logic,
  FinMind field mappings, and volume unit conversions.
---

# Data Sources and Fallback

Sources: `src/data/stock_fetcher.py`, `src/data/twse_scanner.py`, `src/utils/retry.py`

## Data Source Priority

1. **FinMind** (primary) — Taiwan market data API
2. **yfinance** (fallback) — Yahoo Finance
3. **Proxy/TWSE** (last resort) — Direct TWSE scraping

## Rate Limits

| Source | Limit | Implementation |
|--------|-------|----------------|
| FinMind | 2 calls/sec | `RateLimiter(calls_per_second=2.0)` |
| TWSE | 1 call/sec | `RateLimiter(calls_per_second=1.0)` |

TWSE also has a TTL cache of 300 seconds (5 min) for T86 API responses.

## FinMind Field Mappings

```python
# OHLCV
"Trading_Volume" -> "volume"
"max" -> "high"
"min" -> "low"
# open, close unchanged

# Volume: shares -> lots (張)
df["volume"] = df["volume"] / 1000

# Institutional
"Foreign_Investor" / "外資" -> foreign
"Investment_Trust" / "投信" -> trust
"Dealer" / "自營" -> dealer

# Margin
"MarginPurchaseTodayBalance" -> margin_balance
"ShortSaleTodayBalance" -> short_balance

# Fundamentals
"PER" / "本益比" -> pe_ratio
"PBR" / "股價淨值比" -> pb_ratio
"殖利率(%)" / "殖利率" -> dividend_yield
```

## TWSE Volume Conversion

All TWSE institutional volumes are converted from shares to lots: `// 1000`

## Retry Logic (`retry_with_backoff`)

```
max_retries=3, base_delay=1.0s, max_delay=30s, backoff_factor=2.0
```

Progression: 1s -> 2s -> 4s -> ... (capped at 30s)

### Transient Status Codes (retried)
`{408, 429, 500, 502, 503, 504}`

### Transient Exceptions (retried)
- `requests.ConnectionError`, `requests.Timeout`
- `httpx.ConnectError`, `httpx.ReadTimeout`, `httpx.ConnectTimeout`

### Permanent Errors
`PermanentError` exception — immediately raised, never retried.

## RateLimiter

Token-bucket style: tracks `_last_call` timestamp, sleeps if `elapsed < min_interval`.
