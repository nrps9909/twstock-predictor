---
name: Testing Patterns
description: >
  Test fixtures, naming conventions, AAA style, mocking patterns, and
  pytest-asyncio usage in the twstock-predictor test suite.
---

# Testing Patterns

Sources: `tests/conftest.py`, `tests/test_factor_scoring.py`

## Key Fixtures (`conftest.py`)

### `in_memory_db`
SQLite in-memory engine. Creates all tables, yields engine, disposes after test.

### `sample_price_df` (200 rows, seed=42)
- Base price 100 TWD + random walk
- Columns: date, open, high, low, close, volume, foreign_buy_sell, trust_buy_sell, dealer_buy_sell, margin_balance, short_balance
- `np.random.seed(42)`, 200 rows starting from 2025-01-02

### `sample_features_df` (depends on `sample_price_df`, ~43 features)
Adds on top of `sample_price_df`:
- **Returns**: return_1d, return_5d, return_20d
- **Technical**: sma_5/20/60, rsi_14, kd_k/d, macd/signal/hist, bias_5/10/20, bb_upper/lower/width, obv, adx
- **Sentiment**: sentiment_score, sentiment_ma5, sentiment_change, post_volume, bullish_ratio
- **Volatility**: realized_vol_5d, realized_vol_20d, parkinson_vol
- **Microstructure**: volume_ratio_5d, spread_proxy
- **Calendar**: day_of_week, month, is_settlement
- **Targets**: return_next_5d, tb_label, sample_weight
- Drops first 60 rows (warmup), fills NaN with 0

## Test Naming Convention

```
test_{module}_{scenario}
```

Tests are organized in classes by feature:
```python
class TestForeignFlow:
    def test_basic(self): ...
    def test_no_data(self): ...
    def test_heavy_buying(self): ...
    def test_heavy_selling(self): ...

class TestWeightEngine:
    def test_all_available_weights_sum_to_one(self): ...
    def test_missing_data_redistribution(self): ...
```

## AAA Style (Arrange-Act-Assert)

```python
def test_all_available_weights_sum_to_one(self):
    # Arrange
    factors = [FactorResult(name, 0.6, True, 1.0) for name in BASE_WEIGHTS]

    # Act
    weights = _compute_weights(factors, "sideways")

    # Assert
    assert abs(sum(weights.values()) - 1.0) < 1e-6
```

## Mocking Patterns

- Use `@patch("api.services.market_service._compute_xxx")` for expensive factor computations
- Return `FactorResult(name, 0.5, False, 0.0)` as neutral stub
- `AsyncMock` imported in `test_stock_analysis_service.py` for async `call_claude`
- `MagicMock` for sync methods like `trainer.predict`

## pytest-asyncio

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_call()
    assert result is not None
```

## Running Tests

```bash
.venv/Scripts/python -m pytest tests/ -v
.venv/Scripts/python -m pytest tests/test_factor_scoring.py -v  # single file
.venv/Scripts/python -m pytest tests/ -k "test_basic" -v        # by name
```
