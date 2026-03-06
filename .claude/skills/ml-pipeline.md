---
name: ML Training Pipeline
description: >
  ML pipeline: 43 FEATURE_COLUMNS, Triple Barrier labeling, PurgedTimeSeriesSplit,
  CPCV, 3 quality gates, StackingEnsemble architecture, and missing value strategy.
---

# ML Training Pipeline

Sources: `src/analysis/features.py`, `src/models/ensemble.py`, `src/models/trainer.py`

## FEATURE_COLUMNS (43 features)

### Price (8)
close, open, high, low, volume, return_1d, return_5d, return_20d

### Technical Indicators (17)
sma_5, sma_20, sma_60, rsi_14, kd_k, kd_d, macd, macd_signal, macd_hist, bias_5, bias_10, bias_20, bb_upper, bb_lower, bb_width, obv, adx

### Sentiment (5)
sentiment_score, sentiment_ma5, sentiment_change, post_volume, bullish_ratio

### Institutional Flow (5)
foreign_buy_sell, trust_buy_sell, dealer_buy_sell, margin_balance, short_balance

### Volatility (3)
realized_vol_5d, realized_vol_20d, parkinson_vol

### Microstructure (2)
volume_ratio_5d, spread_proxy

### Calendar (3)
day_of_week, month, is_settlement

## Triple Barrier Labeling

```python
triple_barrier_label(df,
    upper_multiplier=2.0,   # ATR * 2.0 for take-profit
    lower_multiplier=2.0,   # ATR * 2.0 for stop-loss
    max_holding=10,          # max 10 days holding period
    atr_window=14,
)
```

Labels are continuous (not categorical). Sample weights computed via `compute_sample_weights(df, label_col, max_holding=10)`.

## PurgedTimeSeriesSplit

```python
PurgedTimeSeriesSplit(
    n_splits=5,
    purge_days=10,     # >= max_holding to avoid label leakage
    embargo_days=5,
)
```

- Removes training samples that overlap with test labels (purge)
- Adds extra safety gap after purge zone (embargo)
- Skips fold if purged training set < 20 samples

## CPCV (Combinatorially Purged Cross-Validation)

```python
cpcv_validate(start_date, end_date,
    n_blocks=6,
    k_test=2,
    seq_len=60,
)
```

Uses `CPCVAnalyzer` to compute PBO (Probability of Backtest Overfitting).

## 3 Quality Gates

All gates must pass before saving trained models:

| Gate | Metric | Threshold |
|------|--------|-----------|
| 1 | Direction accuracy on test set | > 52% |
| 2 | MSE vs naive baseline (predict 0) | MSE < naive MSE |
| 3 | CPCV PBO (overfitting risk) | < 0.6 |

Gate 3 only runs if gate 1+2 pass for at least one model. If PBO > 0.6, both models fail. PBO gate is skipped (not failed) if CPCV errors.

## StackingEnsemble

```
Meta-learner: Ridge regression (alpha=1.0)
Base models: LSTM, XGBoost (optionally TFT)
Input: validation predictions from each base model
Output: weighted combination via learned Ridge coefficients
```

`predict_with_signal()` integrates HMM market state for signal generation.

## Missing Value Strategy

| Category | Strategy |
|----------|----------|
| Sentiment (sentiment_score, sentiment_ma5, etc.) | Fill with 0 |
| Institutional (foreign_buy_sell, margin_balance, etc.) | Forward-fill, then 0 |
| Volatility, microstructure, calendar features | Fill with 0 |
| Technical indicators (warmup period) | Drop rows where sma_60 is NaN |
