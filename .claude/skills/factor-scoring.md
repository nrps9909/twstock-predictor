---
name: 20-Factor Scoring Engine
description: >
  The 20-factor scoring system in market_service.py. Covers BASE_WEIGHTS,
  REGIME_MULTIPLIERS, FactorResult dataclass, confidence calculation, missing
  factor weight redistribution, and IC tracking.
---

# 20-Factor Scoring Engine

Source: `api/services/market_service.py`

## BASE_WEIGHTS (20 factors, sum = 1.00)

### Short-term (39%)
| Factor | Weight | Description |
|--------|--------|-------------|
| foreign_flow | 0.11 | Foreign investor flow |
| technical_signal | 0.08 | Technical signal aggregate |
| short_momentum | 0.07 | Short-term momentum |
| trust_flow | 0.05 | Investment trust flow |
| volume_anomaly | 0.04 | Volume anomaly |
| margin_sentiment | 0.04 | Margin trading sentiment |

### Mid-term (32%)
| Factor | Weight | Description |
|--------|--------|-------------|
| trend_momentum | 0.07 | Mid-term trend momentum |
| revenue_momentum | 0.04 | Monthly revenue momentum |
| institutional_sync | 0.04 | Institutional synchronization |
| volatility_regime | 0.04 | Volatility regime state |
| news_sentiment | 0.03 | News sentiment |
| global_context | 0.03 | Global market correlation |
| margin_quality | 0.04 | Quarterly gross margin trend |
| sector_rotation | 0.03 | Sector capital rotation |

### Long-term (29%)
| Factor | Weight | Description |
|--------|--------|-------------|
| ml_ensemble | 0.07 | ML model prediction |
| fundamental_value | 0.06 | Fundamental value |
| liquidity_quality | 0.04 | Liquidity quality |
| macro_risk | 0.04 | Macro risk environment |
| export_momentum | 0.04 | Taiwan export momentum |
| us_manufacturing | 0.04 | US manufacturing PMI |

## REGIME_MULTIPLIERS

Three market regimes adjust factor weights multiplicatively:

| Factor | Bull | Bear | Sideways |
|--------|------|------|----------|
| foreign_flow | 1.0 | 1.3 | 1.0 |
| technical_signal | 1.1 | 0.8 | 1.3 |
| short_momentum | 1.3 | 0.5 | 0.8 |
| trust_flow | 1.0 | 1.2 | 1.0 |
| volume_anomaly | 1.1 | 0.9 | 1.2 |
| margin_sentiment | 0.8 | 1.5 | 1.0 |
| trend_momentum | 1.3 | 0.5 | 0.7 |
| volatility_regime | 0.7 | 1.5 | 1.2 |
| fundamental_value | 0.8 | 1.2 | 1.0 |
| liquidity_quality | 0.8 | 1.3 | 1.0 |
| macro_risk | 0.8 | 1.3 | 1.0 |

Key pattern: Bear amplifies defensive factors (margin_sentiment 1.5x, volatility_regime 1.5x), bull amplifies momentum (short_momentum 1.3x, trend_momentum 1.3x).

## FactorResult Dataclass

```python
@dataclass
class FactorResult:
    name: str              # e.g. "technical_signal"
    score: float           # 0.0-1.0 (0.5 = neutral)
    available: bool        # True = has real data
    freshness: float       # 0.0-1.0 (1.0 = today)
    components: dict = field(default_factory=dict)
    raw_value: float | None = None  # for IC tracking
```

## Confidence Calculation

```
confidence = (agreement*0.30 + strength*0.30 + coverage*0.25 + freshness*0.15) x risk_discount
```

- **agreement** (30%): max(bullish_count, bearish_count) / total_available
- **strength** (30%): abs(total_score - 0.5) * 2
- **coverage** (25%): sum of BASE_WEIGHTS for available factors
- **freshness** (15%): weighted average of factor freshness values

### Risk Discount (floor = 0.3)
- Annualized vol > 60%: x0.70; > 40%: x0.85
- Avg volume < 200 lots: x0.60; < 500 lots: x0.80
- Margin surge > 10% in 5 days: x0.85
- P/E > 80 or negative: x0.75; > 50: x0.85

## Missing Factor Weight Redistribution

`_compute_weights(factors, regime)`:
1. Filter to `available=True` factors only
2. Apply `BASE_WEIGHTS[name] * REGIME_MULTIPLIERS[regime][name]`
3. Normalize so weights sum to 1.0
4. Unavailable factors get weight 0; their share redistributes proportionally

## IC Tracking

After full market scan, each factor's score is saved per stock per day via `save_factor_ic_records()`. Only `available=True` factors are recorded. `effective_coverage` (sum of base weights for available factors) is also stored.
