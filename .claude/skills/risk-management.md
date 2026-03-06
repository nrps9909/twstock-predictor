---
name: Risk Management Rules
description: >
  Hard risk rules, ATR trailing stop, position sizing, Kelly criterion,
  and regime transition logic. These rules must NEVER be overridden.
---

# Risk Management Rules

Source: `src/risk/manager.py`

## Hard Rules (NEVER OVERRIDE)

| Rule | Value | Description |
|------|-------|-------------|
| max_position_pct | 20% | Maximum single position as % of portfolio |
| max_portfolio_risk | 2% | Maximum loss per trade as % of portfolio |
| max_drawdown_limit | 15% | Circuit breaker — halt all trading |
| atr_trailing_stop_multiplier | 2.5 | ATR multiplier for trailing stop |

## TrailingStopState

```python
@dataclass
class TrailingStopState:
    entry_price: float
    highest_price: float      # highest during hold period
    atr: float
    multiplier: float         # 2.5
    current_stop: float       # current trailing stop price
    triggered: bool = False
```

### Key behaviors:
- **Only moves up, never down**: `new_stop = highest_price - (multiplier * atr)`, only applied if `new_stop > current_stop`
- **Trigger check**: `current_low <= current_stop` -> triggered = True

## Position Sizing (`position_size_by_risk`)

```
risk_per_share = abs(entry_price - stop_loss_price)
max_loss = portfolio_value * max_portfolio_risk (2%)
max_shares = max_loss / risk_per_share
lots = floor(max_shares / 1000)  # 1 lot = 1000 shares in TWSE
```

Capped by `max_position_pct` (20%):
```
max_lots_by_position = floor(portfolio_value * 0.20 / entry_price / 1000)
final = min(lots, max_lots_by_position) * 1000
```

## Kelly Criterion

Uses **1/4 Kelly** (fraction = 0.25) for conservative sizing:
```
b = avg_win / avg_loss
kelly = (b * win_rate - (1 - win_rate)) / b
position = max(0, min(kelly * 0.25, max_position_pct))
```

## Regime Transition Check

Monitors HMM regime transitions and generates orders:

| Transition Action | Effect |
|-------------------|--------|
| `reduce_50%` | Reduce all positions by 50% |
| `close_all` | Close all positions entirely |
| `no_action` | No change |

Example: bull -> bear triggers `reduce_50%` on all holdings.
