# Architecture Optimization Plan v5 — Deep Assessment Report

## Context

This report is a comprehensive architecture review of twstock-predictor, answering:
1. Is the architecture viable? Can it accurately predict Taiwan stock trends?
2. Does it need a full rewrite?
3. What are the highest-ROI optimization directions?

Based on: deep code review (market_service.py, stock_analysis_service.py, trainer.py, cpcv.py, ensemble.py, lstm_model.py, etc.) + 2024-2025 academic research comparison.

---

## 1. Overall Verdict

> **Architecture skeleton is correct, no rewrite needed. But 3 structural defects severely weaken predictive power. After fixes, the system can go from "slightly better than random" to "practically useful reference".**

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Architecture Design | **B+** | Unified pipeline + HMM regime + risk separation, clean design |
| Data Engineering | **B** | Multi-layer fallback + purging correct, but look-ahead bias risk |
| ML Pipeline | **B-** | Training rigorous, but predictions systematically compressed, core components unused |
| Factor System | **C+** | Rich factor design but manual weights, collinearity untreated, no OOS validation |
| Risk Control | **B+** | Kelly + ATR stop correct, but pipeline integration incomplete |
| Overall Viability | **B-** | Can provide directional reference, not suitable for direct auto-trading |

---

## 2. What's Worth Keeping (What You Got Right)

### 2.1 Unified Pipeline (6-phase)
`stock_analysis_service.py`'s `analyze_stock()` single entry + SSE streaming replaced the original 7-agent chaos. Correct simplification.

### 2.2 HMM Regime Detection
3-state GaussianHMM (bull/bear/sideways) with returns + volatility dual observations, plus MA trend override to prevent obvious misclassification. Academic research supports regime-adaptive strategies being more robust in extreme markets.

### 2.3 CPCV Overfitting Detection
Combinatorial Purged CV + PBO is industry best practice (2024 research reconfirms CPCV superior to walk-forward). Purging gap=15 days correctly blocks Triple Barrier label leakage.

### 2.4 Data Leakage Protection
- Feature selection restricted to training set (`trainer.py:211-219`)
- Purge gap = max_holding(10) + embargo(5) = 15 days
- Time-series split 60/20/20, no shuffling

### 2.5 Risk Control Independence
RiskManager's hard thresholds (20% single stock cap, 2% single trade risk, 15% drawdown circuit breaker) cannot be overridden by LLM or factors.

### 2.6 Institutional Flow Weight
Foreign investors (11%) + Investment trust (5%) + Institutional sync (4%) = **20%** on institutional flow. Academic research confirms: Taiwan stock foreign investor T86 fund flow has significant predictive power on next-day index returns (ScienceDirect 2019, PMC 2020).

---

## 3. Three Structural Defects (Core Issues)

### Defect 1: ML Predictions Systematically Compressed to Useless Signal

**Problem**: ML ensemble (LSTM + XGBoost) rigorously trained (100 epochs, CPCV, quality gate), but final result compressed to 5 discrete scores, then mixed in at 7% weight across 20 factors.

```python
# stock_analysis_service.py (old code)
signal_score = {
    "strong_buy": 0.9, "buy": 0.75,
    "hold": 0.5,
    "sell": 0.25, "strong_sell": 0.1,
}
return {stock_id: signal_score.get(result.signal, 0.5)}
```

**Impact calculation**: If ML gives strong_buy (0.9), other 19 factors average 0.5:
- ML marginal contribution = 0.07 x (0.9 - 0.5) = **0.028**
- Virtually undetectable in 0-1 score range

**Even worse**: `StackingEnsemble` (Ridge meta-learner) and `MetaLabeler` both implemented but **never enabled in production** (require `use_tft=True` / `use_meta_label=True`, both default False).

### Defect 2: 20 Factor Collinearity Untreated + Weights Unvalidated

**Collinearity issues**:
- **Institutional trio** (20% weight): foreign_flow / trust_flow / institutional_sync use highly overlapping underlying data (T86 buy/sell)
- **Momentum twins**: short_momentum (1-5d returns) + trend_momentum (20-60d returns) both based on close price
- **Volatility overlap**: volume_anomaly + volatility_regime both depend on volume/volatility
- **Global overlap**: global_context (SOX/TSM) + us_manufacturing (XLI/SPY) + taiwan_etf_momentum (EWT)

ML features have `_remove_collinear()` removing corr > 0.95, but **20 factors have zero decorrelation processing**.

**Weight issues**:
- `BASE_WEIGHTS` (20 values) all manually set
- `REGIME_MULTIPLIERS` (60 multipliers) all manually set
- `SCALE_*` constants (7) manually set
- **IC adaptive** only has 3 discrete tiers — too coarse

**Total 87+ manual parameters**, all without out-of-sample validation.

### Defect 3: Pipeline Phase Order Wrong + LLM No Fallback

**Order issue**:
```
Phase 3: Scoring (20-factor scoring)
Phase 4: Narrative (Opus generates analysis + position suggestion)  <- doesn't know risk result
Phase 5: Risk Control (ATR stop + circuit breaker)                  <- may reject Phase 4's suggestion
Phase 6: Save
```
If Phase 5 risk rejects the position, Phase 4's position_suggestion still shows to user.

**LLM fallback missing**: Opus timeout -> narrative empty -> user sees no analysis. Even with complete factor scores and ML predictions, nothing is displayed.

---

## 4. Other Important Issues

### 4.1 PBO Calculation Math Issue
`cpcv.py` PBO only uses 1 real strategy + 1 random baseline. Correct PBO needs 2+ strategies in performance matrix to detect "IS best = OOS worst". Current PBO is essentially just a "better than random?" test.

### 4.2 CPCV Asymmetry
CPCV only validates XGBoost (`trainer.py:513-514`), LSTM never rejected by PBO gate. If LSTM overfits (memorizes recent patterns), it won't be detected.

### 4.3 Look-ahead Bias Risk
- **Quarterly gross margin** (margin_quality factor): yfinance `quarterly_income_stmt` has no filing date check
- **Monthly revenue** (revenue_momentum): Taiwan regulation requires disclosure by 10th of following month, but system doesn't confirm as-of date
- **STOCK_SECTOR** only has ~80 stocks, remaining 1500+ classified as "other", sector_rotation factor ineffective for most stocks

### 4.4 Global Data Fallback to Zero
When yfinance fails, SOX/TSM return zero -> factor calculates 0.5 neutral score. During US market crash + API failure due to traffic, system **completely ignores global risk signal**.

### 4.5 ATR Stop Not Volatility-Adaptive
Fixed 2.5x ATR multiplier. High volatility (VIX > 30) should widen to 3.5x to avoid noise triggers; low volatility (VIX < 15) should tighten to 2.0x.

### 4.6 Max Drawdown Circuit Breaker Not Wired
`max_drawdown_limit: 0.15` exists but not called in `analyze_stock()`.

---

## 5. Academic Research Comparison

| Current Approach | Academic Best Practice | Gap |
|-----------------|----------------------|-----|
| LSTM + XGBoost stacking | 2025: ensemble not always better than tuned XGBoost; **dual-task (regression + classification)** more robust in extreme markets | Can add classification head |
| Manual 20-factor weights | Factor-GAN (2024): GAN learns factor weights; IC-driven weights is baseline | **Need IC-driven or CV-driven weights** |
| 3-state HMM regime | Adaptive CPCV (2024): dynamic CV strategy; Regime-adaptive hybrid is latest trend | HMM correct, but regime multipliers need validation |
| CPCV + PBO | 2024 research confirms CPCV superior to walk-forward, but needs 2+ strategies | PBO needs fix |
| LLM sentiment extraction | Sentiment-Aware Transformer (2025): LLM-generated alpha + transformer fusion | Haiku direction correct, but contrarian_flag unused |
| Institutional flow 20% weight | Taiwan foreign investor predictive power has academic support (ScienceDirect 2019, PMC 2020) | **Direction correct, weight reasonable** |

---

## 6. Optimization Items (Sorted by ROI)

### Tier 1: High Impact, Medium Cost (Immediate)

#### 1. ML Factor Integration Refactor
- **Goal**: ML prediction from "compressed to 7% discrete signal" to "15% continuous factor"
- **Approach**:
  - Remove `signal_score` mapping
  - Use `result.predicted_returns` sigmoid transform as continuous score
  - Boost `ml_ensemble` weight to 15%
- **Status**: COMPLETED

#### 2. Factor Decorrelation
- **Goal**: From 20 nominal factors to 12-15 effective independent factors
- **Approach**:
  - Merge institutional trio into 1 `composite_institutional` (foreign 50% + trust 30% + sync 20%)
  - Merge short_momentum + trend_momentum into `multi_scale_momentum`
  - Merge global_context + us_manufacturing + taiwan_etf_momentum into `global_macro`
- **Status**: COMPLETED (20 -> 15 factors)

#### 3. Pipeline Order Fix + LLM Fallback
- **Goal**: Risk result must be determined before narrative
- **Approach**:
  - Swap Phase 4 (Narrative) and Phase 5 (Risk)
  - Pass risk result as input to Opus prompt
  - Template-based fallback when Opus times out
- **Status**: COMPLETED

### Tier 2: Medium Impact, Low Cost

#### 4. IC-driven Weight Calibration
- Replace 3-level discrete IC adjustment with continuous sigmoid: `_ic_sigmoid(icir, ic_mean)`
- Legacy factor name mapping for backward compatibility
- **Status**: COMPLETED

#### 5. CPCV Fix
- PBO calculation with 4 strategy variants (original + perturbed + inverted + random)
- LSTM soft CPCV direction check (`_lstm_cpcv_direction_check`, warns if <0.48)
- **Status**: COMPLETED

#### 6. Look-ahead Bias Fix
- margin_quality: filing date delay (quarter end + 45 days)
- revenue_momentum: as-of date validation (10th of following month)
- **Status**: COMPLETED

### Tier 3: Medium Impact, Higher Cost

#### 7. Volatility-Adaptive ATR
- ATR multiplier based on VIX: `multiplier = 2.0 + (vix - 15) * 0.05`, clamped [1.5, 4.0]
- **Status**: COMPLETED

#### 8. Dual-task ML Architecture
- LSTM regression + classification head (3-class: up/neutral/down)
- cls_weight = 0.3, shared LSTM backbone
- Backward-compatible checkpoint loading
- **Status**: COMPLETED

#### 9. Expand STOCK_SECTOR
- Currently only ~80 stocks, need full TWSE industry classification
- Use TWSE API to auto-fetch complete sector mapping
- **Status**: NOT STARTED

---

## 7. Completion Status

| # | Item | Status | Files Modified |
|---|------|--------|---------------|
| 1 | ML Factor Integration | DONE | stock_analysis_service.py, market_service.py |
| 2 | Factor Decorrelation (20->15) | DONE | market_service.py |
| 3 | Pipeline Order + LLM Fallback | DONE | stock_analysis_service.py, narrative_agent.py |
| 4 | Continuous IC Calibration | DONE | market_service.py |
| 5 | CPCV PBO Fix + LSTM Check | DONE | cpcv.py, trainer.py |
| 6 | Look-ahead Bias Fix | DONE | market_service.py |
| 7 | VIX-adaptive ATR | DONE | stock_analysis_service.py |
| 8 | Dual-task LSTM | DONE | lstm_model.py |
| 9 | Expand STOCK_SECTOR | TODO | market_service.py |

**8 of 9 items completed. 224 tests passing.**

---

## 8. Validation Methods

After fixes, validate improvements with:

1. **Factor IC backtest**: Calculate rolling 60d Spearman IC per factor, confirm IC > 0.02 and stable
2. **Direction accuracy**: Test on 2024-2025 holdout data, compare before/after
3. **Signal distribution**: Confirm total_score distribution no longer crowded at 0.45-0.55
4. **Backtest P&L**: Use `scripts/ab_backtest.py` to compare old vs new scoring Sharpe ratio
5. **Unit tests**: `pytest tests/ -v` — all 224 tests passing

---

## References

- [Fama-French Six Factors on Taiwan Stock Returns (SCIRP 2024)](https://www.scirp.org/journal/paperinformation?paperid=132956)
- [Foreign investors' trading behavior in Taiwan (ScienceDirect 2019)](https://www.sciencedirect.com/science/article/abs/pii/S1042444X19300490)
- [Structural changes in foreign investors' impact on Taiwan (PMC 2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7148904/)
- [Backtest overfitting: CPCV comparison (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110)
- [Stacked Heterogeneous Ensemble for Stock Prediction (MDPI 2025)](https://www.mdpi.com/2227-7102/13/4/201)
- [Hybrid AI-Driven Trading: Regime-Adaptive Equity Strategies (arxiv 2026)](https://arxiv.org/html/2601.19504v1)
- [Deep Learning Multi-Factor Approach for Short-Term Equity (arxiv 2025)](https://arxiv.org/html/2508.14656v1)
- [Sentiment-Aware Stock Prediction with Transformer + LLM Alpha (arxiv 2025)](https://arxiv.org/html/2508.04975v1)
- [Factor-GAN: Enhancing Factor Investment (PLOS One 2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0306094)
- [Probability of Backtest Overfitting (Bailey & Borwein)](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
