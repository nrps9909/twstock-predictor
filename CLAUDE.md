# twstock-predictor

## Quick Reference
- **Frontend**: `cd web && npm run dev`
- **Backend**: `.venv\Scripts\uvicorn src.api:app --reload`
- **Tests**: `.venv\Scripts\python -m pytest tests\ -v`
- **Lint**: `.venv\Scripts\python -m ruff check --fix src/ api/ tests/`
- **Format**: `.venv\Scripts\python -m ruff format src/ api/ tests/`
- **Python**: 3.13 via uv, venv at `.venv\`

## Project Structure
- `web/` — Next.js frontend (App Router + Tailwind + Plotly charts)
- `src/` — Core logic (agents, analysis, backtest, data, db, models, monitoring, pipeline, risk, utils)
- `tests/` — 85 tests (pytest)
- `models/` — Trained model artifacts (.json, .pt)
- `data/` — SQLite DB (twstock.db)

## Coding Conventions
- SQLAlchemy 2.x: use `session.get()` not `query.get()`
- `ta` library: uses `n=` parameter style (not `window=`)
- XGBoost: `sample_weight` in `.fit()` call
- DataFrame: use `col in df.columns` (no `.get()`)
- TFT: DataFrame interface via trainer bridge

## API Conventions
- Route prefix: `/api/v1/` for all endpoints
- SSE format: `data: {json}\n\n` — frontend connects directly via `SSE_BASE`
- All numeric responses must sanitize NaN → `null` before JSON serialization
- Frontend `SSE_BASE` points directly to backend (no proxy)

## LLM Integration
- Haiku: sentiment extraction from news/posts
- Sonnet: analysis narrative generation for users
- Both called from `StockAnalysisService`, NOT via multi-agent debate
- Legacy agent system in `src/agents/` is dead code (not used by v3.0 pipeline)
- See skill: `agent-system` for details

## Risk Rules (NEVER OVERRIDE)
- max_position: 20% per stock | max_portfolio_risk: 2% per trade
- max_drawdown: 15% circuit breaker | ATR trailing stop: 2.5x (only moves up)
- Kelly: 1/4 fraction | Regime transition: reduce_50% or close_all
- See skill: `risk-management` for implementation details

## Data Layer
- Priority: FinMind → yfinance → TWSE proxy
- Rate limits: FinMind 2/s, TWSE 1/s
- Volume: always in lots (張 = 1000 shares). FinMind `/1000`, TWSE `//1000`
- See skill: `data-fallback` for field mappings and retry logic

## ML Pipeline
- StackingEnsemble (LSTM + XGBoost) with Ridge meta-learner
- 3 quality gates before model save: direction_acc>52%, MSE<naive, PBO<0.6
- See skill: `ml-pipeline` for 43 features, Triple Barrier, and CPCV details

## 20-Factor Scoring
- 20 factors (short 39% / mid 32% / long 29%) with regime-adaptive weights
- Confidence = (agreement + strength + coverage + freshness) x risk_discount
- See skill: `factor-scoring` for full weight table and redistribution logic

## Key Architecture
- Triple Barrier labels (ATR-based)
- PurgedTimeSeriesSplit + CPCV for cross-validation
- StackingEnsemble (LSTM + XGBoost) with meta-labeling
- 3-state HMM regime detection
- LLM narrative (Haiku sentiment + Sonnet narrative, legacy multi-agent is dead code)
- Risk: ATR Trailing Stop + circuit breaker
