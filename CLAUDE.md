# twstock-predictor

## Quick Reference
- **Streamlit**: `.venv\Scripts\streamlit run app\main.py --server.port 8501`
- **Tests**: `.venv\Scripts\python -m pytest tests\ -v`
- **Python**: 3.13 via uv, venv at `.venv\`

## Project Structure
- `app/` — Streamlit frontend (main.py + 3 pages + components)
- `src/` — Core logic (agents, analysis, backtest, data, db, models, monitoring, pipeline, portfolio, risk, utils)
- `tests/` — 85 tests (pytest)
- `models/` — Trained model artifacts (.json, .pt)
- `data/` — SQLite DB (twstock.db)

## Coding Conventions
- SQLAlchemy 2.x: use `session.get()` not `query.get()`
- `ta` library: uses `n=` parameter style (not `window=`)
- XGBoost: `sample_weight` in `.fit()` call
- DataFrame: use `col in df.columns` (no `.get()`)
- TFT: DataFrame interface via trainer bridge
- Streamlit: dark theme, progressive disclosure (expanders for advanced content)

## Key Architecture
- Triple Barrier labels (ATR-based)
- PurgedTimeSeriesSplit + CPCV for cross-validation
- StackingEnsemble (LSTM + XGBoost) with meta-labeling
- 3-state HMM regime detection
- Multi-agent debate system (4 analysts + researcher + trader + risk)
- Risk: ATR Trailing Stop + circuit breaker
- CVXPY portfolio optimizer with analytic fallback
