---
name: LLM Narrative Agent
description: >
  The LLM-based sentiment extraction and narrative generation used in the
  unified analysis pipeline. The legacy multi-agent debate system exists
  in src/agents/ but is NOT used by the current v3.0 pipeline.
---

# LLM Narrative Agent

Source: `src/agents/narrative_agent.py`

## Current Architecture (v3.0)

The stock analysis pipeline uses **20-factor scoring** as the core engine.
LLM is used for two specific tasks only:

| Function | Model | Purpose |
|----------|-------|---------|
| `extract_sentiment` | Haiku | Fast sentiment scoring from news/posts |
| `generate_narrative` | Sonnet | Comprehensive analysis narrative for users |

Both are called from `api/services/stock_analysis_service.py` within the unified pipeline.

## Pipeline Flow (StockAnalysisService)

1. **Data collection** (parallel fetch)
2. **Feature extraction** (20 factors + HMM regime + ML)
3. **LLM sentiment extraction** (Haiku)
4. **Multi-factor scoring** (weighted sum with regime adjustment)
5. **LLM narrative generation** (Sonnet)
6. **Risk evaluation + position sizing**
7. **Storage** (MarketScanResult, PipelineResult, Prediction, TradeJournal)

Results are streamed to the frontend via SSE (`data: {json}\n\n`).

## Legacy Code (NOT in use)

The following exist in `src/agents/` but are **dead code** — not called by
the v3.0 unified pipeline:

- `base.py` — BaseAgent ABC, AgentRole(8 roles), AgentMessage, Signal, MarketContext, TradeDecision
- `orchestrator.py` — AgentOrchestrator, RuleEngine (ML 60% + Agent 40%)
- `technical_agent.py`, `sentiment_agent.py`, `fundamental_agent.py`, `quant_agent.py`
- `researcher_agent.py`, `trader_agent.py`, `risk_agent.py`
- `memory.py` — agent memory store
- `api/routers/agent.py` — legacy `/api/stocks/{id}/agent/analyze` endpoint

The DB schema still has `agent_scores` and `agent_decision` fields for
backward compatibility, but they are populated with unified pipeline results
(not actual multi-agent debate output).
