---
name: Multi-Agent Analysis System
description: >
  The multi-agent debate system: BaseAgent ABC, AgentRole (8 roles),
  AgentMessage, Signal, MarketContext, TradeDecision, 6-stage pipeline,
  RuleEngine (ML 60% + Agent 40%), and LLM dual-layer architecture.
---

# Multi-Agent Analysis System

Sources: `src/agents/base.py`, `src/agents/orchestrator.py`

## AgentRole (8 roles)

```python
class AgentRole(str, Enum):
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"
    QUANT = "quant"
    RESEARCHER = "researcher"
    TRADER = "trader"
    RISK = "risk"
    ORCHESTRATOR = "orchestrator"
```

## Signal (5 levels)

```python
class Signal(str, Enum):
    STRONG_BUY = "strong_buy"    # score  1.0
    BUY = "buy"                  # score  0.5
    HOLD = "hold"                # score  0.0
    SELL = "sell"                # score -0.5
    STRONG_SELL = "strong_sell"  # score -1.0
```

## Core Dataclasses

### AgentMessage
```python
@dataclass
class AgentMessage:
    sender: AgentRole
    content: dict[str, Any]
    signal: Signal | None = None
    confidence: float = 0.0      # 0.0 ~ 1.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### MarketContext (input)
```python
@dataclass
class MarketContext:
    stock_id: str
    current_price: float
    date: str                    # YYYY-MM-DD
    technical_summary: dict = field(default_factory=dict)
    sentiment_summary: dict = field(default_factory=dict)
    fundamental_summary: dict = field(default_factory=dict)
    model_predictions: dict = field(default_factory=dict)
    position: dict = field(default_factory=dict)
```

### TradeDecision (output)
```python
@dataclass
class TradeDecision:
    stock_id: str
    action: str                  # "buy" | "sell" | "hold"
    quantity: int = 0
    price_limit: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size: float = 0.0   # 0.0 ~ 1.0
    confidence: float = 0.0
    reasoning: str = ""
    approved_by_risk: bool = False
    risk_notes: str = ""
    analyst_reports: list[AgentMessage] = field(default_factory=list)
    researcher_report: AgentMessage | None = None
```

## BaseAgent ABC

```python
class BaseAgent(ABC):
    def __init__(self, role: AgentRole): ...

    @abstractmethod
    async def analyze(self, context: MarketContext) -> AgentMessage: ...
```

## 6-Stage Pipeline

1. **Data collection** (parallel fetch)
2. **Feature extraction** (20 factors + HMM regime + ML + LLM sentiment)
3. **Multi-factor scoring**
4. **LLM narrative generation**
5. **Risk control + position sizing**
6. **Storage + alerts**

## RuleEngine (ML 60% + Agent 40%)

```
When ML has valid prediction (signal != hold, confidence > 0.1):
    ML weight = 0.6, Agent weight = 0.4
When ML unavailable:
    Agent weight = 1.0

combined = ml_w * (ml_signal_score * ml_confidence)
         + agent_w * (agent_signal_score * agent_confidence)

Market state scaling: bear=0.5, sideways=0.7, bull=1.0
Buy threshold: > 0.15, Sell threshold: < -0.15
```

## LLM Dual-Layer Architecture

| Layer | Model | Purpose |
|-------|-------|---------|
| Sentiment extraction | Haiku | Fast sentiment scoring from news/posts |
| Narrative generation | Sonnet | Comprehensive analysis narrative for users |

The orchestrator delegates to `StockAnalysisService` which handles both LLM calls within the unified pipeline, streaming results via SSE.
