"""SQLAlchemy ORM 模型定義

Tables:
- StockPrice: 日K線資料
- SentimentRecord: 社群情緒紀錄
- Prediction: 預測紀錄
- BacktestResult: 回測結果 (Tier 2)
- AgentMemory: Agent 記憶體 (Tier 3)
- TradeJournal: 交易日誌 (Tier 3)
- MarketScanResult: 全市場掃描結果
- PipelineResult: 每日自動深度分析結果
- FactorICRecord: 因子 IC 追蹤紀錄
"""

from datetime import date, datetime

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class StockPrice(Base):
    """日K線資料"""

    __tablename__ = "stock_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)  # 成交量（張）

    # 籌碼面
    foreign_buy_sell = Column(Float)  # 外資買賣超
    trust_buy_sell = Column(Float)  # 投信買賣超
    dealer_buy_sell = Column(Float)  # 自營商買賣超

    # 融資融券
    margin_balance = Column(Float)  # 融資餘額
    short_balance = Column(Float)  # 融券餘額

    # As-of 時間戳（Point-in-time correctness）
    as_of_date = Column(Date)  # 資料實際可取得的日期

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uix_stock_date"),
    )


class SentimentRecord(Base):
    """社群情緒紀錄"""

    __tablename__ = "sentiment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    source = Column(String(50), nullable=False)  # ptt, dcard, cnyes, yahoo
    title = Column(Text)
    content_summary = Column(Text)
    sentiment_label = Column(String(10))  # bullish, bearish, neutral
    sentiment_score = Column(Float)  # -1.0 ~ 1.0
    keywords = Column(Text)  # JSON array
    engagement = Column(Integer, default=0)  # 互動數（推文數/回覆數）
    url = Column(Text)

    # As-of 時間戳（Point-in-time correctness）
    as_of_date = Column(Date)  # 資料實際可取得的日期

    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    """預測紀錄"""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(String(10), nullable=False, index=True)
    prediction_date = Column(Date, nullable=False)  # 預測產生日期
    target_date = Column(Date, nullable=False)  # 預測目標日期
    predicted_price = Column(Float)
    predicted_return = Column(Float)
    confidence_lower = Column(Float)  # 95% CI 下界
    confidence_upper = Column(Float)  # 95% CI 上界
    actual_price = Column(Float)  # 事後填入
    model_type = Column(String(20))  # lstm, xgboost, ensemble, tft, stacking
    signal = Column(String(10))  # buy, sell, hold

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "stock_id", "prediction_date", "target_date", "model_type",
            name="uix_prediction",
        ),
    )


# ── Tier 2: 回測結果 ──────────────────────────────────


class BacktestResult(Base):
    """回測結果紀錄"""

    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(String(10), nullable=False, index=True)
    strategy_name = Column(String(50), nullable=False)  # e.g. "lstm_ensemble", "tft_agent"
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)

    # 績效指標
    total_return = Column(Float)
    annualized_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    direction_accuracy = Column(Float)
    total_trades = Column(Integer)
    avg_mse = Column(Float)

    # 詳細資料（JSON）
    monthly_returns = Column(JSON)  # list[float]
    parameters = Column(JSON)  # 策略參數

    created_at = Column(DateTime, default=datetime.utcnow)


# ── Tier 3: Agent 記憶體 + 交易日誌 ──────────────────


class AgentMemory(Base):
    """Agent 記憶體系統

    三層記憶：
    - short_term: 最近 5 個交易日的市場資料 + 決策
    - long_term: 歷史模式庫
    - episodic: 情境記憶（特定事件相關）
    """

    __tablename__ = "agent_memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(String(10), index=True)  # 可為 NULL（通用記憶）
    memory_type = Column(String(20), nullable=False)  # short_term, long_term, episodic
    category = Column(String(50))  # e.g. "market_pattern", "decision", "event"
    content = Column(Text, nullable=False)  # 記憶內容（結構化 JSON 或文本）
    embedding = Column(Text)  # 向量 embedding（JSON array，用於相似度搜尋）
    relevance_score = Column(Float, default=1.0)  # 記憶重要性分數
    access_count = Column(Integer, default=0)  # 存取次數
    last_accessed = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)  # 短期記憶可設過期時間


class FeatureImportanceRecord(Base):
    """特徵重要性紀錄 — 追蹤每次訓練的特徵重要性"""

    __tablename__ = "feature_importance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(String(10), nullable=False, index=True)
    run_date = Column(Date, nullable=False, index=True)
    feature_name = Column(String(100), nullable=False)
    importance_score = Column(Float, nullable=False)
    method = Column(String(20), nullable=False)  # mutual_info, shap
    rank = Column(Integer)  # 1-based rank

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "stock_id", "run_date", "feature_name", "method",
            name="uix_feature_importance",
        ),
    )


class DelistedStock(Base):
    """已下市股票紀錄 — 存活者偏誤修正"""

    __tablename__ = "delisted_stocks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(String(10), nullable=False, unique=True, index=True)
    name = Column(String(100))
    delist_date = Column(Date, nullable=False)
    reason = Column(String(200))  # 下市原因
    merged_into = Column(String(10))  # 合併對象（若為合併下市）
    last_price = Column(Float)  # 最後交易價格

    created_at = Column(DateTime, default=datetime.utcnow)


class TradeJournal(Base):
    """交易日誌 — 記錄每筆交易決策的完整脈絡

    包含：Agent 推理過程、市場狀態、實際結果、事後檢討
    """

    __tablename__ = "trade_journal"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(String(10), nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)

    # 決策
    action = Column(String(10), nullable=False)  # buy, sell, hold
    quantity = Column(Integer)  # 股數
    price = Column(Float)  # 執行價格
    position_size = Column(Float)  # 倉位比例 (0.0 ~ 1.0)

    # Agent 推理
    technical_analysis = Column(JSON)  # 技術面 Agent 輸出
    sentiment_analysis = Column(JSON)  # 情緒面 Agent 輸出
    fundamental_analysis = Column(JSON)  # 基本面 Agent 輸出
    quant_analysis = Column(JSON)  # 量化 Agent 輸出
    researcher_debate = Column(JSON)  # 研究員多空辯論
    trader_reasoning = Column(Text)  # 交易員決策理由
    risk_assessment = Column(JSON)  # 風控 Agent 評估

    # 結果（事後填入）
    exit_date = Column(Date)
    exit_price = Column(Float)
    pnl = Column(Float)  # 損益金額
    pnl_pct = Column(Float)  # 損益百分比
    review_notes = Column(Text)  # 事後檢討

    # 市場快照
    market_snapshot = Column(JSON)  # 交易當下的市場狀態

    created_at = Column(DateTime, default=datetime.utcnow)


class Alert(Base):
    """警報紀錄 — 掃描後偵測到的重大訊號變動"""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_date = Column(Date, nullable=False, index=True)
    stock_id = Column(String(10), nullable=False, index=True)
    stock_name = Column(String(100))
    alert_type = Column(String(30), nullable=False)
    # "signal_change" | "strong_signal" | "institutional_surge" | "sync_buy" | "score_jump"
    severity = Column(String(10), nullable=False)  # "high" | "medium" | "low"
    title = Column(String(200), nullable=False)
    detail = Column(Text)
    current_signal = Column(String(20))
    previous_signal = Column(String(20))
    current_score = Column(Float)
    previous_score = Column(Float)
    is_read = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class FactorICRecord(Base):
    """因子 IC 追蹤紀錄 — 存每日因子分數 + 遠期報酬，計算因子有效性"""

    __tablename__ = "factor_ic_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    record_date = Column(Date, nullable=False, index=True)
    stock_id = Column(String(10), nullable=False, index=True)
    factor_name = Column(String(30), nullable=False)
    factor_score = Column(Float, nullable=False)
    forward_return_5d = Column(Float)   # 5 交易日後回填
    forward_return_20d = Column(Float)  # 20 交易日後回填

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("record_date", "stock_id", "factor_name",
                         name="uix_factor_ic_record"),
    )


class MarketScanResult(Base):
    """全市場掃描結果"""

    __tablename__ = "market_scans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_date = Column(Date, nullable=False, index=True)
    stock_id = Column(String(10), nullable=False, index=True)
    stock_name = Column(String(100))
    current_price = Column(Float)
    price_change_pct = Column(Float)

    # Signals
    signal = Column(String(20))  # buy, sell, hold, strong_buy, strong_sell
    confidence = Column(Float)
    total_score = Column(Float)

    # Sub-scores (original 5)
    technical_score = Column(Float)
    fundamental_score = Column(Float)
    sentiment_score = Column(Float)
    ml_score = Column(Float)
    momentum_score = Column(Float)

    # Sub-scores (new 4)
    institutional_flow_score = Column(Float)
    margin_retail_score = Column(Float)
    volatility_score = Column(Float)
    liquidity_score = Column(Float)
    value_quality_score = Column(Float)

    # Institutional flows (5-day net, in lots)
    foreign_net_5d = Column(Float)
    trust_net_5d = Column(Float)
    dealer_net_5d = Column(Float)

    # Ranking + reasoning
    ranking = Column(Integer)
    reasoning = Column(Text)

    # Score transparency
    score_coverage = Column(JSON, nullable=True)  # {"technical": true, "ml": false, ...}
    effective_coverage = Column(Float, nullable=True)  # 0.0~1.0

    # Confidence breakdown
    confidence_agreement = Column(Float)
    confidence_strength = Column(Float)
    confidence_coverage = Column(Float)
    confidence_freshness = Column(Float)
    risk_discount = Column(Float)

    # Regime
    market_regime = Column(String(20))
    factor_details = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("scan_date", "stock_id", name="uix_scan_date_stock"),
    )


class PipelineResult(Base):
    """每日自動深度分析結果"""

    __tablename__ = "pipeline_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(String(10), nullable=False, index=True)
    analysis_date = Column(Date, nullable=False, index=True)
    signal = Column(String(20))           # buy/sell/hold
    confidence = Column(Float)
    predicted_price = Column(Float)
    reasoning = Column(Text)              # Agent 決策理由
    agent_scores = Column(JSON)           # 各 agent 分數
    sentiment_summary = Column(Text)      # 情緒摘要
    news_summary = Column(Text)           # 新聞摘要
    technical_data = Column(JSON)         # K 線 + 技術指標
    institutional_data = Column(JSON)     # 法人買賣超
    risk_approved = Column(Integer)       # 風控通過
    pipeline_version = Column(String(20)) # 追蹤版本
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("stock_id", "analysis_date", name="uix_pipeline_stock_date"),
    )
