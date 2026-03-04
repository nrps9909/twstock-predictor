// TypeScript 介面定義

export interface StockInfo {
  stock_id: string;
  name: string;
}

export interface StockPrice {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
}

export interface StockStatus {
  stock_id: string;
  name: string;
  has_model: boolean;
  has_data: boolean;
  data_count: number;
  latest_date: string | null;
  model_files: string[];
}

export interface TechnicalSignal {
  signal: string;
  reason: string;
}

export interface TechnicalSummary {
  signal: string;
  score: number;
  raw_score: number;
  max_score: number;
}

export interface TechnicalSignals {
  kd?: TechnicalSignal;
  rsi?: TechnicalSignal;
  macd?: TechnicalSignal;
  bias?: TechnicalSignal;
  bb?: TechnicalSignal;
  summary?: TechnicalSummary;
  [key: string]: TechnicalSignal | TechnicalSummary | undefined;
}

export interface TechnicalResult {
  signals: TechnicalSignals;
  indicators: Record<string, number | null>;
  latest_price: number | null;
  chart_data: Array<Record<string, number | string | null>>;
}

export interface SentimentSummary {
  total_records: number;
  avg_score: number;
  bullish_ratio: number;
  bearish_ratio: number;
  neutral_ratio: number;
  latest_date: string | null;
  by_source: Record<string, { count: number; avg_score: number }>;
}

export interface PredictionResult {
  predicted_returns: number[];
  predicted_prices: number[];
  confidence_lower: number[];
  confidence_upper: number[];
  signal: string;
  signal_strength: number;
  lstm_weight: number;
  xgb_weight: number;
  market_state: {
    state_name: string;
    probabilities: number[];
  } | null;
}

export interface AnalystReport {
  role: string;
  signal: string | null;
  confidence: number;
  reasoning: string;
}

export interface AgentDecision {
  action: string;
  confidence: number;
  position_size: number;
  reasoning: string;
  approved: boolean;
  risk_notes: string;
  analyst_reports: AnalystReport[];
  researcher: AnalystReport | null;
}

export interface PipelineEvent {
  step: string;
  status: "running" | "done" | "error" | "skipped";
  progress: number;
  message: string;
  data?: Record<string, unknown>;
}

export interface PipelineResult {
  stock_id: string;
  stock_name: string;
  current_price: number;
  signal: string;
  confidence: number;
  target_price: number;
  predicted_change: number;
  reasoning: string;
  technical: {
    signals: TechnicalSignals;
    indicators: Record<string, number | null>;
    current_price: number;
  };
  sentiment: SentimentSummary;
  prediction: PredictionResult | null;
  agent: AgentDecision | null;
}

// Agent sub-step events (emitted during step 8)
export type AgentSubstep =
  | "analysts_start"
  | "analyst_done"
  | "debate_start"
  | "debate_round"
  | "debate_synthesis"
  | "rule_engine"
  | "risk_check";

export interface AgentSubEvent {
  substep: AgentSubstep;
  // analyst_done
  role?: string;
  signal?: string;
  confidence?: number;
  error?: string;
  // debate_round
  round?: number;
  bull_args?: string[];
  bear_args?: string[];
  // analysts_start
  count?: number;
  // rule_engine
  action?: string;
  // risk_check
  approved?: boolean;
  notes?: string;
}

export interface PredictionRecord {
  id: number;
  stock_id: string;
  stock_name: string;
  prediction_date: string;
  signal: string;
  confidence: number;
  predicted_price: number;
  actual_price: number | null;
  reasoning: string;
  agent_action: string;
  agent_approved: boolean;
  analyst_reports: Record<string, unknown> | null;
  market_snapshot: Record<string, unknown> | null;
}

// ── Market Scan Types ─────────────────────────────

export interface StockScanResult {
  stock_id: string;
  stock_name: string;
  current_price: number;
  price_change_pct: number;
  signal: string;
  confidence: number;
  total_score: number;
  // Original 5 sub-scores (backward compat)
  technical_score: number;
  fundamental_score: number;
  sentiment_score: number;
  ml_score: number;
  momentum_score?: number;
  // New 5 sub-scores
  institutional_flow_score?: number;
  margin_retail_score?: number;
  volatility_score?: number;
  liquidity_score?: number;
  value_quality_score?: number;
  // Institutional flows
  foreign_net_5d: number;
  trust_net_5d: number;
  dealer_net_5d?: number;
  ranking: number;
  reasoning: string;
  sparkline?: number[];
  score_coverage?: Record<string, boolean>;
  effective_coverage?: number;
  // Confidence breakdown
  confidence_agreement?: number;
  confidence_strength?: number;
  confidence_coverage?: number;
  confidence_freshness?: number;
  risk_discount?: number;
  // Regime + factor details
  market_regime?: string;
  factor_details?: Record<string, {
    score: number;
    available: boolean;
    freshness: number;
    weight: number;
    components: Record<string, number>;
  }>;
}

export interface FactorICData {
  factor: string;
  ic_mean: number;
  ic_std: number;
  icir: number;
  ic_series: Array<{ date: string; ic: number }>;
}

export interface MarketOverview {
  scan_date: string | null;
  stocks: StockScanResult[];
  buy_recommendations: StockScanResult[];
  sell_recommendations: StockScanResult[];
}

export interface InstitutionalEntry {
  stock_id: string;
  stock_name: string;
  net_buy?: number;
  net_sell?: number;
  consecutive_days?: number;
  foreign_net?: number;
  trust_net?: number;
}

export interface MarketIntel {
  global_news: { title: string; source: string; date: string; url?: string }[];
  tw_news: { title: string; source: string; date: string; url?: string }[];
  trust_top_buy: InstitutionalEntry[];
  trust_top_sell: InstitutionalEntry[];
  foreign_top_buy: InstitutionalEntry[];
  sync_buy: InstitutionalEntry[];
  institutional_total: {
    date: string;
    foreign: number;
    trust: number;
    dealer: number;
    total: number;
  };
}

// ── Deep Pipeline Result (auto-analysis) ─────────

export interface DeepPipelineResult {
  stock_id: string;
  analysis_date: string;
  signal: string;
  confidence: number;
  predicted_price: number | null;
  reasoning: string;
  agent_scores: Record<string, { signal: string; confidence: number }> | null;
  sentiment_summary: string | null;
  news_summary: string | null;
  technical_data: Record<string, unknown> | null;
  institutional_data: Record<string, unknown> | null;
  risk_approved: boolean | null;
  pipeline_version: string | null;
  created_at: string | null;
}

// ── Alert Types ─────────────────────────────────

export interface Alert {
  id: number;
  alert_date: string;
  stock_id: string;
  stock_name: string;
  alert_type: string;
  severity: "high" | "medium" | "low";
  title: string;
  detail: string;
  current_signal?: string;
  previous_signal?: string;
  current_score?: number;
  previous_score?: number;
  is_read: boolean;
  created_at: string;
}

export interface MarketScanEvent {
  step: string;
  status: "running" | "done" | "error";
  progress: number;
  message: string;
  data?: Record<string, unknown>;
}

export type PipelineStep =
  | "check_data"
  | "fetch_data"
  | "technical"
  | "sentiment"
  | "check_model"
  | "train_model"
  | "predict"
  | "agent"
  | "synthesize";

export const PIPELINE_STEPS: { key: PipelineStep; label: string }[] = [
  { key: "check_data", label: "檢查資料" },
  { key: "fetch_data", label: "抓取資料" },
  { key: "technical", label: "技術分析" },
  { key: "sentiment", label: "情緒分析" },
  { key: "check_model", label: "檢查模型" },
  { key: "train_model", label: "訓練模型" },
  { key: "predict", label: "ML 預測" },
  { key: "agent", label: "Agent 分析" },
  { key: "synthesize", label: "彙整結果" },
];
