// 常數定義
import type { AnalysisPhase } from "./types";

export const STOCK_LIST: Record<string, string> = {
  "2330": "台積電",
  "2317": "鴻海",
  "2382": "廣達",
  "2454": "聯發科",
  "2881": "富邦金",
  "2882": "國泰金",
  "2303": "聯電",
  "3711": "日月光投控",
  "2308": "台達電",
  "2412": "中華電",
};

export const SIGNAL_COLORS = {
  buy: "#EF5350",
  strong_buy: "#EF5350",
  weak_buy: "#FF8A80",
  sell: "#26A69A",
  strong_sell: "#26A69A",
  weak_sell: "#80CBC4",
  hold: "#FFC107",
  neutral: "#FFC107",
} as const;

export const SIGNAL_LABELS: Record<string, string> = {
  buy: "買進",
  strong_buy: "強力買進",
  weak_buy: "偏多",
  sell: "賣出",
  strong_sell: "強力賣出",
  weak_sell: "偏空",
  hold: "持有",
  neutral: "中性",
};

// ── 6-Phase Pipeline ─────────────────────────────

export const PIPELINE_PHASES: { key: AnalysisPhase; label: string; icon: string }[] = [
  { key: "data_collection", label: "數據收集", icon: "Database" },
  { key: "feature_extraction", label: "特徵萃取", icon: "Cpu" },
  { key: "scoring", label: "多因子評分", icon: "BarChart3" },
  { key: "narrative", label: "敘事生成", icon: "FileText" },
  { key: "risk_control", label: "風險控制", icon: "Shield" },
  { key: "finalize", label: "儲存結果", icon: "Save" },
];

// ── Factor Groups ────────────────────────────────

export const FACTOR_GROUPS: Record<string, string[]> = {
  "短期 (39%)": [
    "foreign_flow", "technical_signal", "short_momentum",
    "trust_flow", "volume_anomaly", "margin_sentiment",
  ],
  "中期 (32%)": [
    "trend_momentum", "revenue_momentum", "institutional_sync",
    "volatility_regime", "news_sentiment", "global_context",
    "margin_quality", "sector_rotation",
  ],
  "長期 (29%)": [
    "ml_ensemble", "fundamental_value", "liquidity_quality",
    "macro_risk", "export_momentum", "us_manufacturing",
  ],
};

export const FACTOR_LABELS: Record<string, string> = {
  foreign_flow: "外資動向",
  technical_signal: "技術訊號",
  short_momentum: "短期動能",
  trust_flow: "投信動向",
  volume_anomaly: "量能異常",
  margin_sentiment: "融資情緒",
  trend_momentum: "趨勢動能",
  revenue_momentum: "營收動能",
  institutional_sync: "法人同步",
  volatility_regime: "波動體制",
  news_sentiment: "新聞情緒",
  global_context: "全球脈絡",
  margin_quality: "資產品質",
  sector_rotation: "類股輪動",
  ml_ensemble: "ML 集成",
  fundamental_value: "基本面價值",
  liquidity_quality: "流動性品質",
  macro_risk: "宏觀風險",
  export_momentum: "出口動能",
  us_manufacturing: "美國製造",
};

export const REGIME_LABELS: Record<string, string> = {
  bull: "多頭",
  bear: "空頭",
  sideways: "盤整",
};
