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
  { key: "narrative", label: "Opus 分析", icon: "FileText" },
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
    "macro_risk", "taiwan_etf_momentum", "us_manufacturing",
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
  taiwan_etf_momentum: "台股ETF動能",
  us_manufacturing: "美國製造",
};

export const REGIME_LABELS: Record<string, string> = {
  bull: "多頭",
  bear: "空頭",
  sideways: "盤整",
};

// ── Factor Insight Generators (白話解讀) ─────────

export const FACTOR_INSIGHT_GENERATORS: Record<string, (score: number) => { text: string; sentiment: "bull" | "bear" | "neutral" }> = {
  foreign_flow: (s) =>
    s > 0.6 ? { text: "外資近期偏買超，資金流入支撐", sentiment: "bull" }
    : s < 0.4 ? { text: "外資大量賣超，短期壓力沉重", sentiment: "bear" }
    : { text: "外資進出持平，觀望氣氛濃厚", sentiment: "neutral" },
  technical_signal: (s) =>
    s > 0.6 ? { text: "技術指標轉強，短線多方訊號浮現", sentiment: "bull" }
    : s < 0.4 ? { text: "技術面走弱，注意下行風險", sentiment: "bear" }
    : { text: "技術面中性，缺乏明確方向", sentiment: "neutral" },
  short_momentum: (s) =>
    s > 0.6 ? { text: "短期動能強勁，上漲趨勢延續", sentiment: "bull" }
    : s < 0.4 ? { text: "短期動能轉弱，反彈力道不足", sentiment: "bear" }
    : { text: "短期動能平淡，盤整機率高", sentiment: "neutral" },
  trust_flow: (s) =>
    s > 0.6 ? { text: "投信持續加碼，中期看好", sentiment: "bull" }
    : s < 0.4 ? { text: "投信減碼出場，留意賣壓", sentiment: "bear" }
    : { text: "投信動向平穩", sentiment: "neutral" },
  volume_anomaly: (s) =>
    s > 0.6 ? { text: "成交量明顯放大，市場關注度提升", sentiment: "bull" }
    : s < 0.2 ? { text: "量能急縮，流動性風險升高", sentiment: "bear" }
    : s < 0.4 ? { text: "成交量偏低，買氣觀望", sentiment: "bear" }
    : { text: "量能正常，無異常波動", sentiment: "neutral" },
  margin_sentiment: (s) =>
    s > 0.6 ? { text: "融資水位健康，散戶信心穩定", sentiment: "bull" }
    : s < 0.2 ? { text: "融資斷頭或大幅去槓桿，市場恐慌", sentiment: "bear" }
    : s < 0.4 ? { text: "融資情緒偏弱，散戶信心不足", sentiment: "bear" }
    : { text: "融資變化不大", sentiment: "neutral" },
  trend_momentum: (s) =>
    s > 0.6 ? { text: "中期趨勢向上，均線多頭排列", sentiment: "bull" }
    : s < 0.4 ? { text: "中期趨勢偏空，均線空頭排列", sentiment: "bear" }
    : { text: "趨勢不明，多空交纏", sentiment: "neutral" },
  revenue_momentum: (s) =>
    s > 0.6 ? { text: "營收成長動能強，基本面轉佳", sentiment: "bull" }
    : s < 0.4 ? { text: "營收成長放緩，業績展望偏弱", sentiment: "bear" }
    : { text: "營收持平，無明顯變化", sentiment: "neutral" },
  institutional_sync: (s) =>
    s > 0.6 ? { text: "三大法人同步買超，籌碼面共識強", sentiment: "bull" }
    : s < 0.2 ? { text: "三大法人同步大賣，籌碼面壓力沉重", sentiment: "bear" }
    : s < 0.4 ? { text: "法人看法分歧，籌碼面不穩", sentiment: "bear" }
    : { text: "法人看法分歧，籌碼無共識", sentiment: "neutral" },
  volatility_regime: (s) =>
    s > 0.6 ? { text: "波動率處於低檔，走勢相對穩定", sentiment: "bull" }
    : s < 0.2 ? { text: "波動率飆升，價格劇烈震盪", sentiment: "bear" }
    : s < 0.4 ? { text: "波動率偏高，走勢震盪加劇", sentiment: "bear" }
    : { text: "波動率適中", sentiment: "neutral" },
  news_sentiment: (s) =>
    s > 0.6 ? { text: "近期新聞偏正面，市場情緒樂觀", sentiment: "bull" }
    : s < 0.4 ? { text: "負面新聞增加，市場情緒悲觀", sentiment: "bear" }
    : { text: "新聞面中性，無重大消息", sentiment: "neutral" },
  global_context: (s) =>
    s > 0.6 ? { text: "國際環境有利，半導體/科技族群受惠", sentiment: "bull" }
    : s < 0.4 ? { text: "國際局勢不利，外部風險升高", sentiment: "bear" }
    : { text: "國際環境持平，影響有限", sentiment: "neutral" },
  margin_quality: (s) =>
    s > 0.6 ? { text: "毛利率持續擴張，獲利能力穩健", sentiment: "bull" }
    : s < 0.4 ? { text: "毛利率下滑，獲利品質惡化", sentiment: "bear" }
    : { text: "毛利率持穩", sentiment: "neutral" },
  sector_rotation: (s) =>
    s > 0.6 ? { text: "所屬類股資金輪入，板塊表現領先", sentiment: "bull" }
    : s < 0.4 ? { text: "類股資金輪出，板塊表現落後", sentiment: "bear" }
    : { text: "類股表現持平", sentiment: "neutral" },
  ml_ensemble: (s) =>
    s > 0.6 ? { text: "機器學習模型看多，預測報酬正向", sentiment: "bull" }
    : s < 0.4 ? { text: "ML 模型看空，預測報酬偏負", sentiment: "bear" }
    : { text: "ML 模型無明確方向", sentiment: "neutral" },
  fundamental_value: (s) =>
    s > 0.6 ? { text: "估值偏低具吸引力，價值面支撐", sentiment: "bull" }
    : s < 0.4 ? { text: "估值偏高，安全邊際不足", sentiment: "bear" }
    : { text: "估值合理，無明顯偏離", sentiment: "neutral" },
  liquidity_quality: (s) =>
    s > 0.6 ? { text: "流動性充足，買賣價差小", sentiment: "bull" }
    : s < 0.4 ? { text: "流動性不佳，大單進出困難", sentiment: "bear" }
    : { text: "流動性正常", sentiment: "neutral" },
  macro_risk: (s) =>
    s > 0.6 ? { text: "宏觀環境有利，利率/匯率走勢正面", sentiment: "bull" }
    : s < 0.4 ? { text: "宏觀風險升高，留意利率與匯率變動", sentiment: "bear" }
    : { text: "宏觀環境中性", sentiment: "neutral" },
  taiwan_etf_momentum: (s) =>
    s > 0.6 ? { text: "台股ETF動能強勁，資金持續流入", sentiment: "bull" }
    : s < 0.4 ? { text: "台股ETF動能轉弱，市場信心不足", sentiment: "bear" }
    : { text: "台股ETF表現持穩", sentiment: "neutral" },
  us_manufacturing: (s) =>
    s > 0.6 ? { text: "美國製造業擴張，全球需求回溫", sentiment: "bull" }
    : s < 0.4 ? { text: "美國製造業收縮，全球需求疲軟", sentiment: "bear" }
    : { text: "美國製造業持穩", sentiment: "neutral" },
};
