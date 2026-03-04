// 常數定義

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
  sell: "#26A69A",
  strong_sell: "#26A69A",
  hold: "#FFC107",
} as const;

export const SIGNAL_LABELS: Record<string, string> = {
  buy: "買進",
  strong_buy: "強力買進",
  sell: "賣出",
  strong_sell: "強力賣出",
  hold: "持有",
};

export const ROLE_LABELS: Record<string, string> = {
  technical: "技術分析師",
  sentiment: "情緒分析師",
  fundamental: "基本面分析師",
  quant: "量化分析師",
  researcher: "首席研究員",
  risk: "風控主管",
};
