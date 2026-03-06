"""常數定義 — 股票代碼對照、平台 URL 等"""

# 預設支援的股票清單
STOCK_LIST: dict[str, str] = {
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
}

# 技術指標參數
TECHNICAL_PARAMS = {
    "sma_windows": [5, 10, 20, 60],
    "ema_windows": [12, 26],
    "kd_window": 9,
    "rsi_window": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bias_windows": [5, 10, 20],
    "bb_window": 20,
    "adx_window": 14,
}

# 情緒爬蟲平台設定
SENTIMENT_SOURCES = {
    "ptt_stock": {
        "name": "PTT 股票板",
        "base_url": "https://www.ptt.cc/bbs/Stock/index.html",
        "search_url": "https://www.ptt.cc/bbs/Stock/search?q={}",
    },
    "dcard_money": {
        "name": "Dcard 投資理財板",
        "base_url": "https://www.dcard.tw/f/money",
        "search_url": "https://www.dcard.tw/f/money?q={}",
    },
    "cnyes": {
        "name": "鉅亨網",
        "base_url": "https://www.cnyes.com/twstock/",
        "stock_url": "https://www.cnyes.com/twstock/{}",
    },
    "yahoo_tw": {
        "name": "Yahoo 股市",
        "base_url": "https://tw.stock.yahoo.com/",
        "stock_url": "https://tw.stock.yahoo.com/quote/{}.TW",
    },
}

# 情緒分類標籤
SENTIMENT_LABELS = {
    "bullish": "利多",
    "bearish": "利空",
    "neutral": "中性",
}

# 已下市/合併股票（存活者偏誤修正用）
DELISTED_STOCKS: dict[str, dict] = {
    "2049": {
        "name": "上銀",
        "delist_date": "2023-07-27",
        "reason": "合併下市",
        "merged_into": "2002",
    },
    "3474": {
        "name": "華亞科",
        "delist_date": "2016-12-06",
        "reason": "合併下市",
        "merged_into": "2408",
    },
    "2498": {"name": "宏達電", "delist_date": None, "reason": None},  # placeholder
    # 可持續擴充
}
