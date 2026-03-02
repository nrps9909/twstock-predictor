"""Streamlit Page 1: 技術分析"""

from datetime import date, timedelta

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.startup import ensure_db_initialized
from app.components.theme import inject_theme
from app.components.charts import (
    create_candlestick_chart,
    create_kd_chart,
    create_rsi_chart,
    create_macd_chart,
    create_bollinger_chart,
)
from src.data.stock_fetcher import StockFetcher
from src.analysis.technical import TechnicalAnalyzer
from src.db.database import get_stock_prices, upsert_stock_prices

st.set_page_config(page_title="技術分析", page_icon="📊", layout="wide")
inject_theme()

ensure_db_initialized()


@st.cache_data(ttl=300)
def _load_prices(stock_id: str, start: str, end: str):
    """快取 DB 讀取 + 技術指標計算"""
    from src.db.database import get_stock_prices as _get
    df = _get(stock_id, date.fromisoformat(start), date.fromisoformat(end))
    if df.empty:
        return df, df
    analyzer = TechnicalAnalyzer()
    df_ta = analyzer.compute_all(df)
    return df, df_ta

# 側邊欄
params = render_sidebar()
stock_id = params["stock_id"]
stock_name = params["stock_name"]
lookback_days = params["lookback_days"]

st.title(f"📊 技術分析 — {stock_id} {stock_name}")

# ── 資料取得 ─────────────────────────────────────────────

end_date = date.today()
# 多抓一些資料讓指標暖身（最長 SMA 需要 60 日）
start_date = end_date - timedelta(days=lookback_days + 120)

# 先嘗試從 DB 讀取（含快取）
df, _ = _load_prices(stock_id, start_date.isoformat(), end_date.isoformat())

# 若 DB 資料不足，從 API 抓取
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 更新資料", use_container_width=True):
        with st.spinner("從 FinMind 抓取資料中..."):
            fetcher = StockFetcher()
            df_new = fetcher.fetch_all(
                stock_id, start_date.isoformat(), end_date.isoformat()
            )
            if not df_new.empty:
                upsert_stock_prices(df_new, stock_id)
                st.cache_data.clear()
                df = get_stock_prices(stock_id, start_date, end_date)
                st.success(f"已更新 {len(df_new)} 筆資料")
            else:
                st.error("無法取得資料，請確認 FinMind Token 設定")

if df.empty:
    st.warning("尚無資料。請點擊「更新資料」從 API 抓取。")
    st.info("提示：需要設定 FinMind API Token（.env 檔案中的 FINMIND_TOKEN）")
    st.stop()

# ── 計算技術指標（使用快取版本）──────────────────────────

_, df_ta_cached = _load_prices(stock_id, start_date.isoformat(), end_date.isoformat())
if df_ta_cached.empty:
    # 若是剛更新的資料，重新計算
    df_ta = TechnicalAnalyzer().compute_all(df)
else:
    df_ta = df_ta_cached

# analyzer 在訊號計算和圖表生成中使用
analyzer = TechnicalAnalyzer()

# 只顯示使用者要求的天數範圍
display_start = end_date - timedelta(days=lookback_days)
df_display = df_ta[df_ta["date"] >= display_start].reset_index(drop=True)

if df_display.empty:
    st.warning("顯示範圍內無資料")
    st.stop()

# ── 最新報價資訊 ─────────────────────────────────────────

latest = df_display.iloc[-1]
prev = df_display.iloc[-2] if len(df_display) >= 2 else latest

price_change = latest["close"] - prev["close"]
pct_change = (price_change / prev["close"]) * 100 if prev["close"] else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("收盤價", f"{latest['close']:.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
col2.metric("最高", f"{latest['high']:.2f}")
col3.metric("最低", f"{latest['low']:.2f}")
col4.metric("成交量(張)", f"{latest['volume']:,.0f}")
col5.markdown(f"**日期**  \n{latest['date']}")

st.markdown("---")

# ── 買賣訊號 ─────────────────────────────────────────────

signals = analyzer.get_signals(df_display)
if signals:
    summary = signals.get("summary", {})
    signal_text = {"buy": "🟢 買進", "sell": "🔴 賣出", "hold": "🟡 持有"}
    signal_display = signal_text.get(summary.get("signal", "hold"), "🟡 持有")

    st.subheader(f"綜合訊號: {signal_display}")
    st.caption(summary.get("reason", ""))

    with st.expander("各指標訊號明細"):
        sig_cols = st.columns(5)
        indicator_names = {"kd": "KD", "rsi": "RSI", "macd": "MACD", "bias": "乖離率", "bb": "布林"}
        signal_icons = {"buy": "🟢", "sell": "🔴", "neutral": "⚪"}

        for i, (key, name) in enumerate(indicator_names.items()):
            sig = signals.get(key, {})
            icon = signal_icons.get(sig.get("signal", "neutral"), "⚪")
            sig_cols[i].markdown(f"**{name}** {icon}")
            sig_cols[i].caption(sig.get("reason", ""))

st.markdown("---")

# ── 圖表 ────────────────────────────────────────────────

chart_data = analyzer.generate_chart_data(df_display)

# K 線圖
st.plotly_chart(
    create_candlestick_chart(
        chart_data["ohlcv"],
        ma_lines=chart_data["ma_lines"],
        title=f"{stock_id} {stock_name} K 線圖",
    ),
    use_container_width=True,
)

# 指標分頁（避免垂直堆疊過長）
tab_kd, tab_rsi, tab_macd, tab_bb = st.tabs(["KD", "RSI", "MACD", "布林通道"])

with tab_kd:
    st.plotly_chart(create_kd_chart(chart_data["kd"]), use_container_width=True)

with tab_rsi:
    st.plotly_chart(create_rsi_chart(chart_data["rsi"]), use_container_width=True)

with tab_macd:
    st.plotly_chart(create_macd_chart(chart_data["macd"]), use_container_width=True)

with tab_bb:
    st.plotly_chart(create_bollinger_chart(df_display), use_container_width=True)

# ── 原始資料表 ──────────────────────────────────────────

with st.expander("查看原始資料"):
    display_cols = [
        "date", "open", "high", "low", "close", "volume",
        "sma_5", "sma_20", "sma_60", "kd_k", "kd_d", "rsi_14",
        "macd", "macd_signal", "macd_hist",
    ]
    available_cols = [c for c in display_cols if c in df_display.columns]
    st.dataframe(
        df_display[available_cols].tail(30),
        use_container_width=True,
        hide_index=True,
    )
