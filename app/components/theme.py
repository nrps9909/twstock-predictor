"""全域主題樣式 — 注入自訂 CSS 提升視覺一致性"""

import streamlit as st

# ── 設計語言 ─────────────────────────────────────────────
#   色系: 深藍灰底 + 金黃主色 + 台股紅漲綠跌
#   字體: Noto Sans TC（中文）+ JetBrains Mono（數字）
#   風格: 金融終端機 — 資訊密集、沈穩、不花俏
# ────────────────────────────────────────────────────────

_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── CSS Variables ── */
:root {
    --bg-primary: #0B0F19;
    --bg-secondary: #131825;
    --bg-card: #1A1F2E;
    --bg-hover: #222838;
    --text-primary: #E2E4E9;
    --text-secondary: #8B90A0;
    --text-muted: #5A5F70;
    --accent-gold: #D4A017;
    --accent-gold-dim: rgba(212, 160, 23, 0.15);
    --accent-gold-glow: rgba(212, 160, 23, 0.3);
    --signal-buy: #EF5350;    /* 台股紅漲 */
    --signal-sell: #26A69A;   /* 台股綠跌 */
    --signal-hold: #FFC107;
    --signal-buy-bg: rgba(239, 83, 80, 0.1);
    --signal-sell-bg: rgba(38, 166, 154, 0.1);
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-accent: rgba(212, 160, 23, 0.25);
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
}

/* ── Global Typography ── */
html, body, [class*="css"] {
    font-family: 'Noto Sans TC', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Headings */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
}

h2, h3 {
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
}

/* ── Page Title Bar ── */
[data-testid="stMainBlockContainer"] > div:first-child h1 {
    padding-bottom: 0.3rem;
    border-bottom: 2px solid var(--accent-gold);
    margin-bottom: 1.2rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1120 0%, #0B0F19 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] .stMarkdown h1 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: var(--accent-gold) !important;
    letter-spacing: 0.03em !important;
    text-transform: none;
    border-bottom: none !important;
}

/* Sidebar caption */
section[data-testid="stSidebar"] .stCaption {
    background: var(--accent-gold-dim);
    border: 1px solid var(--border-accent);
    border-radius: var(--radius-sm);
    padding: 0.5rem 0.75rem !important;
    margin-top: 0.5rem;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.8rem 1rem !important;
    transition: border-color 0.2s ease;
}

[data-testid="stMetric"]:hover {
    border-color: var(--border-accent) !important;
}

[data-testid="stMetric"] label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 1.4rem !important;
    color: var(--text-primary) !important;
}

/* Positive delta (台股紅漲) */
[data-testid="stMetricDelta"] svg[viewBox*="0 0 8"] + div,
[data-testid="stMetricDelta"][style*="color: rgb(9, 171, 59)"],
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
    transition: all 0.2s ease !important;
    border: 1px solid var(--border-subtle) !important;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #D4A017 0%, #B8860B 100%) !important;
    color: #0B0F19 !important;
    border: none !important;
    font-weight: 600 !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #E0B020 0%, #D4A017 100%) !important;
    box-shadow: 0 0 20px var(--accent-gold-glow) !important;
}

.stButton > button:not([kind="primary"]):hover {
    border-color: var(--accent-gold) !important;
    color: var(--accent-gold) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: var(--bg-secondary) !important;
    border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
    padding: 0.25rem 0.25rem 0 !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.15s ease !important;
    border-bottom: 2px solid transparent !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: var(--bg-hover) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--accent-gold) !important;
    border-bottom: 2px solid var(--accent-gold) !important;
    background: var(--bg-card) !important;
}

/* Tab highlight bar override */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--accent-gold) !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    font-weight: 500 !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
}

.streamlit-expanderHeader:hover {
    border-color: var(--border-accent) !important;
}

/* ── Data Tables ── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}

[data-testid="stDataFrame"] th {
    font-family: 'Noto Sans TC', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-secondary) !important;
}

[data-testid="stDataFrame"] td {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* ── Info / Warning / Error Boxes ── */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    border-left-width: 3px !important;
}

/* ── Plotly Charts ── */
.js-plotly-plot {
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}

/* ── Dividers ── */
hr {
    border-color: var(--border-subtle) !important;
    margin: 1.2rem 0 !important;
}

/* ── Captions ── */
.stCaption {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
}

/* ── Select Slider ── */
[data-testid="stThumbValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
}

/* ── Progress Bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-gold), #E0B020) !important;
    border-radius: 4px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
    background: var(--text-muted);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-secondary);
}

/* ── Signal indicator utility classes ── */
.signal-buy { color: var(--signal-buy) !important; }
.signal-sell { color: var(--signal-sell) !important; }
.signal-hold { color: var(--signal-hold) !important; }
</style>
"""


def inject_theme() -> None:
    """注入全域自訂 CSS — 應在每頁最頂端呼叫"""
    st.markdown(_CSS, unsafe_allow_html=True)
