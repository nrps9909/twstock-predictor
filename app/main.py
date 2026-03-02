"""台股走勢預測系統 — Streamlit 主入口"""

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.startup import ensure_db_initialized
from app.components.theme import inject_theme

st.set_page_config(
    page_title="台股走勢預測系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()

# 初始化資料庫（只執行一次）
ensure_db_initialized()

# 側邊欄
params = render_sidebar()

# ── 主頁面 ───────────────────────────────────────────────

st.title("台股走勢預測系統")

st.markdown(
    "結合 **技術分析**、**社群情緒** 與 **機器學習** 的台股走勢預測工具。"
)

st.markdown("---")

# 功能卡片（3 欄）
col_ta, col_sent, col_pred = st.columns(3)

with col_ta:
    st.markdown(
        """
        #### 📊 技術分析
        K 線、KD、RSI、MACD、布林通道

        計算技術指標並產生買賣訊號
        """,
    )

with col_sent:
    st.markdown(
        """
        #### 💬 情緒分析
        社群爬蟲 + LLM 情緒評分

        追蹤 PTT、Dcard、鉅亨網的市場情緒
        """,
    )

with col_pred:
    st.markdown(
        """
        #### 📈 走勢預測
        ML 集成模型 + Multi-Agent 分析

        基於歷史+技術+情緒的走勢預測
        """,
    )

st.markdown("---")

st.markdown(
    """
    **快速開始**

    1. 左側欄選擇股票代號與分析週期
    2. 點擊上方分頁切換功能
    3. 首次使用需先「更新資料」抓取歷史行情
    4. 預測功能需先「訓練模型」
    """
)

st.caption(
    "投資有風險，預測結果僅供參考，不構成投資建議。"
    " | 教學專題：投資學 114-02, NTNU"
)
