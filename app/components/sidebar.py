"""Streamlit 側邊欄元件 — 股票選擇、參數設定"""

import streamlit as st

from src.utils.constants import STOCK_LIST


def render_sidebar(*, show_predict_days: bool = False) -> dict:
    """渲染側邊欄，回傳使用者選擇的參數

    Args:
        show_predict_days: 是否顯示預測天數滑桿（僅走勢預測頁需要）

    Returns:
        {
            "stock_id": str,
            "stock_name": str,
            "lookback_days": int,
            "predict_days": int,
        }
    """
    st.sidebar.title("台股走勢預測系統")
    st.sidebar.markdown("---")

    # 股票選擇（單一 selectbox，避免雙輸入衝突）
    stock_id = st.sidebar.selectbox(
        "股票代號",
        options=list(STOCK_LIST.keys()),
        format_func=lambda sid: f"{sid} {STOCK_LIST[sid]}",
    )
    stock_name = STOCK_LIST[stock_id]

    st.sidebar.markdown("---")

    # 分析週期
    lookback_days = st.sidebar.select_slider(
        "分析週期（天）",
        options=[30, 60, 90, 180, 365],
        value=90,
    )

    # 預測天數（僅走勢預測頁顯示）
    predict_days = 5
    if show_predict_days:
        predict_days = st.sidebar.select_slider(
            "預測天數",
            options=[5, 10, 20],
            value=5,
        )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"目前選擇: **{stock_id} {stock_name}**")

    return {
        "stock_id": stock_id,
        "stock_name": stock_name,
        "lookback_days": lookback_days,
        "predict_days": predict_days,
    }
