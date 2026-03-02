"""應用啟動元件 — 確保 DB 只初始化一次"""

import streamlit as st

from src.db.database import init_db


def ensure_db_initialized() -> None:
    """用 session_state 旗標確保 init_db 只執行一次"""
    if not st.session_state.get("_db_initialized"):
        init_db()
        st.session_state["_db_initialized"] = True
