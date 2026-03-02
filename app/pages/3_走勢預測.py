"""Streamlit Page 3: 走勢預測 + 回測結果 + Agent 推理"""

from datetime import date, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px

from app.components.sidebar import render_sidebar
from app.components.theme import inject_theme
from app.components.charts import (
    create_prediction_chart,
    create_radar_chart,
)
from app.components.startup import ensure_db_initialized
from src.db.database import get_stock_prices, get_predictions
from src.models.trainer import ModelTrainer
from src.analysis.technical import TechnicalAnalyzer

st.set_page_config(page_title="走勢預測", page_icon="📈", layout="wide")
inject_theme()
ensure_db_initialized()

params = render_sidebar(show_predict_days=True)
stock_id = params["stock_id"]
stock_name = params["stock_name"]
lookback_days = params["lookback_days"]
predict_days = params["predict_days"]

st.title(f"📈 走勢預測 — {stock_id} {stock_name}")

# ── 切換股票時清除過期結果 ────────────────────────────
if st.session_state.get("_active_stock_id") != stock_id:
    for key in [
        "prediction", "pred_stock_id", "backtest_results",
        "agent_decision", "agent_summary",
    ]:
        st.session_state.pop(key, None)
    st.session_state["_active_stock_id"] = stock_id

# ── 模型狀態 ─────────────────────────────────────────────

from pathlib import Path
model_dir = Path(__file__).resolve().parent.parent.parent / "models"
lstm_exists = (model_dir / f"{stock_id}_lstm.pt").exists()
xgb_exists = (model_dir / f"{stock_id}_xgb.json").exists()

col1, col2 = st.columns(2)
col1.markdown(f"**LSTM 模型**：{'✅ 已訓練' if lstm_exists else '❌ 未訓練'}")
col2.markdown(f"**XGBoost 模型**：{'✅ 已訓練' if xgb_exists else '❌ 未訓練'}")

st.markdown("---")

# ── Tab 分頁 ─────────────────────────────────────────────

tab_predict, tab_backtest, tab_agent = st.tabs(["📊 預測", "📈 回測", "🤖 Agent 分析"])

# ── Tab 1: 訓練 & 預測 ────────────────────────────────────

with tab_predict:
    with st.expander("🔧 訓練模型", expanded=not (lstm_exists and xgb_exists)):
        with st.expander("進階設定", expanded=False):
            train_col1, train_col2 = st.columns(2)
            with train_col1:
                train_days = st.number_input("訓練資料天數", 180, 730, 365)
                epochs = st.number_input("LSTM 訓練輪數", 10, 200, 50)
            with train_col2:
                val_ratio = st.slider("驗證集比例", 0.1, 0.3, 0.2)
                test_ratio = st.slider("測試集比例", 0.1, 0.3, 0.2)

        if st.button("🚀 開始訓練", use_container_width=True):
            end_date = date.today()
            start_date = end_date - timedelta(days=train_days)

            with st.spinner(f"訓練 {stock_id} 模型中...（含 3-way split + early stopping）"):
                try:
                    trainer = ModelTrainer(stock_id)
                    results = trainer.train(
                        start_date.isoformat(),
                        end_date.isoformat(),
                        epochs=epochs,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                    )

                    st.success("訓練完成！")

                    # 顯示結果
                    res_cols = st.columns(3)
                    if "lstm" in results:
                        res_cols[0].json(results["lstm"])
                    if "xgboost" in results:
                        res_cols[1].json(results["xgboost"])
                    if "ensemble" in results:
                        res_cols[2].json(results["ensemble"])

                except Exception as e:
                    st.error(f"訓練失敗: {e}")

    st.markdown("---")

    if not (lstm_exists or xgb_exists):
        st.info("尚未訓練模型。請展開上方「訓練模型」區塊並點擊「開始訓練」。")
    else:
        # 載入模型並預測
        if st.button("🔮 執行預測", use_container_width=True, type="primary"):
            end_date = date.today()
            start_date = end_date - timedelta(days=lookback_days + 120)

            with st.spinner("預測中..."):
                try:
                    trainer = ModelTrainer(stock_id)
                    trainer.load_models()

                    result = trainer.predict(
                        start_date.isoformat(),
                        end_date.isoformat(),
                    )

                    if result is None:
                        st.error("預測失敗：無法建立特徵資料")
                        st.stop()

                    st.session_state["prediction"] = result
                    st.session_state["pred_stock_id"] = stock_id

                except Exception as e:
                    st.error(f"預測失敗: {e}")
                    st.stop()

        # 顯示預測結果
        if "prediction" in st.session_state and st.session_state.get("pred_stock_id") == stock_id:
            result = st.session_state["prediction"]

            signal_map = {"buy": "🟢 買進", "sell": "🔴 賣出", "hold": "🟡 持有"}
            signal_display = signal_map.get(result.signal, "🟡 持有")

            st.subheader(f"綜合建議: {signal_display}")
            st.progress(result.signal_strength, text=f"訊號強度: {result.signal_strength:.0%}")

            col1, col2, col3, col4 = st.columns(4)
            total_return = result.predicted_returns.sum()
            col1.metric("預測總報酬", f"{total_return:+.2%}")
            col2.metric("預測終價", f"{result.predicted_prices[-1]:.2f}")
            col3.metric("LSTM 權重", f"{result.lstm_weight:.0%}")
            col4.metric("XGBoost 權重", f"{result.xgb_weight:.0%}")

            st.markdown("---")

            # 預測走勢圖
            end_date = date.today()
            start_date = end_date - timedelta(days=lookback_days)
            price_df = get_stock_prices(stock_id, start_date, end_date)

            if not price_df.empty:
                pred_dates = pd.bdate_range(
                    end_date + timedelta(days=1), periods=len(result.predicted_prices)
                )
                pred_df = pd.DataFrame({
                    "target_date": pred_dates,
                    "predicted_price": result.predicted_prices,
                    "confidence_lower": result.confidence_lower,
                    "confidence_upper": result.confidence_upper,
                })

                st.plotly_chart(
                    create_prediction_chart(price_df, pred_df),
                    use_container_width=True,
                )
                st.caption("預測區間由模型歷史誤差推導，以加權標準差展開。")

            # 雷達圖 + 特徵重要性（折疊）
            with st.expander("模型分析詳情"):
                col_left, col_right = st.columns(2)

                with col_left:
                    ta = TechnicalAnalyzer()
                    if not price_df.empty:
                        df_ta = ta.compute_all(price_df)
                        signals = ta.get_signals(df_ta)
                        tech_score = signals.get("summary", {}).get("score", 0.5)
                    else:
                        tech_score = 0.5

                    pred_score = 0.5 + min(max(total_return * 5, -0.5), 0.5)
                    scores = {
                        "技術面": tech_score,
                        "預測面": pred_score,
                        "信心度": result.signal_strength,
                    }
                    st.plotly_chart(create_radar_chart(scores), use_container_width=True)

                with col_right:
                    try:
                        trainer = ModelTrainer(stock_id)
                        trainer.load_models()
                        if trainer.xgb:
                            importance = trainer.xgb.get_feature_importance(15)
                            imp_df = pd.DataFrame(
                                list(importance.items()),
                                columns=["特徵", "重要性"],
                            )
                            fig = px.bar(
                                imp_df, x="重要性", y="特徵",
                                orientation="h",
                                title="XGBoost 特徵重要性 Top 15",
                            )
                            fig.update_layout(
                                template="twstock",
                                height=400,
                                yaxis=dict(autorange="reversed"),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.info("需要訓練 XGBoost 模型以顯示特徵重要性")

            # 每日預測明細
            with st.expander("📋 預測明細"):
                detail_data = []
                for i, (ret, price, lo, hi) in enumerate(zip(
                    result.predicted_returns,
                    result.predicted_prices,
                    result.confidence_lower,
                    result.confidence_upper,
                )):
                    detail_data.append({
                        "天數": f"T+{i+1}",
                        "預測報酬率": f"{ret:+.4f}",
                        "預測價格": f"{price:.2f}",
                        "CI 下界": f"{lo:.2f}",
                        "CI 上界": f"{hi:.2f}",
                    })
                st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)

# ── Tab 2: 回測結果 ──────────────────────────────────────

with tab_backtest:
    st.subheader("📈 Walk-Forward 回測")

    with st.expander("進階設定", expanded=False):
        bt_col1, bt_col2 = st.columns(2)
        with bt_col1:
            bt_days = st.number_input("回測資料天數", 365, 1095, 730, key="bt_days")
            n_folds = st.number_input("驗證折數", 3, 10, 5, key="n_folds")
        with bt_col2:
            bt_epochs = st.number_input("每折訓練輪數", 10, 50, 30, key="bt_epochs")

    if st.button("🔄 執行 Walk-Forward 回測", use_container_width=True):
        end_date = date.today()
        start_date = end_date - timedelta(days=bt_days)

        with st.spinner(f"Walk-forward 回測中 ({n_folds} folds)..."):
            try:
                trainer = ModelTrainer(stock_id)
                fold_results = trainer.walk_forward_validate(
                    start_date.isoformat(),
                    end_date.isoformat(),
                    n_splits=n_folds,
                    epochs=bt_epochs,
                )

                st.session_state["backtest_results"] = fold_results
                st.success(f"回測完成！{len(fold_results)} 折結果")

            except Exception as e:
                st.error(f"回測失敗: {e}")

    if "backtest_results" in st.session_state:
        results = st.session_state["backtest_results"]

        # 結果表格
        bt_df = pd.DataFrame(results)
        st.dataframe(bt_df, use_container_width=True, hide_index=True)

        # 視覺化
        if "lstm_mse" in bt_df.columns:
            fig_mse = px.bar(
                bt_df, x="fold", y=["lstm_mse", "xgb_mse"],
                barmode="group",
                title="各折 MSE 比較",
            )
            fig_mse.update_layout(template="twstock")
            st.plotly_chart(fig_mse, use_container_width=True)

        if "lstm_direction_acc" in bt_df.columns:
            fig_acc = px.bar(
                bt_df, x="fold", y=["lstm_direction_acc", "xgb_direction_acc"],
                barmode="group",
                title="各折方向準確率",
            )
            fig_acc.update_layout(template="twstock")
            fig_acc.add_hline(y=0.5, line_dash="dash", annotation_text="50% baseline")
            st.plotly_chart(fig_acc, use_container_width=True)

# ── Tab 3: Agent 分析 ─────────────────────────────────────

with tab_agent:
    st.subheader("🤖 Multi-Agent 智慧分析")
    st.caption("使用 4 位分析師 Agent + 研究員辯論 + 交易員決策 + 風控審核")

    if st.button("🧠 執行 Agent 分析", use_container_width=True, type="primary"):
        with st.spinner("Agent 系統分析中...（技術面 → 情緒面 → 基本面 → 量化 → 辯論 → 決策 → 風控）"):
            try:
                import asyncio
                from src.agents.orchestrator import AgentOrchestrator
                from src.agents.base import MarketContext
                from src.data.stock_fetcher import StockFetcher

                fetcher = StockFetcher()
                end_date = date.today()
                start_date = end_date - timedelta(days=lookback_days)
                price_df = get_stock_prices(stock_id, start_date, end_date)

                if price_df.empty:
                    st.error("無法取得價格資料")
                    st.stop()

                current_price = price_df["close"].iloc[-1]

                # 準備市場上下文
                ta = TechnicalAnalyzer()
                df_ta = ta.compute_all(price_df)
                signals = ta.get_signals(df_ta)

                context = MarketContext(
                    stock_id=stock_id,
                    current_price=float(current_price),
                    date=end_date.isoformat(),
                    technical_summary={
                        "rsi_14": float(df_ta["rsi_14"].iloc[-1]) if "rsi_14" in df_ta else 50,
                        "macd": float(df_ta["macd"].iloc[-1]) if "macd" in df_ta else 0,
                        "sma_5": float(df_ta["sma_5"].iloc[-1]) if "sma_5" in df_ta else current_price,
                        "sma_20": float(df_ta["sma_20"].iloc[-1]) if "sma_20" in df_ta else current_price,
                        "bb_width": float(df_ta["bb_width"].iloc[-1]) if "bb_width" in df_ta else 0,
                        "adx": float(df_ta["adx"].iloc[-1]) if "adx" in df_ta else 25,
                        "signals_summary": signals.get("summary", {}),
                    },
                )

                orchestrator = AgentOrchestrator()
                decision = asyncio.run(orchestrator.run_analysis(context))

                st.session_state["agent_decision"] = decision
                st.session_state["agent_summary"] = orchestrator.get_analysis_summary()

            except Exception as e:
                st.error("Agent 分析暫時無法使用")
                import traceback
                with st.expander("錯誤詳情（開發者用）"):
                    st.code(traceback.format_exc())

    if "agent_decision" in st.session_state:
        decision = st.session_state["agent_decision"]
        summary = st.session_state.get("agent_summary", {})

        # 決策結果
        action_map = {"buy": "🟢 買進", "sell": "🔴 賣出", "hold": "🟡 持有"}
        st.subheader(f"Agent 決策: {action_map.get(decision.action, '🟡 持有')}")

        col1, col2 = st.columns(2)
        col1.metric("信心度", f"{decision.confidence:.0%}")
        col2.metric("風控", "✅ 核准" if decision.approved_by_risk else "❌ 否決")

        st.caption(decision.reasoning)

        with st.expander("決策詳情"):
            st.metric("倉位建議", f"{decision.position_size:.0%}")

            st.markdown("**交易邏輯:**")
            st.info(decision.reasoning)

            if decision.risk_notes:
                st.markdown("**風控備註:**")
                st.warning(decision.risk_notes)

        # 各 Agent 觀點
        with st.expander("🔍 各分析師觀點"):
            for msg in decision.analyst_reports:
                signal_emoji = {"strong_buy": "📈📈", "buy": "📈", "hold": "➡️", "sell": "📉", "strong_sell": "📉📉"}
                sig_val = msg.signal.value if msg.signal else "hold"
                st.markdown(
                    f"**{msg.sender.value.upper()}** "
                    f"{signal_emoji.get(sig_val, '➡️')} "
                    f"(信心 {msg.confidence:.0%})"
                )
                st.caption(msg.reasoning)

        if decision.researcher_report:
            with st.expander("🔬 研究員辯論"):
                report = decision.researcher_report
                st.markdown(f"**結論:** {report.signal.value if report.signal else 'N/A'} "
                            f"(信心 {report.confidence:.0%})")
                st.caption(report.reasoning)

                content = report.content
                if isinstance(content, dict):
                    bull = content.get("bull_case", {})
                    bear = content.get("bear_case", {})
                    if bull:
                        st.markdown("**看多:**")
                        for arg in bull.get("key_arguments", []):
                            st.markdown(f"- {arg}")
                    if bear:
                        st.markdown("**看空:**")
                        for arg in bear.get("key_arguments", []):
                            st.markdown(f"- {arg}")
