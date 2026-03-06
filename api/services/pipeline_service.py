"""一鍵預測 Pipeline — 9 步驟串流 Orchestrator"""

import asyncio
import json
import logging
from datetime import date, timedelta
from typing import AsyncGenerator

import pandas as pd

from src.utils.constants import STOCK_LIST
from src.utils.config import settings
from src.db.database import (
    get_stock_prices,
    upsert_stock_prices,
    get_sentiment,
    insert_sentiment,
    save_pipeline_result,
    upsert_data_cache,
    get_data_cache,
)
from src.data.stock_fetcher import StockFetcher
from src.data.sentiment_crawler import SentimentCrawler
from src.analysis.technical import TechnicalAnalyzer
from src.models.trainer import ModelTrainer
from src.agents.base import MarketContext
from src.agents.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

MODEL_DIR = settings.PROJECT_ROOT / "models"


def _event(
    step: str, status: str, progress: int, message: str = "", data: dict | None = None
) -> str:
    """格式化 SSE 事件"""
    payload = {
        "step": step,
        "status": status,
        "progress": progress,
        "message": message,
    }
    if data is not None:
        payload["data"] = data
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _build_fundamental_summary(df: pd.DataFrame) -> dict:
    """從 StockPrice DataFrame 提取法人/融資融券摘要

    Returns:
        dict with institutional flow data for the fundamental agent
    """
    summary = {}
    if df.empty or len(df) < 2:
        return summary

    recent = df.tail(5)
    latest = df.iloc[-1]

    # 法人 5 日累計買賣超
    for col, label in [
        ("foreign_buy_sell", "外資"),
        ("trust_buy_sell", "投信"),
        ("dealer_buy_sell", "自營商"),
    ]:
        if col in df.columns:
            values = recent[col].dropna()
            if not values.empty:
                total = float(values.sum())
                days = len(values)
                # Determine trend direction
                if len(values) >= 2:
                    recent_half = values.iloc[len(values) // 2 :].sum()
                    older_half = values.iloc[: len(values) // 2].sum()
                    if recent_half > older_half:
                        trend = "加速買超" if total > 0 else "減緩賣超"
                    else:
                        trend = "減緩買超" if total > 0 else "加速賣超"
                else:
                    trend = "買超" if total > 0 else "賣超"

                summary[f"{label}_{days}日累計"] = f"{total:+,.0f} 張 ({trend})"
                summary[f"{label}_最新"] = f"{float(values.iloc[-1]):+,.0f} 張"

    # 融資融券
    for col, label in [
        ("margin_balance", "融資餘額"),
        ("short_balance", "融券餘額"),
    ]:
        if col in df.columns and pd.notna(latest.get(col)):
            val = float(latest[col])
            summary[label] = f"{val:,.0f} 張"
            # Compare with 5 days ago
            if len(df) >= 6:
                prev = df.iloc[-6]
                if pd.notna(prev.get(col)):
                    change = val - float(prev[col])
                    pct = (
                        change / float(prev[col]) * 100 if float(prev[col]) != 0 else 0
                    )
                    summary[f"{label}_5日變化"] = f"{change:+,.0f} 張 ({pct:+.1f}%)"

    # 外資連續買賣超天數
    if "foreign_buy_sell" in df.columns:
        fbs = df["foreign_buy_sell"].dropna()
        if not fbs.empty:
            consecutive = 0
            direction = "買超" if fbs.iloc[-1] > 0 else "賣超"
            for val in reversed(fbs.values):
                if (val > 0 and direction == "買超") or (
                    val < 0 and direction == "賣超"
                ):
                    consecutive += 1
                else:
                    break
            if consecutive > 0:
                summary["外資連續天數"] = f"連續 {consecutive} 日{direction}"

    return summary


async def run_pipeline(
    stock_id: str,
    force_retrain: bool = False,
    epochs: int = 50,
) -> AsyncGenerator[str, None]:
    """執行一鍵預測 Pipeline，以 SSE 串流回傳進度

    9 步驟：
    1. CHECK_DATA  — 檢查 DB 是否有近期資料
    2. FETCH_DATA  — 若過期則從 FinMind 抓取
    3. TECHNICAL   — 計算技術指標 + 買賣訊號
    4. SENTIMENT   — 載入已有情緒資料
    5. CHECK_MODEL — 檢查模型檔案是否存在
    6. TRAIN_MODEL — 若不存在則自動訓練
    7. PREDICT     — 執行 ML 集成預測
    8. AGENT       — 執行 Multi-Agent 分析
    9. SYNTHESIZE  — 彙整最終結果
    """
    stock_name = STOCK_LIST.get(stock_id, stock_id)
    today = date.today()
    lookback_start = (today - timedelta(days=730)).isoformat()
    end_str = today.isoformat()

    # 用來收集各步驟結果
    technical_signals = {}
    technical_data = {}
    sentiment_data = {}
    prediction_result = None
    market_state_name = None
    agent_decision = None

    # ── Step 1: CHECK_DATA ────────────────────────────────
    yield _event("check_data", "running", 5, f"檢查 {stock_name} 資料...")

    df = await asyncio.to_thread(get_stock_prices, stock_id)
    has_data = not df.empty
    data_count = len(df)
    latest_date = None
    data_fresh = False

    if has_data:
        latest_date = df["date"].max()
        if isinstance(latest_date, date):
            days_old = (today - latest_date).days
            data_fresh = days_old <= 3  # 3 天內算新鮮
        else:
            data_fresh = False

    yield _event(
        "check_data",
        "done",
        10,
        f"資料 {data_count} 筆" + (f"，最新 {latest_date}" if latest_date else ""),
        {"count": data_count, "latest_date": str(latest_date), "fresh": data_fresh},
    )

    # ── Step 2: FETCH_DATA ────────────────────────────────
    if not data_fresh:
        yield _event("fetch_data", "running", 15, "從 FinMind 抓取最新資料...")

        try:
            fetcher = StockFetcher()
            fetch_start = (today - timedelta(days=400)).isoformat()
            new_df = await asyncio.to_thread(
                fetcher.fetch_all,
                stock_id,
                fetch_start,
                end_str,
            )
            if not new_df.empty:
                await asyncio.to_thread(upsert_stock_prices, new_df, stock_id)
                data_count = len(new_df)
                yield _event(
                    "fetch_data",
                    "done",
                    20,
                    f"抓取 {len(new_df)} 筆資料完成",
                    {"rows": len(new_df)},
                )
            else:
                yield _event("fetch_data", "done", 20, "無新資料（使用既有資料）")
        except Exception as e:
            logger.error("抓取資料失敗: %s", e)
            yield _event("fetch_data", "error", 20, f"抓取失敗: {e}")
            if not has_data:
                yield _event("synthesize", "error", 100, "無資料可用，中止")
                return
    else:
        yield _event("fetch_data", "skipped", 20, "資料已是最新")

    # 重新載入資料
    df = await asyncio.to_thread(get_stock_prices, stock_id)

    # ── Step 3: TECHNICAL ─────────────────────────────────
    yield _event("technical", "running", 25, "計算技術指標...")

    try:
        analyzer = TechnicalAnalyzer()
        df_tech = await asyncio.to_thread(analyzer.compute_all, df)
        technical_signals = analyzer.get_signals(df_tech)
        latest_row = df_tech.iloc[-1]
        current_price = float(latest_row["close"])

        # 提取關鍵指標
        indicators = {}
        for col in [
            "rsi_14",
            "kd_k",
            "kd_d",
            "macd_hist",
            "bias_10",
            "adx",
            "bb_pband",
        ]:
            if col in latest_row.index:
                val = latest_row[col]
                indicators[col] = None if (val != val) else round(float(val), 2)

        technical_data = {
            "signals": technical_signals,
            "indicators": indicators,
            "current_price": current_price,
        }
        yield _event(
            "technical",
            "done",
            35,
            f"技術訊號: {technical_signals.get('summary', {}).get('signal', 'N/A')}",
            technical_data,
        )
    except Exception as e:
        logger.error("技術分析失敗: %s", e)
        yield _event("technical", "error", 35, f"技術分析失敗: {e}")
        current_price = float(df["close"].iloc[-1]) if not df.empty else 0

    # ── Step 4: SENTIMENT ─────────────────────────────────
    yield _event("sentiment", "running", 40, "載入情緒資料...")

    news_headlines = []
    global_context = []

    try:
        # Check DB for recent sentiment data
        sent_df = await asyncio.to_thread(
            get_sentiment,
            stock_id,
            today - timedelta(days=30),
            today,
        )

        # Check freshness — auto-crawl if stale or empty
        data_stale = True
        if not sent_df.empty:
            latest_sent_date = sent_df["date"].max()
            if isinstance(latest_sent_date, date):
                data_stale = (today - latest_sent_date).days > 1

        if data_stale:
            yield _event("sentiment", "running", 41, "自動爬取情緒資料...")
            try:
                crawler = SentimentCrawler()

                # Crawl all sources
                articles = await asyncio.to_thread(crawler.crawl_all, stock_id)
                if articles:
                    # Prepare for DB insert
                    for art in articles:
                        if "sentiment_score" not in art:
                            score_map = {
                                "bullish": 0.5,
                                "bearish": -0.5,
                                "neutral": 0.0,
                            }
                            art["sentiment_score"] = score_map.get(
                                art.get("sentiment_label", "neutral"), 0.0
                            )
                    await asyncio.to_thread(insert_sentiment, articles)
                    yield _event(
                        "sentiment", "running", 43, f"爬取 {len(articles)} 篇情緒資料"
                    )

                # Crawl global context — DB-first (4hr cache)
                global_context = []
                _gc_cache_key = f"global_news:{stock_id}"
                _gc_cached = get_data_cache(_gc_cache_key, today)
                if _gc_cached:
                    try:
                        global_context = json.loads(_gc_cached)
                    except Exception:
                        global_context = []
                if not global_context:
                    global_context = await asyncio.to_thread(
                        crawler.crawl_global_context, stock_id
                    )
                    if global_context:
                        try:
                            upsert_data_cache(
                                _gc_cache_key,
                                today,
                                json.dumps(global_context, ensure_ascii=False),
                            )
                        except Exception:
                            pass

                # Extract news headlines from crawled articles
                news_headlines = [a["title"] for a in articles if a.get("title")]

                # Re-query DB for fresh data
                sent_df = await asyncio.to_thread(
                    get_sentiment,
                    stock_id,
                    today - timedelta(days=30),
                    today,
                )
            except Exception as crawl_err:
                logger.warning("情緒爬取失敗（使用既有資料）: %s", crawl_err)

        # Build sentiment summary
        if not sent_df.empty:
            total = len(sent_df)
            # Handle missing sentiment_score
            scores = sent_df["sentiment_score"].dropna()
            avg_score = float(scores.mean()) if not scores.empty else 0.0
            bullish = len(sent_df[sent_df["sentiment_label"] == "bullish"])
            bearish = len(sent_df[sent_df["sentiment_label"] == "bearish"])
            sentiment_data = {
                "total": total,
                "avg_score": round(avg_score, 3),
                "bullish_ratio": round(bullish / total, 3),
                "bearish_ratio": round(bearish / total, 3),
                "latest_date": str(sent_df["date"].max()),
                "news_headlines": news_headlines[:15],
                "global_context": global_context[:15],
            }
            label = (
                "偏多" if avg_score > 0.1 else ("偏空" if avg_score < -0.1 else "中性")
            )
            yield _event(
                "sentiment",
                "done",
                45,
                f"情緒 {label} ({avg_score:+.2f}, {total} 篇)",
                sentiment_data,
            )
        else:
            sentiment_data = {
                "total": 0,
                "avg_score": 0,
                "bullish_ratio": 0,
                "bearish_ratio": 0,
                "news_headlines": news_headlines[:15],
                "global_context": global_context[:15],
            }
            yield _event(
                "sentiment",
                "done",
                45,
                "無個股情緒資料（將使用全球 context）"
                if global_context
                else "無情緒資料",
                sentiment_data,
            )
    except Exception as e:
        logger.error("情緒載入失敗: %s", e)
        sentiment_data = {
            "total": 0,
            "avg_score": 0,
            "news_headlines": [],
            "global_context": [],
        }
        yield _event("sentiment", "error", 45, f"情緒載入失敗: {e}")

    # ── Step 5: CHECK_MODEL ───────────────────────────────
    yield _event("check_model", "running", 50, "檢查模型...")

    lstm_exists = (MODEL_DIR / f"{stock_id}_lstm.pt").exists()
    xgb_exists = (MODEL_DIR / f"{stock_id}_xgb.json").exists()
    model_exists = lstm_exists and xgb_exists and not force_retrain

    yield _event(
        "check_model",
        "done",
        55,
        "模型已就緒" if model_exists else "需要訓練模型",
        {"exists": model_exists, "lstm": lstm_exists, "xgb": xgb_exists},
    )

    # ── Step 6: TRAIN_MODEL ───────────────────────────────
    trainer = ModelTrainer(stock_id)

    if not model_exists:
        yield _event("train_model", "running", 58, f"訓練模型中（{epochs} epochs）...")

        try:
            train_result = await asyncio.to_thread(
                trainer.train,
                start_date=lookback_start,
                end_date=end_str,
                epochs=epochs,
                use_triple_barrier=True,
            )
            yield _event(
                "train_model",
                "done",
                70,
                "模型訓練完成",
                {
                    "lstm": train_result.get("lstm"),
                    "xgboost": train_result.get("xgboost"),
                },
            )
        except Exception as e:
            logger.error("訓練失敗: %s", e)
            yield _event(
                "train_model",
                "error",
                70,
                f"訓練失敗: {e}（將跳過 ML 預測，僅用 Agent 分析）",
            )
            model_exists = False  # Mark so predict step is skipped
    else:
        yield _event("train_model", "skipped", 70, "使用既有模型")
        try:
            await asyncio.to_thread(trainer.load_models)
        except Exception as e:
            logger.error("載入模型失敗: %s", e)
            yield _event(
                "train_model",
                "error",
                70,
                f"載入模型失敗: {e}（將跳過 ML 預測，僅用 Agent 分析）",
            )

    # ── Step 7: PREDICT ───────────────────────────────────
    # Check if trainer has loaded models (training or loading succeeded)
    _has_models = hasattr(trainer, "lstm_model") and trainer.lstm_model is not None

    if _has_models:
        yield _event("predict", "running", 75, "執行預測...")

        try:
            pred_start = (today - timedelta(days=200)).isoformat()
            prediction_result = await asyncio.to_thread(
                trainer.predict,
                start_date=pred_start,
                end_date=end_str,
            )

            if prediction_result is not None:
                market_state_dict = None
                market_state_name = None
                if prediction_result.market_state:
                    market_state_dict = {
                        "state_name": prediction_result.market_state.state_name,
                        "probabilities": prediction_result.market_state.probabilities.tolist(),
                    }
                    market_state_name = prediction_result.market_state.state_name

                pred_data = {
                    "signal": prediction_result.signal,
                    "signal_strength": round(
                        float(prediction_result.signal_strength), 3
                    ),
                    "predicted_returns": prediction_result.predicted_returns.tolist(),
                    "predicted_prices": prediction_result.predicted_prices.tolist(),
                    "confidence_lower": prediction_result.confidence_lower.tolist(),
                    "confidence_upper": prediction_result.confidence_upper.tolist(),
                    "lstm_weight": float(prediction_result.lstm_weight),
                    "xgb_weight": float(prediction_result.xgb_weight),
                    "market_state": market_state_dict,
                }
                yield _event(
                    "predict",
                    "done",
                    80,
                    f"預測訊號: {prediction_result.signal} ({prediction_result.signal_strength:.0%})",
                    pred_data,
                )
            else:
                yield _event("predict", "error", 80, "預測返回 None")
                prediction_result = None
        except Exception as e:
            logger.error("預測失敗: %s", e)
            yield _event("predict", "error", 80, f"預測失敗: {e}")
    else:
        yield _event(
            "predict", "skipped", 80, "無可用模型，跳過 ML 預測（僅用 Agent 分析）"
        )

    # ── Step 8: AGENT ─────────────────────────────────────
    yield _event("agent", "running", 85, "Multi-Agent 分析中...")

    ml_signal = prediction_result.signal if prediction_result else "hold"
    ml_confidence = (
        float(prediction_result.signal_strength) if prediction_result else 0.0
    )

    # Sub-step progress mapping (85 → 92)
    _SUBSTEP_PROGRESS = {
        "analysts_start": (85, "4 位分析師並行分析中..."),
        "analyst_done": (None, None),  # dynamic
        "debate_start": (88, "研究員辯論開始..."),
        "debate_round": (None, None),  # dynamic
        "debate_synthesis": (91, "辯論綜合判斷中..."),
        "rule_engine": (91, None),  # dynamic
        "risk_check": (92, None),  # dynamic
    }

    # Track analyst_done count for incremental progress (86-88)
    _analyst_done_count = 0

    # Build fundamental summary from price data (institutional flows)
    fundamental_summary = _build_fundamental_summary(df)

    try:
        context = MarketContext(
            stock_id=stock_id,
            current_price=current_price,
            date=end_str,
            technical_summary=technical_signals,
            sentiment_summary=sentiment_data,
            fundamental_summary=fundamental_summary,
            model_predictions={
                "signal": ml_signal,
                "confidence": ml_confidence,
                "ensemble_return": float(prediction_result.predicted_returns[0])
                if prediction_result is not None
                and len(prediction_result.predicted_returns) > 0
                else 0.0,
                "lstm_return": float(
                    prediction_result.predicted_returns[0]
                    * prediction_result.lstm_weight
                )
                if prediction_result is not None
                and len(prediction_result.predicted_returns) > 0
                else 0.0,
                "xgb_return": float(
                    prediction_result.predicted_returns[0]
                    * prediction_result.xgb_weight
                )
                if prediction_result is not None
                and len(prediction_result.predicted_returns) > 0
                else 0.0,
            },
        )

        # Launch orchestrator as background task, drain queue for sub-events
        progress_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        orchestrator = AgentOrchestrator()

        async def _run_orchestrator():
            return await orchestrator.run_analysis(
                context=context,
                ml_signal=ml_signal,
                ml_confidence=ml_confidence,
                market_state=market_state_name
                if prediction_result and prediction_result.market_state
                else None,
                progress_queue=progress_queue,
            )

        task = asyncio.create_task(_run_orchestrator())

        def _map_substep_event(substep, evt, analyst_count):
            """Map a substep event to (progress, message, data) or None to skip."""
            mapping = _SUBSTEP_PROGRESS.get(substep)
            if mapping is None:
                return None

            base_progress, base_message = mapping

            if substep == "analyst_done":
                prog = 85 + min(analyst_count, 4)  # 86, 87, 88, 89
                role = evt.get("role", "")
                sig = evt.get("signal", "")
                conf = evt.get("confidence", 0)
                role_label = {
                    "technical": "技術分析師",
                    "sentiment": "情緒分析師",
                    "fundamental": "基本面分析師",
                    "quant": "量化分析師",
                }.get(role, role)
                if sig == "error":
                    msg = f"{role_label}: 分析失敗"
                else:
                    sig_label = {
                        "buy": "BUY",
                        "strong_buy": "BUY",
                        "sell": "SELL",
                        "strong_sell": "SELL",
                        "hold": "HOLD",
                    }.get(sig, sig.upper())
                    msg = f"{role_label}: {sig_label} ({int(conf * 100)}%)"
                return (prog, msg)

            elif substep == "debate_round":
                rnd = evt.get("round", 1)
                prog = 89 + min(rnd - 1, 1)  # 89, 90
                msg = f"辯論 Round {rnd} 完成"
                return (min(prog, 90), msg)

            elif substep == "rule_engine":
                action = evt.get("action", "hold")
                action_label = {"buy": "買進", "sell": "賣出", "hold": "持有"}.get(
                    action, action
                )
                return (91, f"規則引擎: {action_label}")

            elif substep == "risk_check":
                approved = evt.get("approved", True)
                msg = (
                    "風控: 通過" if approved else f"風控: 否決 — {evt.get('notes', '')}"
                )
                return (92, msg)

            else:
                # Generic mapping (analysts_start, debate_start, debate_synthesis)
                if base_progress is not None and base_message is not None:
                    return (base_progress, base_message)
                return None

        # Drain queue until __done__ sentinel or task completes
        while True:
            # Check if task finished (with or without error)
            if task.done():
                # Drain and yield any remaining queued events
                while not progress_queue.empty():
                    try:
                        evt = progress_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    substep = evt.get("substep", "")
                    if substep == "__done__":
                        break
                    if substep == "analyst_done":
                        _analyst_done_count += 1
                    mapped = _map_substep_event(substep, evt, _analyst_done_count)
                    if mapped:
                        yield _event(
                            "agent",
                            "running",
                            mapped[0],
                            mapped[1],
                            {"substep": substep, **evt},
                        )
                break

            try:
                evt = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
            except (asyncio.TimeoutError, TimeoutError):
                continue

            substep = evt.get("substep", "")
            if substep == "__done__":
                break

            if substep == "analyst_done":
                _analyst_done_count += 1

            mapped = _map_substep_event(substep, evt, _analyst_done_count)
            if mapped:
                yield _event(
                    "agent",
                    "running",
                    mapped[0],
                    mapped[1],
                    {"substep": substep, **evt},
                )

        # Await task result (may raise)
        agent_decision = await task

        # 轉換 Agent 結果
        analyst_reports = []
        for msg in agent_decision.analyst_reports:
            analyst_reports.append(
                {
                    "role": msg.sender.value,
                    "signal": msg.signal.value if msg.signal else None,
                    "confidence": msg.confidence,
                    "reasoning": msg.reasoning,
                }
            )

        researcher = None
        if agent_decision.researcher_report:
            r = agent_decision.researcher_report
            researcher = {
                "role": r.sender.value,
                "signal": r.signal.value if r.signal else None,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
            }

        agent_data = {
            "action": agent_decision.action,
            "confidence": agent_decision.confidence,
            "position_size": agent_decision.position_size,
            "reasoning": agent_decision.reasoning,
            "approved": agent_decision.approved_by_risk,
            "risk_notes": agent_decision.risk_notes,
            "analyst_reports": analyst_reports,
            "researcher": researcher,
        }
        yield _event(
            "agent",
            "done",
            92,
            f"Agent 建議: {agent_decision.action} ({agent_decision.confidence:.0%})",
            agent_data,
        )
    except Exception as e:
        logger.error("Agent 分析失敗: %s", e)
        yield _event("agent", "error", 92, f"Agent 分析失敗: {e}")
        agent_data = None

    # ── Step 9: SYNTHESIZE ────────────────────────────────
    yield _event("synthesize", "running", 95, "彙整最終結果...")

    # 最終結果
    final_signal = "hold"
    final_confidence = 0.0
    final_reasoning = ""

    if agent_decision:
        final_signal = agent_decision.action
        final_confidence = agent_decision.confidence
        final_reasoning = agent_decision.reasoning
    elif prediction_result:
        final_signal = prediction_result.signal
        final_confidence = float(prediction_result.signal_strength)

    # 預測價格
    predicted_change = 0.0
    target_price = current_price
    if prediction_result and len(prediction_result.predicted_prices) > 0:
        target_price = float(prediction_result.predicted_prices[-1])
        predicted_change = (target_price - current_price) / current_price

    synthesis = {
        "stock_id": stock_id,
        "stock_name": stock_name,
        "current_price": current_price,
        "signal": final_signal,
        "confidence": round(final_confidence, 3),
        "target_price": round(target_price, 2),
        "predicted_change": round(predicted_change, 4),
        "reasoning": final_reasoning,
        "technical": technical_data,
        "sentiment": sentiment_data,
        "prediction": {
            "signal": prediction_result.signal if prediction_result else None,
            "signal_strength": float(prediction_result.signal_strength)
            if prediction_result
            else 0,
            "predicted_prices": prediction_result.predicted_prices.tolist()
            if prediction_result
            else [],
            "confidence_lower": prediction_result.confidence_lower.tolist()
            if prediction_result
            else [],
            "confidence_upper": prediction_result.confidence_upper.tolist()
            if prediction_result
            else [],
        }
        if prediction_result
        else None,
        "agent": agent_data,
    }

    # Persist prediction result to DB
    try:
        await asyncio.to_thread(
            save_pipeline_result,
            stock_id=stock_id,
            stock_name=stock_name,
            current_price=current_price,
            signal=final_signal,
            confidence=final_confidence,
            target_price=target_price,
            predicted_change=predicted_change,
            reasoning=final_reasoning,
            agent_decision=agent_data,
            technical_data=technical_data,
            sentiment_data={
                k: v
                for k, v in sentiment_data.items()
                if k not in ("news_headlines", "global_context")
            },
            prediction_data=synthesis.get("prediction"),
        )
        logger.info("Pipeline result saved for %s", stock_id)
    except Exception as e:
        logger.error("儲存預測結果失敗: %s", e)

    yield _event("synthesize", "done", 100, "分析完成", synthesis)
