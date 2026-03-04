"""資料庫連線與 CRUD 操作"""

import json
from datetime import date, timedelta
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, select, delete, desc, text
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Base, StockPrice, SentimentRecord, Prediction, TradeJournal, MarketScanResult, Alert, PipelineResult, FactorICRecord
from src.utils.config import settings


engine = create_engine(settings.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


_MARKET_SCAN_MIGRATIONS = [
    ("institutional_flow_score", "FLOAT"),
    ("margin_retail_score", "FLOAT"),
    ("volatility_score", "FLOAT"),
    ("liquidity_score", "FLOAT"),
    ("value_quality_score", "FLOAT"),
    ("confidence_agreement", "FLOAT"),
    ("confidence_strength", "FLOAT"),
    ("confidence_coverage", "FLOAT"),
    ("confidence_freshness", "FLOAT"),
    ("risk_discount", "FLOAT"),
    ("market_regime", "VARCHAR(20)"),
    ("factor_details", "JSON"),
]


def _migrate_market_scans():
    """Add missing columns to market_scans table (safe, idempotent)."""
    with engine.connect() as conn:
        # Check if table exists first
        tables = {row[0] for row in conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ))}
        if "market_scans" not in tables:
            return

        # Get existing columns
        result = conn.execute(text("PRAGMA table_info(market_scans)"))
        existing = {row[1] for row in result}

        for col_name, col_type in _MARKET_SCAN_MIGRATIONS:
            if col_name not in existing:
                conn.execute(text(
                    f"ALTER TABLE market_scans ADD COLUMN {col_name} {col_type}"
                ))
        conn.commit()


def init_db():
    """建立所有資料表"""
    Base.metadata.create_all(engine)
    _migrate_market_scans()


@contextmanager
def get_session():
    """取得 DB session（context manager）"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Stock Prices ─────────────────────────────────────────


def upsert_stock_prices(df: pd.DataFrame, stock_id: str):
    """批次寫入/更新日K線資料

    df 欄位: date, open, high, low, close, volume,
             foreign_buy_sell, trust_buy_sell, dealer_buy_sell,
             margin_balance, short_balance  (後五欄可選)
    """
    with get_session() as session:
        for _, row in df.iterrows():
            existing = session.execute(
                select(StockPrice).where(
                    StockPrice.stock_id == stock_id,
                    StockPrice.date == row["date"],
                )
            ).scalar_one_or_none()

            if existing:
                for col in [
                    "open", "high", "low", "close", "volume",
                    "foreign_buy_sell", "trust_buy_sell", "dealer_buy_sell",
                    "margin_balance", "short_balance",
                ]:
                    if col in row and pd.notna(row.get(col)):
                        setattr(existing, col, row[col])
            else:
                record = StockPrice(
                    stock_id=stock_id,
                    date=row["date"],
                    open=row.get("open"),
                    high=row.get("high"),
                    low=row.get("low"),
                    close=row.get("close"),
                    volume=row.get("volume"),
                    foreign_buy_sell=row.get("foreign_buy_sell"),
                    trust_buy_sell=row.get("trust_buy_sell"),
                    dealer_buy_sell=row.get("dealer_buy_sell"),
                    margin_balance=row.get("margin_balance"),
                    short_balance=row.get("short_balance"),
                    as_of_date=row.get("as_of_date", date.today()),
                )
                session.add(record)


def get_stock_prices(
    stock_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """讀取日K線資料，回傳 DataFrame"""
    with get_session() as session:
        stmt = select(StockPrice).where(StockPrice.stock_id == stock_id)
        if start_date:
            stmt = stmt.where(StockPrice.date >= start_date)
        if end_date:
            stmt = stmt.where(StockPrice.date <= end_date)
        stmt = stmt.order_by(StockPrice.date)

        rows = session.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()

        data = [
            {
                "date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "foreign_buy_sell": r.foreign_buy_sell,
                "trust_buy_sell": r.trust_buy_sell,
                "dealer_buy_sell": r.dealer_buy_sell,
                "margin_balance": r.margin_balance,
                "short_balance": r.short_balance,
            }
            for r in rows
        ]
        return pd.DataFrame(data)


def get_stock_prices_point_in_time(
    stock_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
    as_of_date: date | None = None,
) -> pd.DataFrame:
    """Point-in-time 查詢：只取得在 as_of_date 之前可取得的資料

    避免回測中的前視偏誤。

    Args:
        stock_id: 股票代碼
        start_date: 開始日期
        end_date: 結束日期
        as_of_date: 截止可取得日期（只取 as_of_date <= 此日期的資料）

    Returns:
        DataFrame
    """
    with get_session() as session:
        stmt = select(StockPrice).where(StockPrice.stock_id == stock_id)
        if start_date:
            stmt = stmt.where(StockPrice.date >= start_date)
        if end_date:
            stmt = stmt.where(StockPrice.date <= end_date)
        if as_of_date:
            stmt = stmt.where(
                (StockPrice.as_of_date <= as_of_date) | (StockPrice.as_of_date.is_(None))
            )
        stmt = stmt.order_by(StockPrice.date)

        rows = session.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()

        data = [
            {
                "date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "foreign_buy_sell": r.foreign_buy_sell,
                "trust_buy_sell": r.trust_buy_sell,
                "dealer_buy_sell": r.dealer_buy_sell,
                "margin_balance": r.margin_balance,
                "short_balance": r.short_balance,
                "as_of_date": r.as_of_date,
            }
            for r in rows
        ]
        return pd.DataFrame(data)


# ── Sentiment ────────────────────────────────────────────


def insert_sentiment(records: list[dict]):
    """批次寫入情緒紀錄"""
    with get_session() as session:
        for rec in records:
            if "as_of_date" not in rec:
                rec = {**rec, "as_of_date": date.today()}
            session.add(SentimentRecord(**rec))


def get_sentiment(
    stock_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """讀取情緒紀錄"""
    with get_session() as session:
        stmt = select(SentimentRecord).where(
            SentimentRecord.stock_id == stock_id
        )
        if start_date:
            stmt = stmt.where(SentimentRecord.date >= start_date)
        if end_date:
            stmt = stmt.where(SentimentRecord.date <= end_date)
        stmt = stmt.order_by(SentimentRecord.date)

        rows = session.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()

        data = [
            {
                "date": r.date,
                "source": r.source,
                "title": r.title,
                "sentiment_label": r.sentiment_label,
                "sentiment_score": r.sentiment_score,
                "keywords": r.keywords,
                "engagement": r.engagement,
                "url": r.url,
            }
            for r in rows
        ]
        return pd.DataFrame(data)


# ── Predictions ──────────────────────────────────────────


def insert_predictions(records: list[dict]):
    """批次寫入預測紀錄"""
    with get_session() as session:
        for rec in records:
            session.add(Prediction(**rec))


def get_predictions(
    stock_id: str,
    prediction_date: date | None = None,
    model_type: str = "ensemble",
) -> pd.DataFrame:
    """讀取預測紀錄"""
    with get_session() as session:
        stmt = select(Prediction).where(
            Prediction.stock_id == stock_id,
            Prediction.model_type == model_type,
        )
        if prediction_date:
            stmt = stmt.where(Prediction.prediction_date == prediction_date)
        stmt = stmt.order_by(Prediction.target_date)

        rows = session.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()

        data = [
            {
                "prediction_date": r.prediction_date,
                "target_date": r.target_date,
                "predicted_price": r.predicted_price,
                "predicted_return": r.predicted_return,
                "confidence_lower": r.confidence_lower,
                "confidence_upper": r.confidence_upper,
                "actual_price": r.actual_price,
                "signal": r.signal,
            }
            for r in rows
        ]
        return pd.DataFrame(data)


# ── Pipeline Result Persistence ─────────────────────────


def save_pipeline_result(
    stock_id: str,
    stock_name: str,
    current_price: float,
    signal: str,
    confidence: float,
    target_price: float,
    predicted_change: float,
    reasoning: str,
    agent_decision: dict | None = None,
    technical_data: dict | None = None,
    sentiment_data: dict | None = None,
    prediction_data: dict | None = None,
) -> int:
    """儲存 pipeline 完整結果到 Prediction + TradeJournal

    Returns:
        prediction ID
    """
    today = date.today()
    target_date = today + timedelta(days=5)

    with get_session() as session:
        # 1. Write Prediction record
        pred = Prediction(
            stock_id=stock_id,
            prediction_date=today,
            target_date=target_date,
            predicted_price=target_price,
            predicted_return=predicted_change,
            confidence_lower=target_price * (1 - max(0.02, abs(predicted_change))),
            confidence_upper=target_price * (1 + max(0.02, abs(predicted_change))),
            actual_price=None,
            model_type="pipeline",
            signal=signal,
        )
        session.add(pred)
        session.flush()  # get pred.id
        pred_id = pred.id

        # 2. Write TradeJournal record
        analyst_reports = agent_decision.get("analyst_reports", []) if agent_decision else []
        # Extract per-role analyses
        tech_analysis = next((r for r in analyst_reports if r.get("role") == "technical"), None)
        sent_analysis = next((r for r in analyst_reports if r.get("role") == "sentiment"), None)
        fund_analysis = next((r for r in analyst_reports if r.get("role") == "fundamental"), None)
        quant_analysis = next((r for r in analyst_reports if r.get("role") == "quant"), None)

        journal = TradeJournal(
            stock_id=stock_id,
            trade_date=today,
            action=signal,
            price=current_price,
            position_size=agent_decision.get("position_size", 0) if agent_decision else 0,
            technical_analysis=tech_analysis,
            sentiment_analysis=sent_analysis,
            fundamental_analysis=fund_analysis,
            quant_analysis=quant_analysis,
            researcher_debate=agent_decision.get("researcher") if agent_decision else None,
            trader_reasoning=reasoning,
            risk_assessment={
                "approved": agent_decision.get("approved", False),
                "risk_notes": agent_decision.get("risk_notes", ""),
            } if agent_decision else None,
            market_snapshot={
                "current_price": current_price,
                "target_price": target_price,
                "confidence": confidence,
                "technical": technical_data,
                "sentiment": {k: v for k, v in (sentiment_data or {}).items()
                              if k not in ("news_headlines", "global_context")},
                "prediction": prediction_data,
            },
        )
        session.add(journal)

    return pred_id


def get_prediction_history(
    stock_id: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """取得預測歷史紀錄（結合 Prediction + TradeJournal）

    Returns:
        list of dicts with prediction + journal data
    """
    with get_session() as session:
        stmt = select(Prediction).where(Prediction.model_type == "pipeline")
        if stock_id:
            stmt = stmt.where(Prediction.stock_id == stock_id)
        stmt = stmt.order_by(desc(Prediction.prediction_date)).limit(limit)

        predictions = session.execute(stmt).scalars().all()
        if not predictions:
            return []

        results = []
        for p in predictions:
            # Find matching journal entry
            j_stmt = select(TradeJournal).where(
                TradeJournal.stock_id == p.stock_id,
                TradeJournal.trade_date == p.prediction_date,
            ).limit(1)
            journal = session.execute(j_stmt).scalar_one_or_none()

            from src.utils.constants import STOCK_LIST
            results.append({
                "id": p.id,
                "stock_id": p.stock_id,
                "stock_name": STOCK_LIST.get(p.stock_id, p.stock_id),
                "prediction_date": str(p.prediction_date),
                "signal": p.signal,
                "confidence": float(p.confidence_upper - p.predicted_price) / p.predicted_price
                    if p.predicted_price and p.confidence_upper else 0,
                "predicted_price": p.predicted_price,
                "actual_price": p.actual_price,
                "reasoning": journal.trader_reasoning if journal else "",
                "agent_action": journal.action if journal else p.signal,
                "agent_approved": journal.risk_assessment.get("approved", False)
                    if journal and journal.risk_assessment else False,
                "analyst_reports": journal.technical_analysis
                    if journal else None,
                "market_snapshot": journal.market_snapshot if journal else None,
            })

        return results


def update_prediction_actuals(stock_id: str) -> int:
    """回填已到期預測的 actual_price

    Returns:
        更新的筆數
    """
    today = date.today()
    updated = 0

    with get_session() as session:
        # Find predictions with target_date <= today and no actual_price
        stmt = select(Prediction).where(
            Prediction.stock_id == stock_id,
            Prediction.target_date <= today,
            Prediction.actual_price.is_(None),
        )
        preds = session.execute(stmt).scalars().all()

        for p in preds:
            # Get actual price on target_date (or closest prior)
            price_stmt = select(StockPrice).where(
                StockPrice.stock_id == stock_id,
                StockPrice.date <= p.target_date,
            ).order_by(desc(StockPrice.date)).limit(1)

            price_row = session.execute(price_stmt).scalar_one_or_none()
            if price_row and price_row.close:
                p.actual_price = price_row.close
                updated += 1

    return updated


# ── Market Scan ─────────────────────────────────────────


def save_market_scan(results: list[dict]):
    """批次寫入市場掃描結果（同一天的結果會覆蓋）"""
    if not results:
        return

    scan_date = results[0].get("scan_date", date.today())

    with get_session() as session:
        # Delete existing results for this scan_date
        session.execute(
            delete(MarketScanResult).where(MarketScanResult.scan_date == scan_date)
        )
        session.flush()

        for rec in results:
            session.add(MarketScanResult(**rec))


def get_latest_market_scan() -> list[dict]:
    """取得最新一次市場掃描結果

    Returns:
        list of dicts sorted by ranking
    """
    with get_session() as session:
        # Find the latest scan_date
        latest_stmt = select(MarketScanResult.scan_date).order_by(
            desc(MarketScanResult.scan_date)
        ).limit(1)
        latest_date = session.execute(latest_stmt).scalar_one_or_none()

        if not latest_date:
            return []

        stmt = select(MarketScanResult).where(
            MarketScanResult.scan_date == latest_date
        ).order_by(MarketScanResult.ranking)

        rows = session.execute(stmt).scalars().all()
        return [
            {
                "stock_id": r.stock_id,
                "stock_name": r.stock_name,
                "scan_date": str(r.scan_date),
                "current_price": r.current_price,
                "price_change_pct": r.price_change_pct,
                "signal": r.signal,
                "confidence": r.confidence,
                "total_score": r.total_score,
                "technical_score": r.technical_score,
                "fundamental_score": r.fundamental_score,
                "sentiment_score": r.sentiment_score,
                "ml_score": r.ml_score,
                "momentum_score": r.momentum_score,
                "institutional_flow_score": r.institutional_flow_score,
                "margin_retail_score": r.margin_retail_score,
                "volatility_score": r.volatility_score,
                "liquidity_score": r.liquidity_score,
                "value_quality_score": r.value_quality_score,
                "foreign_net_5d": r.foreign_net_5d,
                "trust_net_5d": r.trust_net_5d,
                "dealer_net_5d": r.dealer_net_5d,
                "ranking": r.ranking,
                "reasoning": r.reasoning,
                "score_coverage": r.score_coverage,
                "effective_coverage": r.effective_coverage,
                "confidence_agreement": r.confidence_agreement,
                "confidence_strength": r.confidence_strength,
                "confidence_coverage": r.confidence_coverage,
                "confidence_freshness": r.confidence_freshness,
                "risk_discount": r.risk_discount,
                "market_regime": r.market_regime,
                "factor_details": r.factor_details,
            }
            for r in rows
        ]


def get_all_stocks_latest_prices(stock_ids: list[str]) -> pd.DataFrame:
    """批次取得多檔股票的最新收盤價 + 近期漲跌幅

    Returns:
        DataFrame with stock_id, close, prev_close, price_change_pct
    """
    with get_session() as session:
        records = []
        for sid in stock_ids:
            stmt = select(StockPrice).where(
                StockPrice.stock_id == sid
            ).order_by(desc(StockPrice.date)).limit(2)
            rows = session.execute(stmt).scalars().all()

            if rows:
                latest = rows[0]
                prev = rows[1] if len(rows) > 1 else rows[0]
                pct = ((latest.close - prev.close) / prev.close * 100
                       if prev.close and latest.close else 0)
                records.append({
                    "stock_id": sid,
                    "close": latest.close,
                    "prev_close": prev.close,
                    "price_change_pct": round(pct, 2),
                    "latest_date": latest.date,
                })

        return pd.DataFrame(records) if records else pd.DataFrame()


# ── Alerts ─────────────────────────────────────────────


def save_alerts(alerts: list[dict]):
    """批次寫入警報紀錄（同日同股同類型去重）"""
    if not alerts:
        return
    with get_session() as session:
        for rec in alerts:
            # Check for existing alert with same date/stock/type
            existing = session.execute(
                select(Alert).where(
                    Alert.alert_date == rec["alert_date"],
                    Alert.stock_id == rec["stock_id"],
                    Alert.alert_type == rec["alert_type"],
                )
            ).scalar_one_or_none()
            if existing is None:
                session.add(Alert(**rec))


def get_alerts(
    limit: int = 50,
    unread_only: bool = False,
    severity: str | None = None,
) -> list[dict]:
    """查詢警報"""
    with get_session() as session:
        stmt = select(Alert)
        if unread_only:
            stmt = stmt.where(Alert.is_read == 0)
        if severity:
            stmt = stmt.where(Alert.severity == severity)
        stmt = stmt.order_by(desc(Alert.created_at)).limit(limit)

        rows = session.execute(stmt).scalars().all()
        return [
            {
                "id": r.id,
                "alert_date": str(r.alert_date),
                "stock_id": r.stock_id,
                "stock_name": r.stock_name,
                "alert_type": r.alert_type,
                "severity": r.severity,
                "title": r.title,
                "detail": r.detail,
                "current_signal": r.current_signal,
                "previous_signal": r.previous_signal,
                "current_score": r.current_score,
                "previous_score": r.previous_score,
                "is_read": bool(r.is_read),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


def get_unread_alert_count() -> int:
    """取得未讀警報數量"""
    with get_session() as session:
        from sqlalchemy import func
        stmt = select(func.count(Alert.id)).where(Alert.is_read == 0)
        return session.execute(stmt).scalar() or 0


def mark_alert_read(alert_id: int):
    """標記單一警報已讀"""
    with get_session() as session:
        alert = session.get(Alert, alert_id)
        if alert:
            alert.is_read = 1


def mark_all_alerts_read():
    """標記所有警報已讀"""
    with get_session() as session:
        from sqlalchemy import update
        session.execute(
            update(Alert).where(Alert.is_read == 0).values(is_read=1)
        )


def get_top_institutional_stocks(top_n: int = 50) -> list[str]:
    """從最新 MarketScanResult 取外資+投信交易量最大的 N 支股票

    Returns:
        list of stock_id strings
    """
    with get_session() as session:
        from sqlalchemy import func, case
        latest_stmt = select(MarketScanResult.scan_date).order_by(
            desc(MarketScanResult.scan_date)
        ).limit(1)
        latest_date = session.execute(latest_stmt).scalar_one_or_none()
        if not latest_date:
            return []

        # Sort by absolute sum of foreign + trust net volume
        stmt = (
            select(MarketScanResult.stock_id)
            .where(MarketScanResult.scan_date == latest_date)
            .order_by(
                desc(
                    func.abs(func.coalesce(MarketScanResult.foreign_net_5d, 0))
                    + func.abs(func.coalesce(MarketScanResult.trust_net_5d, 0))
                )
            )
            .limit(top_n)
        )
        rows = session.execute(stmt).scalars().all()
        return list(rows)


def get_previous_market_scan(before_date: date) -> list[dict]:
    """取得指定日期之前最近一次的市場掃描結果

    Args:
        before_date: 排除此日期（取更早的掃描）

    Returns:
        list of dicts
    """
    with get_session() as session:
        # Find the latest scan_date before the given date
        latest_stmt = select(MarketScanResult.scan_date).where(
            MarketScanResult.scan_date < before_date
        ).order_by(desc(MarketScanResult.scan_date)).limit(1)
        latest_date = session.execute(latest_stmt).scalar_one_or_none()

        if not latest_date:
            return []

        stmt = select(MarketScanResult).where(
            MarketScanResult.scan_date == latest_date
        )
        rows = session.execute(stmt).scalars().all()
        return [
            {
                "stock_id": r.stock_id,
                "stock_name": r.stock_name,
                "signal": r.signal,
                "total_score": r.total_score,
                "foreign_net_5d": r.foreign_net_5d,
                "trust_net_5d": r.trust_net_5d,
            }
            for r in rows
        ]


# ── Pipeline Results ─────────────────────────────────────


def save_pipeline_result_record(result: dict):
    """Upsert pipeline 分析結果 by stock_id + analysis_date"""
    stock_id = result["stock_id"]
    analysis_date = result.get("analysis_date", date.today())

    with get_session() as session:
        existing = session.execute(
            select(PipelineResult).where(
                PipelineResult.stock_id == stock_id,
                PipelineResult.analysis_date == analysis_date,
            )
        ).scalar_one_or_none()

        if existing:
            for key in [
                "signal", "confidence", "predicted_price", "reasoning",
                "agent_scores", "sentiment_summary", "news_summary",
                "technical_data", "institutional_data", "risk_approved",
                "pipeline_version",
            ]:
                if key in result:
                    setattr(existing, key, result[key])
        else:
            session.add(PipelineResult(
                stock_id=stock_id,
                analysis_date=analysis_date,
                signal=result.get("signal"),
                confidence=result.get("confidence"),
                predicted_price=result.get("predicted_price"),
                reasoning=result.get("reasoning"),
                agent_scores=result.get("agent_scores"),
                sentiment_summary=result.get("sentiment_summary"),
                news_summary=result.get("news_summary"),
                technical_data=result.get("technical_data"),
                institutional_data=result.get("institutional_data"),
                risk_approved=result.get("risk_approved"),
                pipeline_version=result.get("pipeline_version", "2.0"),
            ))


def get_pipeline_result(stock_id: str, target_date: date | None = None) -> dict | None:
    """取得個股最新 pipeline 結果"""
    with get_session() as session:
        stmt = select(PipelineResult).where(PipelineResult.stock_id == stock_id)
        if target_date:
            stmt = stmt.where(PipelineResult.analysis_date == target_date)
        stmt = stmt.order_by(desc(PipelineResult.analysis_date)).limit(1)

        row = session.execute(stmt).scalar_one_or_none()
        if not row:
            return None

        return {
            "stock_id": row.stock_id,
            "analysis_date": str(row.analysis_date),
            "signal": row.signal,
            "confidence": row.confidence,
            "predicted_price": row.predicted_price,
            "reasoning": row.reasoning,
            "agent_scores": row.agent_scores,
            "sentiment_summary": row.sentiment_summary,
            "news_summary": row.news_summary,
            "technical_data": row.technical_data,
            "institutional_data": row.institutional_data,
            "risk_approved": bool(row.risk_approved) if row.risk_approved is not None else None,
            "pipeline_version": row.pipeline_version,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }


def get_pipeline_results_batch(stock_ids: list[str], target_date: date | None = None) -> list[dict]:
    """批次取得多支股票的 pipeline 結果"""
    results = []
    for sid in stock_ids:
        r = get_pipeline_result(sid, target_date)
        if r:
            results.append(r)
    return results


# ── Factor IC Tracking ────────────────────────────────


def save_factor_ic_records(records: list[dict]):
    """批次寫入因子 IC 追蹤紀錄（upsert by record_date+stock_id+factor_name）"""
    if not records:
        return
    with get_session() as session:
        for rec in records:
            existing = session.execute(
                select(FactorICRecord).where(
                    FactorICRecord.record_date == rec["record_date"],
                    FactorICRecord.stock_id == rec["stock_id"],
                    FactorICRecord.factor_name == rec["factor_name"],
                )
            ).scalar_one_or_none()
            if existing:
                existing.factor_score = rec["factor_score"]
            else:
                session.add(FactorICRecord(**rec))


def backfill_forward_returns(lookback_days: int = 5):
    """回填因子 IC 紀錄的遠期報酬

    For records that are old enough (>= lookback_days ago),
    fill in forward_return_5d and forward_return_20d from StockPrice.
    """
    today = date.today()
    cutoff_5d = today - timedelta(days=lookback_days + 3)   # buffer for non-trading days
    cutoff_20d = today - timedelta(days=25 + 3)

    with get_session() as session:
        # Fill 5-day forward returns
        stmt = select(FactorICRecord).where(
            FactorICRecord.record_date <= cutoff_5d,
            FactorICRecord.forward_return_5d.is_(None),
        )
        records = session.execute(stmt).scalars().all()

        for rec in records:
            # Get the close price on record_date
            base_price_row = session.execute(
                select(StockPrice).where(
                    StockPrice.stock_id == rec.stock_id,
                    StockPrice.date <= rec.record_date,
                ).order_by(desc(StockPrice.date)).limit(1)
            ).scalar_one_or_none()

            if not base_price_row or not base_price_row.close:
                continue

            # Get close ~5 trading days later
            future_date = rec.record_date + timedelta(days=lookback_days + 3)
            future_row = session.execute(
                select(StockPrice).where(
                    StockPrice.stock_id == rec.stock_id,
                    StockPrice.date > rec.record_date,
                    StockPrice.date <= future_date,
                ).order_by(desc(StockPrice.date)).limit(1)
            ).scalar_one_or_none()

            if future_row and future_row.close:
                rec.forward_return_5d = (future_row.close - base_price_row.close) / base_price_row.close

        # Fill 20-day forward returns
        stmt_20d = select(FactorICRecord).where(
            FactorICRecord.record_date <= cutoff_20d,
            FactorICRecord.forward_return_20d.is_(None),
        )
        records_20d = session.execute(stmt_20d).scalars().all()

        for rec in records_20d:
            base_price_row = session.execute(
                select(StockPrice).where(
                    StockPrice.stock_id == rec.stock_id,
                    StockPrice.date <= rec.record_date,
                ).order_by(desc(StockPrice.date)).limit(1)
            ).scalar_one_or_none()

            if not base_price_row or not base_price_row.close:
                continue

            future_date = rec.record_date + timedelta(days=30)
            future_row = session.execute(
                select(StockPrice).where(
                    StockPrice.stock_id == rec.stock_id,
                    StockPrice.date > rec.record_date,
                    StockPrice.date <= future_date,
                ).order_by(desc(StockPrice.date)).limit(1)
            ).scalar_one_or_none()

            if future_row and future_row.close:
                rec.forward_return_20d = (future_row.close - base_price_row.close) / base_price_row.close


def get_factor_ic_rolling(factor_name: str, window: int = 60) -> dict:
    """計算因子滾動 Spearman IC

    Args:
        factor_name: 因子名稱
        window: 滾動窗口大小（交易日）

    Returns:
        dict with ic_mean, ic_std, icir, ic_series
    """
    from scipy import stats

    with get_session() as session:
        stmt = select(FactorICRecord).where(
            FactorICRecord.factor_name == factor_name,
            FactorICRecord.forward_return_5d.isnot(None),
        ).order_by(FactorICRecord.record_date)

        rows = session.execute(stmt).scalars().all()
        if not rows:
            return {"factor": factor_name, "ic_mean": 0, "ic_std": 0, "icir": 0, "ic_series": []}

        # Group by date
        date_groups: dict[date, list[tuple[float, float]]] = {}
        for r in rows:
            d = r.record_date
            if d not in date_groups:
                date_groups[d] = []
            date_groups[d].append((r.factor_score, r.forward_return_5d))

        # Compute cross-sectional Spearman IC per date
        ic_series = []
        sorted_dates = sorted(date_groups.keys())
        for d in sorted_dates:
            pairs = date_groups[d]
            if len(pairs) < 5:
                continue
            scores = [p[0] for p in pairs]
            returns = [p[1] for p in pairs]
            corr, _ = stats.spearmanr(scores, returns)
            if not pd.isna(corr):
                ic_series.append({"date": str(d), "ic": round(corr, 4)})

        if not ic_series:
            return {"factor": factor_name, "ic_mean": 0, "ic_std": 0, "icir": 0, "ic_series": []}

        # Rolling stats over the window
        recent = ic_series[-window:]
        ics = [x["ic"] for x in recent]
        ic_mean = sum(ics) / len(ics) if ics else 0
        ic_std = (sum((x - ic_mean) ** 2 for x in ics) / len(ics)) ** 0.5 if len(ics) > 1 else 0
        icir = ic_mean / ic_std if ic_std > 0 else 0

        return {
            "factor": factor_name,
            "ic_mean": round(ic_mean, 4),
            "ic_std": round(ic_std, 4),
            "icir": round(icir, 4),
            "ic_series": ic_series,
        }
