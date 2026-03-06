"""警報偵測服務 — 比對掃描結果，產生警報"""

import asyncio
import logging
from datetime import date

from src.db.database import get_previous_market_scan, save_alerts
from src.utils.constants import STOCK_LIST

logger = logging.getLogger(__name__)

# Key stocks (STOCK_LIST) get institutional surge checks
KEY_STOCK_IDS = set(STOCK_LIST.keys())


def generate_alerts_from_scan(
    current_results: list[dict],
    scan_date: date | None = None,
) -> list[dict]:
    """比對當前掃描結果與前次掃描，偵測 5 種警報

    Alert types:
    - signal_change: 訊號翻轉 (hold/sell -> buy 或反向)
    - strong_signal: 分數 > 0.75 (strong_buy) 或 < 0.25 (strong_sell)
    - institutional_surge: 法人 5 日淨買超 > 3000 張 (key stocks)
    - sync_buy: 外資+投信同步買超 (key stocks)
    - score_jump: 分數變動 > 0.15

    Returns:
        list of alert dicts ready for save_alerts()
    """
    if not current_results:
        return []

    scan_date = scan_date or date.today()

    # Get previous scan for comparison
    previous = get_previous_market_scan(scan_date)
    prev_map = {r["stock_id"]: r for r in previous}

    alerts = []

    for stock in current_results:
        sid = stock.get("stock_id", "")
        sname = stock.get("stock_name", sid)
        signal = stock.get("signal", "hold")
        score = stock.get("total_score", 0.5)
        prev = prev_map.get(sid)
        prev_signal = prev["signal"] if prev else None
        prev_score = prev["total_score"] if prev else None

        # 1. Signal change
        if prev_signal and signal != prev_signal:
            buy_signals = {"buy", "strong_buy"}
            sell_signals = {"sell", "strong_sell"}

            # Meaningful flips: sell->buy or buy->sell or hold->strong
            is_flip = (
                prev_signal in sell_signals | {"hold"} and signal in buy_signals
            ) or (prev_signal in buy_signals | {"hold"} and signal in sell_signals)
            if is_flip:
                direction = "買進" if signal in buy_signals else "賣出"
                alerts.append(
                    {
                        "alert_date": scan_date,
                        "stock_id": sid,
                        "stock_name": sname,
                        "alert_type": "signal_change",
                        "severity": "high",
                        "title": f"{sid} {sname} 轉為{direction}訊號",
                        "detail": f"訊號從 {prev_signal} 變更為 {signal}",
                        "current_signal": signal,
                        "previous_signal": prev_signal,
                        "current_score": score,
                        "previous_score": prev_score,
                    }
                )

        # 2. Strong signal
        if score > 0.75:
            alerts.append(
                {
                    "alert_date": scan_date,
                    "stock_id": sid,
                    "stock_name": sname,
                    "alert_type": "strong_signal",
                    "severity": "high",
                    "title": f"{sid} {sname} 強烈買進 ({score:.2f})",
                    "detail": f"總分 {score:.2f} 超過 0.75 門檻",
                    "current_signal": signal,
                    "current_score": score,
                }
            )
        elif score < 0.25:
            alerts.append(
                {
                    "alert_date": scan_date,
                    "stock_id": sid,
                    "stock_name": sname,
                    "alert_type": "strong_signal",
                    "severity": "high",
                    "title": f"{sid} {sname} 強烈賣出 ({score:.2f})",
                    "detail": f"總分 {score:.2f} 低於 0.25 門檻",
                    "current_signal": signal,
                    "current_score": score,
                }
            )

        # 3. Institutional surge (key stocks only)
        if sid in KEY_STOCK_IDS:
            foreign_net = stock.get("foreign_net_5d", 0)
            if abs(foreign_net) > 3000:
                direction = "買超" if foreign_net > 0 else "賣超"
                alerts.append(
                    {
                        "alert_date": scan_date,
                        "stock_id": sid,
                        "stock_name": sname,
                        "alert_type": "institutional_surge",
                        "severity": "medium",
                        "title": f"{sid} 外資 5 日{direction} {foreign_net:+,.0f} 張",
                        "detail": f"外資 5 日累計淨額 {foreign_net:+,.0f} 張",
                        "current_signal": signal,
                        "current_score": score,
                    }
                )

        # 4. Sync buy (key stocks only)
        if sid in KEY_STOCK_IDS:
            foreign_net = stock.get("foreign_net_5d", 0)
            trust_net = stock.get("trust_net_5d", 0)
            if foreign_net > 0 and trust_net > 0:
                alerts.append(
                    {
                        "alert_date": scan_date,
                        "stock_id": sid,
                        "stock_name": sname,
                        "alert_type": "sync_buy",
                        "severity": "medium",
                        "title": f"{sid} {sname} 外資投信同步買超",
                        "detail": f"外資 {foreign_net:+,.0f} 張, 投信 {trust_net:+,.0f} 張",
                        "current_signal": signal,
                        "current_score": score,
                    }
                )

        # 5. Score jump
        if prev_score is not None:
            score_diff = score - prev_score
            if abs(score_diff) > 0.15:
                direction = "上升" if score_diff > 0 else "下降"
                alerts.append(
                    {
                        "alert_date": scan_date,
                        "stock_id": sid,
                        "stock_name": sname,
                        "alert_type": "score_jump",
                        "severity": "low",
                        "title": f"{sid} {sname} 分數大幅{direction} {score_diff:+.2f}",
                        "detail": f"分數從 {prev_score:.2f} 變為 {score:.2f} ({score_diff:+.2f})",
                        "current_signal": signal,
                        "previous_signal": prev_signal,
                        "current_score": score,
                        "previous_score": prev_score,
                    }
                )

    # Deduplicate: keep highest severity per stock per type
    seen = set()
    deduped = []
    for a in alerts:
        key = (a["stock_id"], a["alert_type"])
        if key not in seen:
            seen.add(key)
            deduped.append(a)

    # Save to DB
    try:
        save_alerts(deduped)
        logger.info("Generated %d alerts for scan date %s", len(deduped), scan_date)
    except Exception as e:
        logger.error("Failed to save alerts: %s", e)

    # Trigger reanalysis for HIGH severity alerts (non-blocking)
    high_alert_stocks = list(
        {a["stock_id"] for a in deduped if a.get("severity") == "high"}
    )
    if high_alert_stocks:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                from api.services.auto_pipeline_service import trigger_reanalysis

                asyncio.ensure_future(trigger_reanalysis(high_alert_stocks))
                logger.info(
                    "Triggered reanalysis for %d high-severity stocks",
                    len(high_alert_stocks),
                )
        except Exception as e:
            logger.warning("Failed to trigger reanalysis: %s", e)

    return deduped
