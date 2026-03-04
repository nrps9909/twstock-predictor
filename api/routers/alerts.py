"""警報 API — 查詢、標記已讀"""

from fastapi import APIRouter

from src.db.database import (
    get_alerts,
    get_unread_alert_count,
    mark_alert_read,
    mark_all_alerts_read,
)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("")
async def list_alerts(
    limit: int = 50,
    unread_only: bool = False,
    severity: str | None = None,
):
    """取得最近警報"""
    return get_alerts(limit=limit, unread_only=unread_only, severity=severity)


@router.get("/unread-count")
async def unread_count():
    """未讀警報數量"""
    return {"count": get_unread_alert_count()}


@router.patch("/{alert_id}/read")
async def read_alert(alert_id: int):
    """標記單一警報已讀"""
    mark_alert_read(alert_id)
    return {"ok": True}


@router.patch("/read-all")
async def read_all_alerts():
    """全部已讀"""
    mark_all_alerts_read()
    return {"ok": True}
