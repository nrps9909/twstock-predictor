"""短期記憶系統

保留最近 N 個交易日的市場資料 + 決策 (in-memory)。
"""

import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """短期記憶 — 最近 N 個交易日（in-memory）"""

    def __init__(self, max_days: int = 5):
        self.max_days = max_days
        self._memories: deque[dict[str, Any]] = deque(maxlen=max_days)

    def add(self, date: str, data: dict[str, Any]):
        """新增一天的記憶"""
        self._memories.append({"date": date, **data})

    def get_recent(self, n: int | None = None) -> list[dict]:
        """取得最近 N 天的記憶"""
        memories = list(self._memories)
        if n is not None:
            return memories[-n:]
        return memories

    def get_summary(self) -> dict:
        """取得摘要"""
        if not self._memories:
            return {"status": "empty"}
        return {
            "days": len(self._memories),
            "date_range": f"{self._memories[0]['date']} ~ {self._memories[-1]['date']}",
            "latest": self._memories[-1],
        }

    def clear(self):
        self._memories.clear()
