"""警報與通知模組

支援：LINE Notify / Telegram Bot
"""

import logging
from enum import Enum

import httpx


logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SIGNAL = "signal"


class AlertManager:
    """警報管理器"""

    def __init__(
        self,
        line_token: str | None = None,
        telegram_token: str | None = None,
        telegram_chat_id: str | None = None,
    ):
        self.line_token = line_token
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id

    async def send(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
    ):
        """發送通知到所有已設定的管道"""
        prefix = {
            AlertLevel.INFO: "",
            AlertLevel.WARNING: "[Warning] ",
            AlertLevel.CRITICAL: "[CRITICAL] ",
            AlertLevel.SIGNAL: "[SIGNAL] ",
        }[level]

        full_message = f"{prefix}{message}"

        if self.line_token:
            await self._send_line(full_message)
        if self.telegram_token and self.telegram_chat_id:
            await self._send_telegram(full_message)
        if not self.line_token and not self.telegram_token:
            logger.info("Alert (no channel): %s", full_message)

    async def _send_line(self, message: str):
        """LINE Notify"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://notify-api.line.me/api/notify",
                    headers={"Authorization": f"Bearer {self.line_token}"},
                    data={"message": message},
                    timeout=10,
                )
                if resp.status_code != 200:
                    logger.warning("LINE 通知失敗: %d", resp.status_code)
        except Exception as e:
            logger.error("LINE 通知錯誤: %s", e)

    async def _send_telegram(self, message: str):
        """Telegram Bot"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                    json={
                        "chat_id": self.telegram_chat_id,
                        "text": message,
                        "parse_mode": "Markdown",
                    },
                    timeout=10,
                )
                if resp.status_code != 200:
                    logger.warning("Telegram 通知失敗: %d", resp.status_code)
        except Exception as e:
            logger.error("Telegram 通知錯誤: %s", e)

    async def send_trade_alert(
        self,
        stock_id: str,
        action: str,
        price: float,
        reasoning: str,
    ):
        """交易警報"""
        message = (
            f"交易信號 {stock_id}\n動作: {action}\n價格: {price:.2f}\n理由: {reasoning}"
        )
        await self.send(message, AlertLevel.SIGNAL)

    async def send_drawdown_alert(
        self,
        current_drawdown: float,
        max_allowed: float,
    ):
        """回撤警報"""
        message = (
            f"回撤警報!\n當前回撤: {current_drawdown:.2%}\n允許上限: {max_allowed:.2%}"
        )
        level = (
            AlertLevel.CRITICAL
            if abs(current_drawdown) > abs(max_allowed) * 0.8
            else AlertLevel.WARNING
        )
        await self.send(message, level)

    async def send_system_alert(self, error: str):
        """系統異常警報"""
        await self.send(f"系統異常: {error}", AlertLevel.CRITICAL)
