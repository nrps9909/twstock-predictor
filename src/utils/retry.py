"""指數退避重試工具

Bug 5 fix: API 呼叫加入重試機制，區分暫時性/永久性錯誤
"""

import functools
import logging
import time
from typing import Callable, TypeVar

import requests
import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 暫時性 HTTP 錯誤碼（應重試）
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class PermanentError(Exception):
    """不可重試的永久性錯誤"""
    pass


def is_transient(exc: Exception) -> bool:
    """判斷錯誤是否為暫時性（應重試）"""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout,
                        httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in TRANSIENT_STATUS_CODES
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in TRANSIENT_STATUS_CODES
    return False


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
):
    """指數退避重試裝飾器

    Args:
        max_retries: 最大重試次數
        base_delay: 初始延遲秒數
        max_delay: 最大延遲秒數
        backoff_factor: 退避倍率

    Usage:
        @retry_with_backoff(max_retries=3)
        def fetch_data():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exc = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except PermanentError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    if attempt >= max_retries:
                        break
                    if not is_transient(exc):
                        logger.warning(
                            "%s: 非暫時性錯誤，不重試: %s", func.__name__, exc
                        )
                        break

                    logger.info(
                        "%s: 暫時性錯誤 (attempt %d/%d), %.1fs 後重試: %s",
                        func.__name__, attempt + 1, max_retries, delay, exc
                    )
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


class RateLimiter:
    """簡單速率限制器 — token bucket"""

    def __init__(self, calls_per_second: float = 2.0):
        self.min_interval = 1.0 / calls_per_second
        self._last_call = 0.0

    def wait(self):
        """必要時等待至可呼叫"""
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()
