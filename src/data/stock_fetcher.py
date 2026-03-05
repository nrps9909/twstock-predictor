"""股價資料抓取模組 — FinMind API + twstock"""

from datetime import date, timedelta
import logging

import pandas as pd
import requests

from src.utils.config import settings
from src.utils.constants import STOCK_LIST
from src.utils.retry import retry_with_backoff, RateLimiter

logger = logging.getLogger(__name__)

# 共用速率限制器（FinMind 免費方案限制）
_rate_limiter = RateLimiter(calls_per_second=2.0)


class StockFetcher:
    """台股資料抓取器，主要使用 FinMind API"""

    def __init__(self, token: str | None = None):
        self.token = token or settings.FINMIND_TOKEN
        self.base_url = settings.FINMIND_BASE_URL

    # ── FinMind 通用查詢 ─────────────────────────────────

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _query_finmind(
        self, dataset: str, stock_id: str, start: str, end: str
    ) -> pd.DataFrame:
        """FinMind API 通用查詢（含重試 + 速率限制）"""
        _rate_limiter.wait()

        params = {
            "dataset": dataset,
            "data_id": stock_id,
            "start_date": start,
            "end_date": end,
            "token": self.token,
        }
        resp = requests.get(self.base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != 200:
            logger.warning("FinMind API 回傳非 200: %s", data.get("msg"))
            return pd.DataFrame()
        return pd.DataFrame(data.get("data", []))

    # ── 日K線資料 ────────────────────────────────────────

    def fetch_daily_prices(
        self, stock_id: str, start: str, end: str
    ) -> pd.DataFrame:
        """抓取日K線資料

        Args:
            stock_id: 股票代號（如 "2330"）
            start: 開始日期 "YYYY-MM-DD"
            end: 結束日期 "YYYY-MM-DD"

        Returns:
            DataFrame(date, open, high, low, close, volume)
        """
        try:
            df = self._query_finmind(
                "TaiwanStockPrice", stock_id, start, end
            )
        except Exception as e:
            logger.error("抓取 %s 日K線失敗: %s", stock_id, e)
            return pd.DataFrame()

        if df.empty:
            return df

        df = df.rename(columns={
            "Trading_Volume": "volume",
            "Trading_money": "trading_money",
            "open": "open",
            "max": "high",
            "min": "low",
            "close": "close",
        })
        df["date"] = pd.to_datetime(df["date"]).dt.date
        # FinMind volume 單位為股，轉為張（÷1000）
        if "volume" in df.columns:
            df["volume"] = df["volume"] / 1000

        return df[["date", "open", "high", "low", "close", "volume"]]

    # ── 三大法人買賣超 ───────────────────────────────────

    def fetch_institutional(
        self, stock_id: str, start: str, end: str
    ) -> pd.DataFrame:
        """抓取三大法人買賣超資料

        Returns:
            DataFrame(date, foreign_buy_sell, trust_buy_sell, dealer_buy_sell)
        """
        try:
            df = self._query_finmind(
                "TaiwanStockInstitutionalInvestorsBuySell", stock_id, start, end
            )
        except Exception as e:
            logger.error("抓取 %s 法人買賣超失敗: %s", stock_id, e)
            return pd.DataFrame()

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"]).dt.date

        # FinMind 回傳 buy/sell 分開欄位，需先計算淨買賣超
        if "buy" in df.columns and "sell" in df.columns:
            df["buy_sell"] = df["buy"].fillna(0) - df["sell"].fillna(0)
        elif "buy_sell" not in df.columns:
            logger.warning("找不到 buy/sell 或 buy_sell 欄位: %s", list(df.columns))
            return pd.DataFrame()

        # 彙總各法人類別
        pivot = df.pivot_table(
            index="date", columns="name", values="buy_sell", aggfunc="sum"
        ).reset_index()

        result = pd.DataFrame({"date": pivot["date"]})
        # 外資 — FinMind 欄位名稱可能為 "Foreign_Investor" 或含子分類
        foreign_cols = [c for c in pivot.columns if "Foreign" in str(c) or "外資" in str(c)]
        result["foreign_buy_sell"] = pivot[foreign_cols].sum(axis=1) if foreign_cols else 0

        trust_cols = [c for c in pivot.columns if "Investment_Trust" in str(c) or "投信" in str(c)]
        result["trust_buy_sell"] = pivot[trust_cols].sum(axis=1) if trust_cols else 0

        dealer_cols = [c for c in pivot.columns if "Dealer" in str(c) or "自營" in str(c)]
        result["dealer_buy_sell"] = pivot[dealer_cols].sum(axis=1) if dealer_cols else 0

        return result

    # ── 融資融券 ─────────────────────────────────────────

    def fetch_margin_trading(
        self, stock_id: str, start: str, end: str
    ) -> pd.DataFrame:
        """抓取融資融券資料

        Returns:
            DataFrame(date, margin_balance, short_balance)
        """
        try:
            df = self._query_finmind(
                "TaiwanStockMarginPurchaseShortSale", stock_id, start, end
            )
        except Exception as e:
            logger.error("抓取 %s 融資融券失敗: %s", stock_id, e)
            return pd.DataFrame()

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"]).dt.date
        result = pd.DataFrame({
            "date": df["date"],
            "margin_balance": df.get("MarginPurchaseTodayBalance", 0),
            "short_balance": df.get("ShortSaleTodayBalance", 0),
        })
        return result

    # ── 合併所有資料 ─────────────────────────────────────

    def fetch_all(
        self, stock_id: str, start: str, end: str
    ) -> pd.DataFrame:
        """抓取並合併日K + 法人 + 融資融券

        Returns:
            完整的 DataFrame，含所有欄位
        """
        prices = self.fetch_daily_prices(stock_id, start, end)
        if prices.empty:
            logger.warning("無法取得 %s 日K線資料", stock_id)
            return prices

        institutional = self.fetch_institutional(stock_id, start, end)
        margin = self.fetch_margin_trading(stock_id, start, end)

        df = prices
        if not institutional.empty:
            df = df.merge(institutional, on="date", how="left")
        if not margin.empty:
            df = df.merge(margin, on="date", how="left")

        return df

    # ── FinMind 每日 P/E, P/B, 殖利率 ──────────────────

    def fetch_per_pbr(
        self, stock_id: str, start: str, end: str
    ) -> pd.DataFrame:
        """FinMind TaiwanStockPER — 每日 P/E, P/B, 殖利率

        Returns:
            DataFrame(date, PER, PBR, dividend_yield)
        """
        try:
            df = self._query_finmind("TaiwanStockPER", stock_id, start, end)
        except Exception as e:
            logger.error("抓取 %s P/E P/B 失敗: %s", stock_id, e)
            return pd.DataFrame()

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"]).dt.date
        rename_map = {}
        if "PER" not in df.columns and "本益比" in df.columns:
            rename_map["本益比"] = "PER"
        if "PBR" not in df.columns and "股價淨值比" in df.columns:
            rename_map["股價淨值比"] = "PBR"
        if "dividend_yield" not in df.columns and "殖利率(%)" in df.columns:
            rename_map["殖利率(%)"] = "dividend_yield"
        elif "dividend_yield" not in df.columns and "殖利率" in df.columns:
            rename_map["殖利率"] = "dividend_yield"
        if rename_map:
            df = df.rename(columns=rename_map)

        cols = ["date"]
        for c in ["PER", "PBR", "dividend_yield"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                cols.append(c)
        return df[cols] if len(cols) > 1 else pd.DataFrame()

    def fetch_dividend_history(
        self, stock_id: str, start: str, end: str
    ) -> pd.DataFrame:
        """FinMind TaiwanStockDividendResult — 歷史除權息

        Returns:
            DataFrame with dividend result columns
        """
        try:
            df = self._query_finmind(
                "TaiwanStockDividendResult", stock_id, start, end
            )
        except Exception as e:
            logger.error("抓取 %s 除權息失敗: %s", stock_id, e)
            return pd.DataFrame()

        if df.empty:
            return df

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    # ── 即時報價（twstock） ──────────────────────────────

    @staticmethod
    def fetch_realtime(stock_id: str) -> dict | None:
        """使用 twstock 取得即時報價"""
        try:
            import twstock
            stock = twstock.realtime.get(stock_id)
            if stock.get("success"):
                info = stock["realtime"]
                return {
                    "stock_id": stock_id,
                    "name": stock["info"]["name"],
                    "price": float(info.get("latest_trade_price", 0)),
                    "open": float(info.get("open", 0)),
                    "high": float(info.get("high", 0)),
                    "low": float(info.get("low", 0)),
                    "volume": float(info.get("accumulate_trade_volume", 0)),
                    "time": info.get("latest_trade_time", ""),
                }
        except Exception as e:
            logger.error("twstock 即時報價錯誤: %s", e)
        return None

    # ── 存活者偏誤修正 ──────────────────────────────────

    @staticmethod
    def fetch_delisted_stocks() -> list[dict]:
        """取得已下市股票清單

        Returns:
            list of {"stock_id", "name", "delist_date", "reason", "merged_into"}
        """
        from src.utils.constants import DELISTED_STOCKS
        return [
            {"stock_id": k, **v}
            for k, v in DELISTED_STOCKS.items()
        ]

    # ── As-of 時間戳 ──────────────────────────────────

    def fetch_all_with_as_of(
        self, stock_id: str, start: str, end: str,
    ) -> pd.DataFrame:
        """抓取資料並標記 as_of_date = today"""
        df = self.fetch_all(stock_id, start, end)
        if not df.empty:
            df["as_of_date"] = date.today()
        return df

    # ── 工具方法 ─────────────────────────────────────────

    @staticmethod
    def get_stock_name(stock_id: str) -> str:
        """取得股票中文名稱"""
        return STOCK_LIST.get(stock_id, stock_id)
