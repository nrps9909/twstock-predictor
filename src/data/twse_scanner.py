"""TWSE 三大法人買賣超掃描器 — T86 全市場資料

從證交所 API 取得三大法人每日買賣超資料，
篩選投信連續買超 / 累計買超前 N 名，建立動態股票母體。
"""

import logging
import time
import urllib3
from datetime import date, timedelta

import pandas as pd
import requests

from src.utils.constants import STOCK_LIST
from src.utils.retry import retry_with_backoff, RateLimiter

logger = logging.getLogger(__name__)

# TWSE 憑證有已知的 Subject Key Identifier 問題，需跳過驗證
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_rate_limiter = RateLimiter(calls_per_second=1.0)

# TTL cache for TWSE daily data — avoid redundant fetches within 5 minutes
_t86_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_CACHE_TTL = 21600  # 6 hours — T86 data only updates once daily after market close

# TWSE T86 JSON endpoint
T86_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"


class TWSEScanner:
    """TWSE 三大法人買賣超掃描器"""

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def fetch_institutional_daily(self, trade_date: str) -> pd.DataFrame:
        """抓取某日全市場三大法人買賣超（TWSE T86 JSON）

        Args:
            trade_date: 日期字串 "YYYYMMDD" 格式

        Returns:
            DataFrame with columns:
                stock_id, stock_name,
                foreign_buy, foreign_sell, foreign_net,
                trust_buy, trust_sell, trust_net,
                dealer_net, total_net
        """
        # Check TTL cache first
        now = time.time()
        if trade_date in _t86_cache and now - _t86_cache[trade_date][0] < _CACHE_TTL:
            logger.debug("TWSE T86 cache hit: %s", trade_date)
            return _t86_cache[trade_date][1]

        _rate_limiter.wait()

        params = {
            "date": trade_date,
            "selectType": "ALLBUT0999",
            "response": "json",
        }

        resp = requests.get(
            T86_URL,
            params=params,
            timeout=15,
            verify=False,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("stat") != "OK" or "data" not in data:
            logger.warning("TWSE T86 %s: %s", trade_date, data.get("stat", "no data"))
            return pd.DataFrame()

        rows = data["data"]
        records = []
        for row in rows:
            try:
                # T86 fields: 證券代號, 證券名稱, 外陸資買進股數(不含外資自營商),
                # 外陸資賣出股數(不含外資自營商), 外陸資買賣超股數(不含外資自營商),
                # 外資自營商買進股數, 外資自營商賣出股數, 外資自營商買賣超股數,
                # 投信買進股數, 投信賣出股數, 投信買賣超股數,
                # 自營商買賣超股數, ...
                stock_id = row[0].strip()
                stock_name = row[1].strip()

                # Skip non-stock entries (ETF, warrants, etc. with special chars)
                if not stock_id.isdigit() or len(stock_id) != 4:
                    continue

                def parse_int(s):
                    """Parse comma-separated integer string"""
                    if isinstance(s, (int, float)):
                        return int(s)
                    return int(str(s).replace(",", "").strip())

                # Foreign investors (combined: general + dealer)
                foreign_buy = parse_int(row[2]) + parse_int(row[5])
                foreign_sell = parse_int(row[3]) + parse_int(row[6])
                foreign_net = parse_int(row[4]) + parse_int(row[7])

                # Investment trust
                trust_buy = parse_int(row[8])
                trust_sell = parse_int(row[9])
                trust_net = parse_int(row[10])

                # Dealer (combined)
                dealer_net = parse_int(row[11])

                # Convert from shares to lots (張)
                records.append(
                    {
                        "stock_id": stock_id,
                        "stock_name": stock_name,
                        "foreign_buy": foreign_buy // 1000,
                        "foreign_sell": foreign_sell // 1000,
                        "foreign_net": foreign_net // 1000,
                        "trust_buy": trust_buy // 1000,
                        "trust_sell": trust_sell // 1000,
                        "trust_net": trust_net // 1000,
                        "dealer_net": dealer_net // 1000,
                        "total_net": (foreign_net + trust_net + dealer_net) // 1000,
                    }
                )
            except (ValueError, IndexError):
                continue

        df = pd.DataFrame(records)
        if not df.empty:
            _t86_cache[trade_date] = (time.time(), df)
        return df

    def _get_trading_dates(self, days: int) -> list[str]:
        """取得最近 N 個交易日的日期（YYYYMMDD 格式）

        透過回推日曆日並跳過週末來近似交易日。
        如果 TWSE 回傳空資料（假日），該日會被跳過。
        """
        dates = []
        current = date.today()
        # Look back extra days to account for weekends and holidays
        for i in range(days * 2 + 10):
            d = current - timedelta(days=i)
            if d.weekday() < 5:  # Monday to Friday
                dates.append(d.strftime("%Y%m%d"))
            if len(dates) >= days + 5:  # Get extra buffer for holidays
                break
        return dates

    def _aggregate_institutional(self, days: int = 5) -> pd.DataFrame:
        """取得近 N 日三大法人彙總資料（完整版，不做排序截斷）

        從 fetch loop + concat + groupby + consecutive_days 計算。
        TTL cache 在 fetch_institutional_daily 層級已生效，
        多次呼叫不會重複抓 TWSE。

        Returns:
            DataFrame with columns: stock_id, stock_name,
            trust_cumulative, foreign_cumulative, dealer_cumulative,
            total_cumulative, trade_days,
            trust_consecutive_days, foreign_consecutive_days, sync_buy
        """
        candidate_dates = self._get_trading_dates(days)
        all_dfs = []
        fetched_days = 0

        for d in candidate_dates:
            if fetched_days >= days:
                break
            try:
                df = self.fetch_institutional_daily(d)
                if not df.empty:
                    df["trade_date"] = d
                    all_dfs.append(df)
                    fetched_days += 1
                    logger.info("TWSE T86 fetched: %s (%d stocks)", d, len(df))
            except Exception as e:
                logger.warning("TWSE T86 fetch failed for %s: %s", d, e)
                continue

        if not all_dfs:
            logger.warning("No TWSE T86 data available")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)

        # Aggregate by stock
        agg = (
            combined.groupby(["stock_id", "stock_name"])
            .agg(
                trust_cumulative=("trust_net", "sum"),
                foreign_cumulative=("foreign_net", "sum"),
                dealer_cumulative=("dealer_net", "sum"),
                total_cumulative=("total_net", "sum"),
                trade_days=("trade_date", "nunique"),
            )
            .reset_index()
        )

        # Calculate consecutive buy days for trust
        def _consecutive_buy_days(stock_df, column):
            """Count consecutive buy/sell days from latest date backward.
            Positive = consecutive buy days, negative = consecutive sell days."""
            sorted_df = stock_df.sort_values("trade_date", ascending=False)
            count = 0
            for _, row in sorted_df.iterrows():
                if row[column] > 0:
                    if count < 0:
                        break
                    count += 1
                elif row[column] < 0:
                    if count > 0:
                        break
                    count -= 1
                else:
                    break
            return count

        consecutive_data = {}
        for stock_id in agg["stock_id"].unique():
            stock_df = combined[combined["stock_id"] == stock_id]
            consecutive_data[stock_id] = {
                "trust_consecutive_days": _consecutive_buy_days(stock_df, "trust_net"),
                "foreign_consecutive_days": _consecutive_buy_days(
                    stock_df, "foreign_net"
                ),
            }

        # Add consecutive data
        agg["trust_consecutive_days"] = agg["stock_id"].map(
            lambda x: consecutive_data.get(x, {}).get("trust_consecutive_days", 0)
        )
        agg["foreign_consecutive_days"] = agg["stock_id"].map(
            lambda x: consecutive_data.get(x, {}).get("foreign_consecutive_days", 0)
        )

        # Sync buy: both foreign and trust cumulative positive
        agg["sync_buy"] = (agg["trust_cumulative"] > 0) & (
            agg["foreign_cumulative"] > 0
        )

        return agg

    def get_trust_top_stocks(self, days: int = 5, top_n: int = 40) -> list[dict]:
        """取得投信近 N 日累計買超前 top_n 的股票

        Returns:
            list of dicts: [{
                stock_id, stock_name,
                trust_cumulative, foreign_cumulative,
                trust_consecutive_days, foreign_consecutive_days,
                sync_buy (外資+投信同步買超)
            }]
        """
        agg = self._aggregate_institutional(days)
        if agg.empty:
            return []
        agg = agg.sort_values("trust_cumulative", ascending=False).head(top_n)
        return agg.to_dict("records")

    def get_trust_top_sellers(self, days: int = 5, top_n: int = 15) -> list[dict]:
        """投信累計賣超前 N 名

        Returns:
            list of dicts sorted by trust_cumulative ascending (most negative first)
        """
        agg = self._aggregate_institutional(days)
        if agg.empty:
            return []
        sellers = agg[agg["trust_cumulative"] < 0]
        sellers = sellers.sort_values("trust_cumulative", ascending=True).head(top_n)
        return sellers.to_dict("records")

    def get_trust_info(self, stock_id: str, days: int = 5) -> dict:
        """取得特定股票的法人籌碼彙總資訊

        Args:
            stock_id: 股票代號 (e.g. "2330")
            days: 回看天數

        Returns:
            dict with keys: foreign_cumulative, trust_cumulative, dealer_cumulative,
            foreign_consecutive_days, trust_consecutive_days, sync_buy, trade_days
            若找不到該股票回傳空 dict
        """
        agg = self._aggregate_institutional(days)
        if agg.empty:
            return {}
        row = agg[agg["stock_id"] == stock_id]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    def get_active_universe(self, days: int = 5, top_n: int = 40) -> list[str]:
        """動態股票母體 = 投信買超 top + 原始 STOCK_LIST 聯集

        Returns:
            list of stock_id strings
        """
        # Always include base STOCK_LIST
        universe = set(STOCK_LIST.keys())

        try:
            top_stocks = self.get_trust_top_stocks(days=days, top_n=top_n)
            for stock in top_stocks:
                universe.add(stock["stock_id"])
        except Exception as e:
            logger.error("Failed to get trust top stocks: %s", e)

        return sorted(universe)

    def fetch_industry_indices(self) -> dict[str, float]:
        """抓取 TWSE 產業指數報酬率

        TWSE 公開 API: /rwd/zh/afterTrading/MI_INDEX?type=IND (每日收盤產業指數)
        回傳: {semiconductor: return_pct, finance: return_pct, ...}
        """
        import re

        url = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"

        # TWSE 產業指數名稱 → 內部 sector key
        # Uses substring matching, so partial names work across API format changes
        INDEX_MAP = {
            "半導體": "semiconductor",
            "電子零組件": "electronic_parts",
            "電子": "electronics",
            "金融保險": "finance",
            "航運": "shipping",
            "生技醫療": "biotech",
            "水泥": "traditional",
            "塑膠": "traditional",
            "紡織纖維": "traditional",
            "油電燃氣": "green_energy",
            "通信網路": "telecom",
            "光電": "optoelectronics",
            "鋼鐵": "steel",
            "觀光餐旅": "tourism",
            "電機機械": "machinery",
            "化學": "chemical",
        }

        # Time pattern to detect MI_5MINS_INDEX-style rows (HH:MM:SS)
        _time_pattern = re.compile(r"^\d{2}:\d{2}:\d{2}$")

        def _try_fetch_date(target_date: date) -> dict[str, float] | None:
            """Attempt to fetch industry indices for a specific date."""
            _rate_limiter.wait()
            params = {
                "response": "json",
                "date": target_date.strftime("%Y%m%d"),
                "type": "IND",
            }
            resp = requests.get(
                url,
                params=params,
                timeout=15,
                verify=False,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("stat") != "OK":
                logger.debug(
                    "TWSE MI_INDEX %s: %s",
                    target_date,
                    data.get("stat", "no data"),
                )
                return None

            # afterTrading/MI_INDEX uses "tables" array, not top-level "data"
            tables = data.get("tables", [])
            rows = []
            for table in tables:
                table_data = table.get("data", [])
                if table_data:
                    rows.extend(table_data)

            # Fallback: some API versions may use top-level "data"
            if not rows:
                rows = data.get("data", [])

            result: dict[str, float] = {}
            if not rows:
                return None

            unmatched = []
            for row in rows:
                try:
                    index_name = str(row[0]).strip()

                    # Skip time-format rows (from MI_5MINS_INDEX-style data)
                    if _time_pattern.match(index_name):
                        continue

                    sector = None
                    for pattern, sec in INDEX_MAP.items():
                        if pattern in index_name:
                            sector = sec
                            break
                    if sector is None:
                        unmatched.append(index_name)
                        continue

                    # afterTrading format: [指數, 收盤指數, 漲跌(+/-), 漲跌點數, 漲跌百分比(%)]
                    change_pct = None
                    for col_idx in [4, 3, 2]:
                        try:
                            raw = str(row[col_idx])
                            # Strip HTML tags (e.g. <p style='color:green'>-</p>)
                            val = re.sub(r"<[^>]+>", "", raw)
                            val = (
                                val.replace(",", "")
                                .replace("%", "")
                                .strip()
                            )
                            if val and val not in ("--", "-", ""):
                                change_pct = float(val)
                                break
                        except (ValueError, IndexError):
                            continue

                    if change_pct is not None:
                        # Keep first match per sector (higher priority index)
                        if sector not in result:
                            result[sector] = change_pct / 100.0

                except (ValueError, IndexError):
                    continue

            if unmatched:
                logger.debug(
                    "TWSE industry indices unmatched names: %s", unmatched
                )

            return result if result else None

        try:
            # Try today first, then fall back to previous 5 business days
            today = date.today()
            for i in range(6):
                candidate = today - timedelta(days=i)
                if candidate.weekday() >= 5:  # Skip weekends
                    continue
                try:
                    result = _try_fetch_date(candidate)
                    if result:
                        logger.info(
                            "TWSE industry indices (%s): %s",
                            candidate,
                            {k: f"{v:.4f}" for k, v in result.items()},
                        )
                        return result
                except Exception as e:
                    logger.debug(
                        "TWSE MI_INDEX %s failed: %s", candidate, e
                    )
                    continue

            logger.warning("TWSE industry indices: no data found in last 6 days")
            return {}

        except Exception as e:
            logger.warning("TWSE industry indices fetch failed: %s", e)
            return {}

    def get_institutional_summary(self) -> dict:
        """取得今日三大法人整體買賣超概況

        Returns:
            dict with foreign_total, trust_total, dealer_total
        """
        today_str = date.today().strftime("%Y%m%d")
        candidate_dates = self._get_trading_dates(3)

        for d in candidate_dates:
            try:
                df = self.fetch_institutional_daily(d)
                if not df.empty:
                    return {
                        "date": d,
                        "foreign_total": int(df["foreign_net"].sum()),
                        "trust_total": int(df["trust_net"].sum()),
                        "dealer_total": int(df["dealer_net"].sum()),
                        "total": int(df["total_net"].sum()),
                        "stock_count": len(df),
                    }
            except Exception:
                continue

        return {
            "date": today_str,
            "foreign_total": 0,
            "trust_total": 0,
            "dealer_total": 0,
            "total": 0,
            "stock_count": 0,
        }
