#!/usr/bin/env python3
"""批次抓取歷史資料

Usage:
    python scripts/fetch_history.py 2330          # 抓取台積電 1 年資料
    python scripts/fetch_history.py 2330 365      # 指定天數
    python scripts/fetch_history.py all            # 抓取所有預設股票
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.stock_fetcher import StockFetcher
from src.db.database import init_db, upsert_stock_prices
from src.utils.constants import STOCK_LIST


def fetch_stock(stock_id: str, days: int = 365):
    """抓取單支股票的歷史資料"""
    fetcher = StockFetcher()
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    print(f"抓取 {stock_id} ({StockFetcher.get_stock_name(stock_id)}) "
          f"從 {start_date} 到 {end_date}...")

    df = fetcher.fetch_all(stock_id, start_date.isoformat(), end_date.isoformat())

    if df.empty:
        print(f"  ❌ 無法取得 {stock_id} 資料")
        return False

    upsert_stock_prices(df, stock_id)
    print(f"  ✅ 寫入 {len(df)} 筆資料")
    return True


def main():
    init_db()

    if len(sys.argv) < 2:
        print("Usage: python scripts/fetch_history.py <stock_id|all> [days]")
        sys.exit(1)

    target = sys.argv[1]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365

    if target == "all":
        success = 0
        for stock_id in STOCK_LIST:
            if fetch_stock(stock_id, days):
                success += 1
        print(f"\n完成: {success}/{len(STOCK_LIST)} 支股票")
    else:
        fetch_stock(target, days)


if __name__ == "__main__":
    main()
