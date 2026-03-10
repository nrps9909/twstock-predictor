"""Batch fetch historical data for all stocks in a sector.

Usage:
    python scripts/fetch_sector_data.py [sector] [--days N]

Examples:
    python scripts/fetch_sector_data.py electronics
    python scripts/fetch_sector_data.py semiconductor --days 2000
    python scripts/fetch_sector_data.py --all --days 3000
"""

import argparse
import logging
import sys
import time
from datetime import date, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("fetch_sector")

# Suppress noisy loggers
for name in ["httpx", "urllib3", "httpcore", "requests"]:
    logging.getLogger(name).setLevel(logging.WARNING)


def fetch_stock_data(stock_id: str, start: str, end: str, fetcher) -> dict:
    """Fetch and store all data for a single stock.

    Returns:
        {"stock_id": ..., "rows": N, "status": "ok"|"empty"|"error", "elapsed": ...}
    """
    from src.db.database import upsert_stock_prices

    t0 = time.time()
    try:
        df = fetcher.fetch_all(stock_id, start, end)
        elapsed = time.time() - t0

        if df.empty:
            return {
                "stock_id": stock_id,
                "rows": 0,
                "status": "empty",
                "elapsed": elapsed,
            }

        upsert_stock_prices(df, stock_id)
        return {
            "stock_id": stock_id,
            "rows": len(df),
            "status": "ok",
            "elapsed": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("  Failed %s: %s", stock_id, e)
        return {
            "stock_id": stock_id,
            "rows": 0,
            "status": f"error: {e}",
            "elapsed": elapsed,
        }


def fetch_sector(sector: str, days: int = 3000) -> list[dict]:
    """Fetch data for all stocks in a sector."""
    from api.services.market_service import STOCK_SECTOR
    from src.data.stock_fetcher import StockFetcher

    stocks = [sid for sid, sec in STOCK_SECTOR.items() if sec == sector]
    if not stocks:
        logger.error("No stocks found for sector '%s'", sector)
        all_sectors = sorted(set(STOCK_SECTOR.values()))
        logger.info("Available sectors: %s", ", ".join(all_sectors))
        return []

    today = date.today()
    start = (today - timedelta(days=days)).isoformat()
    end = today.isoformat()

    print(f"\n{'=' * 60}")
    print(f"  Fetching sector: {sector}")
    print(f"  Stocks: {len(stocks)} — {stocks}")
    print(f"  Date range: {start} ~ {end} ({days} days)")
    print(f"{'=' * 60}\n")

    fetcher = StockFetcher()
    results = []

    for i, stock_id in enumerate(stocks, 1):
        logger.info("[%d/%d] Fetching %s ...", i, len(stocks), stock_id)
        result = fetch_stock_data(stock_id, start, end, fetcher)
        results.append(result)

        status_icon = (
            "OK"
            if result["status"] == "ok"
            else "SKIP"
            if result["status"] == "empty"
            else "ERR"
        )
        logger.info(
            "  %s — %s: %d rows (%.1fs)",
            status_icon,
            stock_id,
            result["rows"],
            result["elapsed"],
        )

    return results


def print_summary(sector: str, results: list[dict]):
    """Print fetch summary."""
    ok = [r for r in results if r["status"] == "ok"]
    empty = [r for r in results if r["status"] == "empty"]
    errors = [r for r in results if r["status"].startswith("error")]
    total_rows = sum(r["rows"] for r in results)
    total_time = sum(r["elapsed"] for r in results)

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {sector}")
    print(f"{'=' * 60}")
    print(f"  Total stocks: {len(results)}")
    print(f"  Succeeded:    {len(ok)}")
    print(f"  Empty:        {len(empty)}")
    print(f"  Failed:       {len(errors)}")
    print(f"  Total rows:   {total_rows:,}")
    print(f"  Total time:   {total_time:.1f}s")
    print()

    if ok:
        print("  Fetched:")
        for r in sorted(ok, key=lambda x: x["rows"], reverse=True):
            print(f"    {r['stock_id']}: {r['rows']:>5} rows ({r['elapsed']:.1f}s)")

    if empty:
        print(f"\n  Empty (no data): {[r['stock_id'] for r in empty]}")

    if errors:
        print(f"\n  Errors:")
        for r in errors:
            print(f"    {r['stock_id']}: {r['status']}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Fetch sector stock data")
    parser.add_argument("sector", nargs="?", default="electronics", help="Sector name")
    parser.add_argument(
        "--days", type=int, default=3000, help="Days of history (default: 3000)"
    )
    parser.add_argument("--all", action="store_true", help="Fetch ALL sectors")
    args = parser.parse_args()

    if args.all:
        from api.services.market_service import STOCK_SECTOR

        all_sectors = sorted(set(STOCK_SECTOR.values()))
        print(f"Fetching ALL {len(all_sectors)} sectors: {all_sectors}")
        for sector in all_sectors:
            results = fetch_sector(sector, args.days)
            print_summary(sector, results)
    else:
        results = fetch_sector(args.sector, args.days)
        print_summary(args.sector, results)


if __name__ == "__main__":
    sys.exit(main() or 0)
