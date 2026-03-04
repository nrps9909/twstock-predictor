"""市場情報聚合服務 — 新聞 + 法人動向"""

import asyncio
import logging
from datetime import date, timedelta

from src.data.twse_scanner import TWSEScanner
from src.data.sentiment_crawler import SentimentCrawler

logger = logging.getLogger(__name__)


async def get_market_intel() -> dict:
    """聚合市場情報

    Returns:
        dict with:
        - global_news: 全球宏觀新聞
        - tw_news: 台股市場新聞
        - trust_top_buy: 投信買超排行
        - trust_top_sell: 投信賣超排行
        - sync_buy: 外資+投信同步買超
        - institutional_total: 三大法人整體買賣超
    """
    scanner = TWSEScanner()

    # Fetch institutional data and news in parallel
    async def _get_trust_top_buy():
        try:
            return await asyncio.to_thread(
                scanner.get_trust_top_stocks, days=5, top_n=15)
        except Exception as e:
            logger.error("Failed to get trust top buy: %s", e)
            return []

    async def _get_trust_top_sell():
        try:
            return await asyncio.to_thread(
                scanner.get_trust_top_sellers, days=5, top_n=15)
        except Exception as e:
            logger.error("Failed to get trust top sell: %s", e)
            return []

    async def _get_institutional_summary():
        try:
            return await asyncio.to_thread(scanner.get_institutional_summary)
        except Exception as e:
            logger.error("Failed to get institutional summary: %s", e)
            return {"foreign_total": 0, "trust_total": 0, "dealer_total": 0, "total": 0}

    async def _get_news():
        try:
            crawler = SentimentCrawler()
            # Get global news via Google RSS
            global_news = await asyncio.to_thread(
                crawler.crawl_global_context, "台股")
            return global_news
        except Exception as e:
            logger.error("Failed to get news: %s", e)
            return []

    async def _get_full_institutional():
        """Get full institutional data for sync_buy and foreign ranking"""
        try:
            return await asyncio.to_thread(
                scanner.get_trust_top_stocks, days=5, top_n=50)
        except Exception as e:
            logger.error("Failed to get full institutional data: %s", e)
            return []

    trust_top_buy, trust_top_sell, summary, global_news, trust_data = await asyncio.gather(
        _get_trust_top_buy(),
        _get_trust_top_sell(),
        _get_institutional_summary(),
        _get_news(),
        _get_full_institutional(),
    )

    # Sync buy: both foreign and trust positive
    sync_buy = [
        s for s in trust_data
        if s.get("trust_cumulative", 0) > 0 and s.get("foreign_cumulative", 0) > 0
    ]
    sync_buy.sort(key=lambda x: x.get("trust_cumulative", 0) + x.get("foreign_cumulative", 0),
                  reverse=True)

    # Foreign top buy
    foreign_top_buy = sorted(trust_data, key=lambda x: x.get("foreign_cumulative", 0), reverse=True)[:15]

    # Format news
    formatted_news = []
    for item in global_news[:20]:
        if isinstance(item, dict):
            formatted_news.append({
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "date": item.get("date", ""),
                "url": item.get("url", ""),
            })
        elif isinstance(item, str):
            formatted_news.append({"title": item, "source": "", "date": "", "url": ""})

    return {
        "global_news": formatted_news,
        "tw_news": [],  # Can be populated from Yahoo TW separately
        "trust_top_buy": [
            {
                "stock_id": s["stock_id"],
                "stock_name": s.get("stock_name", s["stock_id"]),
                "net_buy": s.get("trust_cumulative", 0),
                "consecutive_days": s.get("trust_consecutive_days", 0),
                "foreign_net": s.get("foreign_cumulative", 0),
            }
            for s in trust_top_buy
        ],
        "trust_top_sell": [
            {
                "stock_id": s["stock_id"],
                "stock_name": s.get("stock_name", s["stock_id"]),
                "net_sell": abs(s.get("trust_cumulative", 0)),
                "consecutive_days": s.get("trust_consecutive_days", 0),
                "foreign_net": s.get("foreign_cumulative", 0),
            }
            for s in trust_top_sell
        ],
        "foreign_top_buy": [
            {
                "stock_id": s["stock_id"],
                "stock_name": s.get("stock_name", s["stock_id"]),
                "net_buy": s.get("foreign_cumulative", 0),
                "trust_net": s.get("trust_cumulative", 0),
            }
            for s in foreign_top_buy
        ],
        "sync_buy": [
            {
                "stock_id": s["stock_id"],
                "stock_name": s.get("stock_name", s["stock_id"]),
                "trust_net": s.get("trust_cumulative", 0),
                "foreign_net": s.get("foreign_cumulative", 0),
            }
            for s in sync_buy[:15]
        ],
        "institutional_total": {
            "date": summary.get("date", ""),
            "foreign": summary.get("foreign_total", 0),
            "trust": summary.get("trust_total", 0),
            "dealer": summary.get("dealer_total", 0),
            "total": summary.get("total", 0),
        },
    }
