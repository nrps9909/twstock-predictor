"""財經新聞收集模組 — 鉅亨網、工商時報、經濟日報

優先使用免費方案：httpx 直接爬取 + BeautifulSoup 解析
鉅亨網有公開 JSON API，直接用；其餘用 HTML 解析
Firecrawl 作為可選 fallback（免費額度有限）
"""

import logging
from datetime import date, datetime

import httpx
from bs4 import BeautifulSoup

from src.utils.config import settings
from src.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

# User-Agent 避免被擋
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# 鉅亨網公開 API
CNYES_API_URL = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock"

# 新聞來源設定
NEWS_SOURCES = {
    "cnyes": {
        "name": "鉅亨網",
        "search_url": "https://www.cnyes.com/search/news?keyword={}",
    },
    "ctee": {
        "name": "工商時報",
        "search_url": "https://www.ctee.com.tw/search/{}",
    },
    "udn_money": {
        "name": "經濟日報",
        "search_url": "https://money.udn.com/search/result/1001/{}/1",
    },
}


class NewsCrawler:
    """財經新聞收集器（免費直接爬取優先）"""

    def __init__(self, firecrawl_api_key: str | None = None):
        self.firecrawl_key = firecrawl_api_key or settings.FIRECRAWL_API_KEY
        self.client = httpx.Client(timeout=60, headers=_HEADERS, follow_redirects=True)

    # ── 直接爬取（免費） ─────────────────────────────────

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def _scrape_direct(self, url: str) -> str:
        """用 httpx + BeautifulSoup 直接抓取網頁主要文字"""
        resp = self.client.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 移除 script/style
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # 取 article 或 main，否則取 body
        main = soup.find("article") or soup.find("main") or soup.find("body")
        return main.get_text(separator="\n", strip=True) if main else ""

    # ── 鉅亨網 JSON API（免費、最穩定） ──────────────────

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def _fetch_cnyes_api(self, keyword: str, limit: int = 20) -> list[dict]:
        """透過鉅亨網公開 API 取得台股新聞"""
        resp = self.client.get(
            CNYES_API_URL,
            params={"limit": limit},
        )
        resp.raise_for_status()
        data = resp.json()

        articles = []
        items = data.get("items", {}).get("data", [])
        for item in items:
            title = item.get("title", "")
            # 只保留含關鍵字的新聞
            if keyword and keyword not in title:
                continue
            pub_at = item.get("publishAt", 0)
            news_date = (
                datetime.fromtimestamp(pub_at).date() if pub_at else date.today()
            )
            articles.append({
                "source": "cnyes",
                "title": title[:200],
                "date": news_date,
                "url": f"https://news.cnyes.com/news/id/{item.get('newsId', '')}",
            })
        return articles

    # ── Firecrawl fallback（可選） ────────────────────────

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def _scrape_firecrawl(self, url: str) -> str:
        """Firecrawl API fallback（免費額度有限）"""
        if not self.firecrawl_key:
            return ""

        resp = self.client.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={"Authorization": f"Bearer {self.firecrawl_key}"},
            json={
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": True,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", {}).get("markdown", "")

    # ── 公開方法 ─────────────────────────────────────────

    def fetch_news(self, keyword: str, source: str = "cnyes") -> list[dict]:
        """抓取特定關鍵字的新聞

        Args:
            keyword: 搜尋關鍵字（股票名稱或代號）
            source: 新聞來源 key

        Returns:
            list of {source, title, date, url}
        """
        # 鉅亨網走 JSON API
        if source == "cnyes":
            try:
                return self._fetch_cnyes_api(keyword)
            except Exception as e:
                logger.warning("鉅亨網 API 失敗，嘗試 HTML 爬取: %s", e)

        src_config = NEWS_SOURCES.get(source)
        if not src_config:
            logger.warning("不支援的新聞來源: %s", source)
            return []

        search_url = src_config["search_url"].format(keyword)

        # 優先直接爬取
        try:
            content = self._scrape_direct(search_url)
        except Exception as e:
            logger.warning("直接爬取失敗 (%s)，嘗試 Firecrawl fallback: %s", source, e)
            try:
                content = self._scrape_firecrawl(search_url)
            except Exception as e2:
                logger.error("新聞抓取全部失敗 (%s): %s", source, e2)
                return []

        if not content:
            return []

        # 從文字提取新聞標題
        articles = []
        for line in content.split("\n"):
            line = line.strip()
            if not line or len(line) < 10:
                continue
            if len(line) < 200:
                articles.append({
                    "source": source,
                    "title": line[:200],
                    "date": date.today(),
                })

        return articles[:20]  # 最多 20 則

    def fetch_all_sources(self, keyword: str) -> list[dict]:
        """從所有新聞來源抓取資料"""
        all_news = []
        for source_key in NEWS_SOURCES:
            news = self.fetch_news(keyword, source_key)
            all_news.extend(news)
            logger.info("%s: 取得 %d 則新聞", source_key, len(news))
        return all_news
