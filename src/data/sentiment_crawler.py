"""社群情緒資料收集 — httpx 直接爬取優先，Firecrawl 作為可選 fallback

Bug 6 fix: 優先用 LLM 結構化提取（Claude Haiku），regex 作為 fallback
免費方案：PTT 用 httpx + cookie，鉅亨網用公開 JSON API
"""

import json
import logging
from datetime import date, datetime

import httpx
from bs4 import BeautifulSoup

from src.utils.config import settings
from src.utils.constants import SENTIMENT_SOURCES, STOCK_LIST
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

# PTT 需要 over18 cookie
_PTT_COOKIES = {"over18": "1"}

# 鉅亨網公開 API
CNYES_NEWS_API = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock"


class SentimentCrawler:
    """社群/論壇情緒資料收集器

    爬取優先順序：
    1. 直接爬取（httpx + BeautifulSoup，免費）
    2. Firecrawl fallback（免費額度有限，可選）

    情緒解析：優先使用 LLM 結構化提取（Claude Haiku），regex 作為 fallback
    """

    def __init__(
        self,
        firecrawl_api_key: str | None = None,
        openclaw_url: str | None = None,
        anthropic_api_key: str | None = None,
    ):
        self.firecrawl_key = firecrawl_api_key or settings.FIRECRAWL_API_KEY
        self.openclaw_url = openclaw_url  # e.g. "http://localhost:3000"
        self.anthropic_key = anthropic_api_key or settings.ANTHROPIC_API_KEY
        self.client = httpx.Client(
            timeout=60, headers=_HEADERS, follow_redirects=True
        )

    # ── 直接爬取（免費） ─────────────────────────────────

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def _scrape_direct(self, url: str, cookies: dict | None = None) -> str:
        """用 httpx + BeautifulSoup 直接抓取網頁主要文字"""
        resp = self.client.get(url, cookies=cookies)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 移除 script/style/nav
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # 取 article 或 main，否則取 body
        main = soup.find("article") or soup.find("main") or soup.find("body")
        return main.get_text(separator="\n", strip=True) if main else ""

    # ── Firecrawl fallback ───────────────────────────────

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

    def _scrape_with_fallback(self, url: str, cookies: dict | None = None) -> str:
        """先直接爬取，失敗再 fallback 到 Firecrawl"""
        try:
            content = self._scrape_direct(url, cookies=cookies)
            if content:
                return content
        except Exception as e:
            logger.warning("直接爬取失敗 (%s): %s", url, e)

        # Firecrawl fallback
        try:
            return self._scrape_firecrawl(url)
        except Exception as e:
            logger.warning("Firecrawl fallback 也失敗 (%s): %s", url, e)
            return ""

    # ── LLM 結構化提取 ─────────────────────────────────

    def _extract_with_llm(self, content: str, stock_id: str) -> list[dict]:
        """Bug 6 fix: 用 Claude Haiku 結構化提取文章與情緒

        Returns:
            list of {title, sentiment, engagement, date_str}
        """
        if not self.anthropic_key:
            return []

        stock_name = STOCK_LIST.get(stock_id, stock_id)
        prompt = f"""從以下網頁內容中提取與股票 {stock_id} ({stock_name}) 相關的文章。

回傳 JSON array，每篇文章格式：
{{"title": "文章標題", "sentiment": "bullish|bearish|neutral", "engagement": 0, "date_str": "YYYY-MM-DD 或空字串"}}

規則：
- 只提取與 {stock_id} 或 {stock_name} 直接相關的文章
- sentiment 基於標題內容判斷（利多=bullish, 利空=bearish, 中性=neutral）
- engagement 是推文數/回覆數（如果看得出來），否則填 0
- 如果沒有相關文章，回傳空 array []

只回傳 JSON，不要其他文字。

網頁內容：
{content[:4000]}"""

        try:
            resp = self.client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["content"][0]["text"].strip()

            # Parse JSON from response
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception as e:
            logger.warning("LLM 提取失敗，將使用 regex fallback: %s", e)
            return []

    # ── OpenClaw 模式 ───────────────────────────────────

    def _query_openclaw(self, stock_id: str) -> list[dict]:
        """透過 OpenClaw Gateway 執行情緒爬蟲 skill"""
        if not self.openclaw_url:
            logger.warning("未設定 OpenClaw URL")
            return []

        try:
            resp = self.client.post(
                f"{self.openclaw_url}/api/chat",
                json={
                    "message": f"分析 {stock_id} {STOCK_LIST.get(stock_id, '')} 的市場情緒",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("sentiment_data", [])
        except Exception as e:
            logger.error("OpenClaw 查詢失敗: %s", e)
            return []

    # ── PTT 爬蟲 ────────────────────────────────────────

    def crawl_ptt(self, stock_id: str) -> list[dict]:
        """爬取 PTT 股票板相關文章（直接爬取，帶 over18 cookie）"""
        stock_name = STOCK_LIST.get(stock_id, stock_id)
        search_url = SENTIMENT_SOURCES["ptt_stock"]["search_url"].format(stock_name)

        content = self._scrape_with_fallback(search_url, cookies=_PTT_COOKIES)
        if not content:
            return []

        # 優先使用 LLM 結構化提取
        llm_results = self._extract_with_llm(content, stock_id)
        if llm_results:
            articles = []
            for item in llm_results:
                articles.append({
                    "stock_id": stock_id,
                    "source": "ptt",
                    "title": item.get("title", ""),
                    "sentiment_label": item.get("sentiment", "neutral"),
                    "date": date.today(),
                    "engagement": item.get("engagement", 0),
                })
            return articles

        # Fallback: regex 解析
        return self._parse_ptt_content(content, stock_id)

    def _parse_ptt_content(self, content: str, stock_id: str) -> list[dict]:
        """解析 PTT 文字內容，提取文章資訊（fallback regex）"""
        articles = []
        lines = content.split("\n")
        current_article = {}

        for line in lines:
            line = line.strip()
            if not line:
                if current_article.get("title"):
                    current_article["stock_id"] = stock_id
                    current_article["source"] = "ptt"
                    current_article["date"] = date.today()
                    articles.append(current_article)
                    current_article = {}
                continue

            if line.startswith("#") or line.startswith("["):
                current_article["title"] = line.lstrip("#[] ")
            elif "推" in line or "噓" in line:
                push_count = line.count("推")
                boo_count = line.count("噓")
                current_article["engagement"] = push_count + boo_count

        if current_article.get("title"):
            current_article["stock_id"] = stock_id
            current_article["source"] = "ptt"
            current_article["date"] = date.today()
            articles.append(current_article)

        return articles

    # ── 鉅亨網爬蟲 ──────────────────────────────────────

    def crawl_cnyes(self, stock_id: str) -> list[dict]:
        """爬取鉅亨網個股新聞（優先用公開 JSON API）"""
        # 先嘗試 JSON API
        try:
            return self._fetch_cnyes_api(stock_id)
        except Exception as e:
            logger.warning("鉅亨網 API 失敗，改用 HTML 爬取: %s", e)

        # Fallback: HTML 爬取
        url = SENTIMENT_SOURCES["cnyes"]["stock_url"].format(stock_id)
        content = self._scrape_with_fallback(url)
        if not content:
            return []

        # 優先使用 LLM 結構化提取
        llm_results = self._extract_with_llm(content, stock_id)
        if llm_results:
            articles = []
            for item in llm_results:
                articles.append({
                    "stock_id": stock_id,
                    "source": "cnyes",
                    "title": item.get("title", ""),
                    "sentiment_label": item.get("sentiment", "neutral"),
                    "date": date.today(),
                    "engagement": item.get("engagement", 0),
                })
            return articles

        # Fallback: 簡單字串匹配
        articles = []
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if len(line) > 10 and any(
                kw in line for kw in [stock_id, STOCK_LIST.get(stock_id, "")]
            ):
                articles.append({
                    "stock_id": stock_id,
                    "source": "cnyes",
                    "title": line[:200],
                    "date": date.today(),
                    "engagement": 0,
                })

        return articles

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def _fetch_cnyes_api(self, stock_id: str, limit: int = 20) -> list[dict]:
        """透過鉅亨網公開 JSON API 取得新聞"""
        resp = self.client.get(
            CNYES_NEWS_API,
            params={"limit": limit},
        )
        resp.raise_for_status()
        data = resp.json()

        stock_name = STOCK_LIST.get(stock_id, "")
        articles = []
        items = data.get("items", {}).get("data", [])
        for item in items:
            title = item.get("title", "")
            if stock_id not in title and stock_name not in title:
                continue
            pub_at = item.get("publishAt", 0)
            news_date = (
                datetime.fromtimestamp(pub_at).date() if pub_at else date.today()
            )
            articles.append({
                "stock_id": stock_id,
                "source": "cnyes",
                "title": title[:200],
                "sentiment_label": "neutral",
                "date": news_date,
                "engagement": 0,
            })
        return articles

    # ── 整合爬蟲 ────────────────────────────────────────

    def crawl_all(self, stock_id: str) -> list[dict]:
        """爬取所有平台的情緒資料

        Returns:
            list of {stock_id, source, title, content_summary, date, engagement}
        """
        all_articles = []

        # 嘗試 OpenClaw 模式
        if self.openclaw_url:
            openclaw_data = self._query_openclaw(stock_id)
            if openclaw_data:
                return openclaw_data

        # 直接爬取模式（免費優先）
        logger.info("使用直接爬取模式收集 %s 情緒資料", stock_id)

        ptt_articles = self.crawl_ptt(stock_id)
        all_articles.extend(ptt_articles)
        logger.info("PTT: 取得 %d 篇文章", len(ptt_articles))

        cnyes_articles = self.crawl_cnyes(stock_id)
        all_articles.extend(cnyes_articles)
        logger.info("鉅亨網: 取得 %d 篇文章", len(cnyes_articles))

        return all_articles
