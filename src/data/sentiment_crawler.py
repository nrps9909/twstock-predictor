"""社群情緒資料收集 — 標題 + 內文分析，農場文過濾

爬取流程：
1. 從 PTT / 鉅亨網 / Google News / Yahoo TW 收集新聞
2. 為有連結的文章抓取內文摘要（並行，限制 10 篇）
3. LLM 分析標題 + 內文 → 情緒 + 可信度（過濾農場標題）
"""

import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from html import unescape

import httpx
from bs4 import BeautifulSoup

from src.utils.config import settings
from src.utils.constants import SENTIMENT_SOURCES, STOCK_LIST
from src.utils.llm_client import call_claude_sync, parse_json_response
from src.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

_PTT_COOKIES = {"over18": "1"}

CNYES_NEWS_API = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock"


class SentimentCrawler:
    """社群/論壇情緒資料收集器 — 標題 + 內文 + 可信度"""

    def __init__(
        self,
        firecrawl_api_key: str | None = None,
        openclaw_url: str | None = None,
        anthropic_api_key: str | None = None,
    ):
        self.firecrawl_key = firecrawl_api_key or settings.FIRECRAWL_API_KEY
        self.openclaw_url = openclaw_url
        self.anthropic_key = anthropic_api_key or settings.ANTHROPIC_API_KEY
        self.client = httpx.Client(timeout=60, headers=_HEADERS, follow_redirects=True)

    # ── 網頁爬取 ─────────────────────────────────────────

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def _scrape_direct(self, url: str, cookies: dict | None = None) -> str:
        """httpx + BeautifulSoup 直接抓取網頁主要文字"""
        resp = self.client.get(url, cookies=cookies)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        main = soup.find("article") or soup.find("main") or soup.find("body")
        return main.get_text(separator="\n", strip=True) if main else ""

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def _scrape_firecrawl(self, url: str) -> str:
        """Firecrawl API fallback"""
        if not self.firecrawl_key:
            return ""
        resp = self.client.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={"Authorization": f"Bearer {self.firecrawl_key}"},
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
        )
        resp.raise_for_status()
        return resp.json().get("data", {}).get("markdown", "")

    def _scrape_with_fallback(self, url: str, cookies: dict | None = None) -> str:
        """先直接爬取，失敗再 fallback Firecrawl"""
        try:
            content = self._scrape_direct(url, cookies=cookies)
            if content:
                return content
        except Exception as e:
            logger.warning("直接爬取失敗 (%s): %s", url, e)
        try:
            return self._scrape_firecrawl(url)
        except Exception as e:
            logger.warning("Firecrawl fallback 也失敗 (%s): %s", url, e)
            return ""

    # ── 文章內文抓取 ──────────────────────────────────────

    def _fetch_article_summary(self, url: str, max_chars: int = 800) -> str:
        """抓取單篇文章內文摘要（前 N 字元）"""
        try:
            resp = self.client.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(
                [
                    "script",
                    "style",
                    "nav",
                    "footer",
                    "header",
                    "aside",
                    "iframe",
                    "form",
                ]
            ):
                tag.decompose()
            main = soup.find("article") or soup.find("main") or soup.find("body")
            if not main:
                return ""
            text = main.get_text(separator=" ", strip=True)
            return " ".join(text.split())[:max_chars]
        except Exception:
            return ""

    def _enrich_articles(self, articles: list[dict], max_fetch: int = 10) -> list[dict]:
        """並行抓取文章內文摘要（限制 N 篇）"""
        to_fetch = [
            a for a in articles if a.get("link") and not a.get("content_summary")
        ][:max_fetch]
        if not to_fetch:
            return articles

        def fetch_one(article):
            content = self._fetch_article_summary(article["link"])
            if content and len(content) > 50:
                article["content_summary"] = content

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_one, a) for a in to_fetch]
            for future in as_completed(futures, timeout=45):
                try:
                    future.result()
                except Exception:
                    pass

        enriched = sum(1 for a in articles if a.get("content_summary"))
        if enriched > 0:
            logger.info("內文充實: %d/%d 篇文章取得摘要", enriched, len(to_fetch))
        return articles

    # ── LLM 結構化提取（fallback 用） ─────────────────────

    def _extract_with_llm(self, content: str, stock_id: str) -> list[dict]:
        """用 Claude Haiku 從頁面文字提取文章標題與情緒"""
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
            text = call_claude_sync(
                prompt, model="claude-haiku-4-5-20251001", timeout=90
            )
            return parse_json_response(text)
        except Exception as e:
            logger.warning("LLM 提取失敗，將使用 regex fallback: %s", e)
            return []

    # ── OpenClaw ──────────────────────────────────────────

    def _query_openclaw(self, stock_id: str) -> list[dict]:
        if not self.openclaw_url:
            return []
        try:
            resp = self.client.post(
                f"{self.openclaw_url}/api/chat",
                json={
                    "message": f"分析 {stock_id} {STOCK_LIST.get(stock_id, '')} 的市場情緒"
                },
            )
            resp.raise_for_status()
            return resp.json().get("sentiment_data", [])
        except Exception as e:
            logger.error("OpenClaw 查詢失敗: %s", e)
            return []

    # ── PTT 爬蟲 ──────────────────────────────────────────

    def crawl_ptt(self, stock_id: str) -> list[dict]:
        """爬取 PTT 股票板 — 優先 HTML 解析取得連結，fallback LLM"""
        stock_name = STOCK_LIST.get(stock_id, stock_id)
        search_url = SENTIMENT_SOURCES["ptt_stock"]["search_url"].format(stock_name)

        # 方法1: 直接 HTML 爬取（保留連結）
        try:
            resp = self.client.get(search_url, cookies=_PTT_COOKIES, timeout=15)
            resp.raise_for_status()
            articles = self._parse_ptt_html(resp.text, stock_id)
            if articles:
                return articles
        except Exception as e:
            logger.warning("PTT 直接爬取失敗: %s", e)

        # 方法2: Firecrawl + LLM fallback（無連結）
        content = ""
        try:
            content = self._scrape_firecrawl(search_url)
        except Exception:
            pass
        if not content:
            return []

        llm_results = self._extract_with_llm(content, stock_id)
        if llm_results:
            return [
                {
                    "stock_id": stock_id,
                    "source": "ptt",
                    "title": item.get("title", ""),
                    "sentiment_label": item.get("sentiment", "neutral"),
                    "date": date.today(),
                    "engagement": item.get("engagement", 0),
                }
                for item in llm_results
            ]

        return self._parse_ptt_content(content, stock_id)

    def _parse_ptt_html(self, html: str, stock_id: str) -> list[dict]:
        """從 PTT HTML 解析文章標題、連結、推文數"""
        soup = BeautifulSoup(html, "html.parser")
        articles = []
        for div in soup.select("div.r-ent"):
            title_el = div.select_one("div.title a")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            href = title_el.get("href", "")
            link = f"https://www.ptt.cc{href}" if href else ""

            push_el = div.select_one("div.nrec span")
            engagement = 0
            if push_el:
                push_text = push_el.get_text(strip=True)
                if push_text == "爆":
                    engagement = 99
                elif push_text == "XX":
                    engagement = -99
                elif push_text.lstrip("-").isdigit():
                    engagement = int(push_text)

            articles.append(
                {
                    "stock_id": stock_id,
                    "source": "ptt",
                    "title": title[:200],
                    "link": link,
                    "sentiment_label": "neutral",
                    "date": date.today(),
                    "engagement": engagement,
                }
            )
        return articles[:15]

    def _parse_ptt_content(self, content: str, stock_id: str) -> list[dict]:
        """Fallback: 從純文字解析 PTT 文章"""
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
                current_article["engagement"] = line.count("推") + line.count("噓")

        if current_article.get("title"):
            current_article["stock_id"] = stock_id
            current_article["source"] = "ptt"
            current_article["date"] = date.today()
            articles.append(current_article)

        return articles

    # ── 鉅亨網爬蟲 ────────────────────────────────────────

    def crawl_cnyes(self, stock_id: str) -> list[dict]:
        """爬取鉅亨網個股新聞"""
        try:
            return self._fetch_cnyes_api(stock_id)
        except Exception as e:
            logger.warning("鉅亨網 API 失敗，改用 HTML 爬取: %s", e)

        url = SENTIMENT_SOURCES["cnyes"]["stock_url"].format(stock_id)
        content = self._scrape_with_fallback(url)
        if not content:
            return []

        llm_results = self._extract_with_llm(content, stock_id)
        if llm_results:
            return [
                {
                    "stock_id": stock_id,
                    "source": "cnyes",
                    "title": item.get("title", ""),
                    "sentiment_label": item.get("sentiment", "neutral"),
                    "date": date.today(),
                    "engagement": item.get("engagement", 0),
                }
                for item in llm_results
            ]

        stock_name = STOCK_LIST.get(stock_id, "")
        return [
            {
                "stock_id": stock_id,
                "source": "cnyes",
                "title": line.strip()[:200],
                "date": date.today(),
                "engagement": 0,
            }
            for line in content.split("\n")
            if len(line.strip()) > 10
            and any(kw in line for kw in [stock_id, stock_name])
        ]

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def _fetch_cnyes_api(self, stock_id: str, limit: int = 20) -> list[dict]:
        """鉅亨網 JSON API — 寬鬆過濾 + 摘要 + 連結"""
        stock_name = STOCK_LIST.get(stock_id, "")

        # 多抓一些，因為要過濾出相關文章
        resp = self.client.get(CNYES_NEWS_API, params={"limit": limit * 3})
        resp.raise_for_status()
        data = resp.json()

        articles = []
        items = data.get("items", {}).get("data", [])
        for item in items:
            title = item.get("title", "")
            summary = item.get("summary", "") or ""
            news_id = item.get("newsId", "")

            # 寬鬆過濾: 標題或摘要中包含股票代號或名稱
            check_text = f"{title} {summary}"
            if stock_id not in check_text and stock_name not in check_text:
                continue

            pub_at = item.get("publishAt", 0)
            news_date = (
                datetime.fromtimestamp(pub_at).date() if pub_at else date.today()
            )
            link = f"https://news.cnyes.com/news/id/{news_id}" if news_id else ""

            articles.append(
                {
                    "stock_id": stock_id,
                    "source": "cnyes",
                    "title": title[:200],
                    "content_summary": summary[:500] if summary else "",
                    "link": link,
                    "sentiment_label": "neutral",
                    "date": news_date,
                    "engagement": 0,
                }
            )
        return articles[:limit]

    # ── Google News RSS ──────────────────────────────────

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def crawl_google_news(self, stock_id: str) -> list[dict]:
        """Google News RSS — 標題 + 連結（內文由 enrichment 補充）"""
        stock_name = STOCK_LIST.get(stock_id, stock_id)
        query = f"{stock_id} {stock_name}"
        rss_url = (
            f"https://news.google.com/rss/search?q={query}"
            f"&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        )

        try:
            resp = self.client.get(rss_url)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Google News RSS 抓取失敗: %s", e)
            return []

        articles = []
        try:
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item")[:20]:
                title_el = item.find("title")
                pub_date_el = item.find("pubDate")
                link_el = item.find("link")
                source_el = item.find("source")

                title = (
                    unescape(title_el.text)
                    if title_el is not None and title_el.text
                    else ""
                )
                if not title:
                    continue

                # 優先用 source url（直接連結），否則用 Google redirect
                link = ""
                if source_el is not None and source_el.get("url"):
                    link = source_el.get("url")
                elif link_el is not None:
                    link = (link_el.text or link_el.tail or "").strip()

                news_date = date.today()
                if pub_date_el is not None and pub_date_el.text:
                    try:
                        dt = datetime.strptime(
                            pub_date_el.text[:25].strip(),
                            "%a, %d %b %Y %H:%M:%S",
                        )
                        news_date = dt.date()
                    except (ValueError, TypeError):
                        pass

                articles.append(
                    {
                        "stock_id": stock_id,
                        "source": "google_news",
                        "title": title[:200],
                        "link": link,
                        "sentiment_label": "neutral",
                        "date": news_date,
                        "engagement": 0,
                    }
                )
        except ET.ParseError as e:
            logger.warning("Google News RSS XML 解析失敗: %s", e)

        return articles

    # ── Yahoo Finance TW ─────────────────────────────────

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def crawl_yahoo_tw(self, stock_id: str) -> list[dict]:
        """Yahoo Finance TW — 優先 HTML 解析取得連結"""
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}.TW/news"

        # 方法1: 直接 HTML 解析（保留連結）
        try:
            resp = self.client.get(url, timeout=15)
            resp.raise_for_status()
            articles = self._parse_yahoo_html(resp.text, stock_id)
            if articles:
                return articles
        except Exception as e:
            logger.warning("Yahoo TW 直接解析失敗: %s", e)

        # 方法2: Firecrawl + LLM fallback
        content = self._scrape_with_fallback(url)
        if not content:
            return []

        llm_results = self._extract_with_llm(content, stock_id)
        if llm_results:
            return [
                {
                    "stock_id": stock_id,
                    "source": "yahoo_tw",
                    "title": item.get("title", ""),
                    "sentiment_label": item.get("sentiment", "neutral"),
                    "date": date.today(),
                    "engagement": item.get("engagement", 0),
                }
                for item in llm_results
            ]

        # Fallback: 關鍵字匹配
        stock_name = STOCK_LIST.get(stock_id, "")
        articles = []
        for line in content.split("\n"):
            line = line.strip()
            if len(line) > 10 and any(kw in line for kw in [stock_id, stock_name]):
                articles.append(
                    {
                        "stock_id": stock_id,
                        "source": "yahoo_tw",
                        "title": line[:200],
                        "sentiment_label": "neutral",
                        "date": date.today(),
                        "engagement": 0,
                    }
                )
        return articles[:15]

    def _parse_yahoo_html(self, html: str, stock_id: str) -> list[dict]:
        """從 Yahoo TW HTML 解析新聞連結"""
        soup = BeautifulSoup(html, "html.parser")
        articles = []
        seen = set()
        for a_tag in soup.select("a[href*='/news/']"):
            title = a_tag.get_text(strip=True)
            if not title or len(title) < 8 or title in seen:
                continue
            seen.add(title)
            href = a_tag.get("href", "")
            if href.startswith("/"):
                href = f"https://tw.stock.yahoo.com{href}"

            articles.append(
                {
                    "stock_id": stock_id,
                    "source": "yahoo_tw",
                    "title": title[:200],
                    "link": href,
                    "sentiment_label": "neutral",
                    "date": date.today(),
                    "engagement": 0,
                }
            )
        return articles[:15]

    # ── 全球 context ─────────────────────────────────────

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def crawl_global_context(self, stock_id: str) -> list[str]:
        """抓取半導體/AI/宏觀新聞標題"""
        queries = [
            "semiconductor+AI+chip",
            "TSMC+台積電",
            "Fed+interest+rate+economy",
        ]
        headlines = []

        for q in queries:
            rss_url = f"https://news.google.com/rss/search?q={q}&hl=en&gl=US&ceid=US:en"
            try:
                resp = self.client.get(rss_url)
                resp.raise_for_status()
                root = ET.fromstring(resp.text)
                for item in root.findall(".//item")[:5]:
                    title_el = item.find("title")
                    if title_el is not None and title_el.text:
                        headlines.append(unescape(title_el.text))
            except Exception as e:
                logger.warning("Global context RSS 抓取失敗 (%s): %s", q, e)

        logger.info("全球 context: 取得 %d 則標題", len(headlines))
        return headlines

    # ── 整合爬蟲 ──────────────────────────────────────────

    def crawl_all(self, stock_id: str) -> list[dict]:
        """爬取所有平台 → 內文充實 → 回傳含 content_summary 的文章"""
        all_articles = []

        if self.openclaw_url:
            openclaw_data = self._query_openclaw(stock_id)
            if openclaw_data:
                return openclaw_data

        logger.info("使用直接爬取模式收集 %s 情緒資料", stock_id)

        ptt_articles = self.crawl_ptt(stock_id)
        all_articles.extend(ptt_articles)
        logger.info("PTT: 取得 %d 篇文章", len(ptt_articles))

        cnyes_articles = self.crawl_cnyes(stock_id)
        all_articles.extend(cnyes_articles)
        logger.info("鉅亨網: 取得 %d 篇文章", len(cnyes_articles))

        google_articles = self.crawl_google_news(stock_id)
        all_articles.extend(google_articles)
        logger.info("Google News: 取得 %d 篇文章", len(google_articles))

        yahoo_articles = self.crawl_yahoo_tw(stock_id)
        all_articles.extend(yahoo_articles)
        logger.info("Yahoo TW: 取得 %d 篇文章", len(yahoo_articles))

        # 內文充實: 為有連結的文章抓取內文摘要
        if all_articles:
            all_articles = self._enrich_articles(all_articles, max_fetch=10)

        return all_articles
