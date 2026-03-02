/**
 * OpenClaw Skill: 台股市場情緒爬蟲
 *
 * 流程：
 * 1. 接收股票代號 → 產生各平台搜尋 URL
 * 2. 呼叫 Firecrawl scrape 各 URL → 取得 markdown 內容
 * 3. 用 LLM 從 markdown 中提取：文章標題、內容摘要、推噓文統計
 * 4. 回傳結構化 JSON
 */

interface SentimentArticle {
  source: string;
  title: string;
  contentSummary: string;
  sentimentLabel: "bullish" | "bearish" | "neutral";
  sentimentScore: number;
  keywords: string[];
  engagement: number;
  url: string;
  date: string;
}

interface SentimentResult {
  stockId: string;
  articles: SentimentArticle[];
  summary: {
    totalArticles: number;
    avgScore: number;
    bullishRatio: number;
    bearishRatio: number;
    topKeywords: string[];
  };
}

// 股票名稱對照
const STOCK_NAMES: Record<string, string> = {
  "2330": "台積電",
  "2317": "鴻海",
  "2382": "廣達",
  "2454": "聯發科",
  "2881": "富邦金",
};

// 平台搜尋 URL 模板
function getSearchUrls(stockId: string): { source: string; url: string }[] {
  const name = STOCK_NAMES[stockId] || stockId;
  return [
    {
      source: "ptt",
      url: `https://www.ptt.cc/bbs/Stock/search?q=${encodeURIComponent(name)}`,
    },
    {
      source: "cnyes",
      url: `https://www.cnyes.com/twstock/${stockId}`,
    },
    {
      source: "yahoo",
      url: `https://tw.stock.yahoo.com/quote/${stockId}.TW/news`,
    },
  ];
}

// Skill 主入口
export default async function handler(params: {
  stock_id: string;
  tools: {
    firecrawl: {
      scrape: (url: string) => Promise<{ markdown: string }>;
    };
    llm: {
      chat: (prompt: string) => Promise<string>;
    };
  };
}): Promise<SentimentResult> {
  const { stock_id, tools } = params;
  const searchUrls = getSearchUrls(stock_id);
  const articles: SentimentArticle[] = [];

  // 並行爬取所有平台
  const scrapeResults = await Promise.allSettled(
    searchUrls.map(async ({ source, url }) => {
      try {
        const result = await tools.firecrawl.scrape(url);
        return { source, url, markdown: result.markdown };
      } catch (error) {
        console.error(`Failed to scrape ${source}: ${error}`);
        return { source, url, markdown: "" };
      }
    })
  );

  // 對每個平台的內容進行 LLM 分析
  for (const result of scrapeResults) {
    if (result.status !== "fulfilled" || !result.value.markdown) continue;

    const { source, url, markdown } = result.value;
    const truncated = markdown.slice(0, 4000);

    const prompt = `分析以下來自${source}的台股相關內容。
提取所有與股票${stock_id}相關的文章/討論。
對每篇文章判斷情緒（利多/利空/中性）並給分（-1到1）。

回傳 JSON 陣列格式（不要其他文字）:
[{"title": "文章標題", "summary": "摘要", "label": "bullish"|"bearish"|"neutral", "score": 0.5, "keywords": ["關鍵詞"], "engagement": 推文數}]

內容:
${truncated}`;

    try {
      const llmResponse = await tools.llm.chat(prompt);
      const parsed: any[] = JSON.parse(llmResponse);

      for (const item of parsed) {
        articles.push({
          source,
          title: item.title || "",
          contentSummary: item.summary || "",
          sentimentLabel: item.label || "neutral",
          sentimentScore: item.score || 0,
          keywords: item.keywords || [],
          engagement: item.engagement || 0,
          url,
          date: new Date().toISOString().split("T")[0],
        });
      }
    } catch (error) {
      console.error(`LLM analysis failed for ${source}: ${error}`);
    }
  }

  // 彙總
  const totalArticles = articles.length;
  const avgScore =
    totalArticles > 0
      ? articles.reduce((sum, a) => sum + a.sentimentScore, 0) / totalArticles
      : 0;
  const bullishCount = articles.filter(
    (a) => a.sentimentLabel === "bullish"
  ).length;
  const bearishCount = articles.filter(
    (a) => a.sentimentLabel === "bearish"
  ).length;

  // 提取高頻關鍵詞
  const keywordCounts: Record<string, number> = {};
  for (const a of articles) {
    for (const kw of a.keywords) {
      keywordCounts[kw] = (keywordCounts[kw] || 0) + 1;
    }
  }
  const topKeywords = Object.entries(keywordCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([kw]) => kw);

  return {
    stockId: stock_id,
    articles,
    summary: {
      totalArticles,
      avgScore: Math.round(avgScore * 100) / 100,
      bullishRatio:
        totalArticles > 0
          ? Math.round((bullishCount / totalArticles) * 100) / 100
          : 0,
      bearishRatio:
        totalArticles > 0
          ? Math.round((bearishCount / totalArticles) * 100) / 100
          : 0,
      topKeywords,
    },
  };
}
