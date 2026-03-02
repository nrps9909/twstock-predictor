"""Streamlit Page 2: 情緒分析"""

from datetime import date, timedelta
import json

import streamlit as st
import pandas as pd
import plotly.express as px

from app.components.sidebar import render_sidebar
from app.components.startup import ensure_db_initialized
from app.components.theme import inject_theme
from app.components.charts import create_sentiment_trend_chart
from src.data.sentiment_crawler import SentimentCrawler
from src.analysis.sentiment import SentimentAnalyzer, DailySentiment
from src.db.database import (
    get_sentiment,
    get_stock_prices,
    insert_sentiment,
)

st.set_page_config(page_title="情緒分析", page_icon="💬", layout="wide")
inject_theme()
ensure_db_initialized()


@st.cache_data(ttl=600)
def _load_sentiment(stock_id: str, start: str, end: str):
    """快取情緒聚合計算"""
    from src.db.database import get_sentiment as _get
    df = _get(stock_id, date.fromisoformat(start), date.fromisoformat(end))
    if df.empty:
        return df, pd.DataFrame()
    daily = df.groupby("date").agg(
        avg_score=("sentiment_score", "mean"),
        post_count=("sentiment_score", "count"),
        bullish_count=("sentiment_label", lambda x: (x == "bullish").sum()),
        bearish_count=("sentiment_label", lambda x: (x == "bearish").sum()),
    ).reset_index()
    daily["bullish_ratio"] = daily["bullish_count"] / daily["post_count"]
    daily["bearish_ratio"] = daily["bearish_count"] / daily["post_count"]
    daily["sentiment_score"] = daily["avg_score"]
    daily["sentiment_ma5"] = daily["sentiment_score"].rolling(5, min_periods=1).mean()
    return df, daily

params = render_sidebar()
stock_id = params["stock_id"]
stock_name = params["stock_name"]
lookback_days = params["lookback_days"]

st.title(f"💬 情緒分析 — {stock_id} {stock_name}")

# ── 情緒資料取得 ─────────────────────────────────────────

end_date = date.today()
start_date = end_date - timedelta(days=lookback_days)

# 從 DB 讀取（含快取）
sentiment_df, _ = _load_sentiment(stock_id, start_date.isoformat(), end_date.isoformat())

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 爬取最新情緒", use_container_width=True):
        with st.spinner("爬取社群資料中..."):
            crawler = SentimentCrawler()
            articles = crawler.crawl_all(stock_id)

            if articles:
                analyzer = SentimentAnalyzer()
                sentiment_records = []

                progress = st.progress(0)
                for i, article in enumerate(articles):
                    # 分析每篇文章情緒
                    text = article.get("title", "") + " " + article.get("content_summary", "")
                    score = analyzer.analyze_text(text)

                    sentiment_records.append({
                        "stock_id": stock_id,
                        "date": article.get("date", date.today()),
                        "source": article.get("source", "unknown"),
                        "title": article.get("title", ""),
                        "content_summary": article.get("content_summary", ""),
                        "sentiment_label": score.label,
                        "sentiment_score": score.score,
                        "keywords": json.dumps(score.keywords, ensure_ascii=False),
                        "engagement": article.get("engagement", 0),
                        "url": article.get("url", ""),
                    })
                    progress.progress((i + 1) / len(articles))

                insert_sentiment(sentiment_records)
                st.cache_data.clear()
                sentiment_df = get_sentiment(stock_id, start_date, end_date)
                st.success(f"已分析 {len(sentiment_records)} 篇文章")
            else:
                st.warning("未爬取到任何文章，請確認 Firecrawl API Key 設定")

if sentiment_df.empty:
    st.info("尚無情緒資料。請點擊「爬取最新情緒」開始收集。")
    with st.expander("查看頁面功能說明"):
        st.markdown(
            "爬取完成後，此頁面將顯示：情緒趨勢、看多/看空比例、"
            "討論熱度、關鍵詞分析等"
        )
    st.stop()

# ── 每日聚合（使用快取版本）────────────────────────────────

_, daily_agg = _load_sentiment(stock_id, start_date.isoformat(), end_date.isoformat())
if daily_agg.empty:
    # 若剛爬取完，快取可能未命中，直接聚合
    daily_agg = sentiment_df.groupby("date").agg(
        avg_score=("sentiment_score", "mean"),
        post_count=("sentiment_score", "count"),
        bullish_count=("sentiment_label", lambda x: (x == "bullish").sum()),
        bearish_count=("sentiment_label", lambda x: (x == "bearish").sum()),
    ).reset_index()
    daily_agg["bullish_ratio"] = daily_agg["bullish_count"] / daily_agg["post_count"]
    daily_agg["bearish_ratio"] = daily_agg["bearish_count"] / daily_agg["post_count"]
    daily_agg["sentiment_score"] = daily_agg["avg_score"]
    daily_agg["sentiment_ma5"] = daily_agg["sentiment_score"].rolling(5, min_periods=1).mean()

# ── 總覽指標 ─────────────────────────────────────────────

latest_score = daily_agg["sentiment_score"].iloc[-1] if not daily_agg.empty else 0
total_posts = sentiment_df.shape[0]
bullish_total = (sentiment_df["sentiment_label"] == "bullish").sum()
bearish_total = (sentiment_df["sentiment_label"] == "bearish").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("最新情緒分數", f"{latest_score:.2f}")
col2.metric("總分析文章數", total_posts)
col3.metric("看多比例", f"{bullish_total/total_posts*100:.1f}%" if total_posts else "N/A")
col4.metric("看空比例", f"{bearish_total/total_posts*100:.1f}%" if total_posts else "N/A")

st.markdown("---")

# ── 情緒趨勢圖 ──────────────────────────────────────────

price_df = get_stock_prices(stock_id, start_date, end_date)

st.plotly_chart(
    create_sentiment_trend_chart(
        daily_agg[["date", "sentiment_score", "sentiment_ma5"]],
        price_df if not price_df.empty else None,
    ),
    use_container_width=True,
)
st.caption("情緒分數由 LLM 分析社群文章產生（-1 看空，+1 看多）")

with st.expander("詳細分析"):
    st.subheader("📈 討論熱度")
    fig_volume = px.bar(
        daily_agg, x="date", y="post_count",
        title="每日文章數量",
        color="sentiment_score",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
    )
    fig_volume.update_layout(template="twstock", height=300)
    st.plotly_chart(fig_volume, use_container_width=True)

    # 來源分布 + 關鍵詞
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 情緒來源分布")
        source_counts = sentiment_df["source"].value_counts()
        fig_pie = px.pie(
            values=source_counts.values,
            names=source_counts.index,
            title="文章來源分布",
        )
        fig_pie.update_layout(template="twstock", height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("🏷️ 關鍵詞分析")
        all_keywords = []
        for kw_json in sentiment_df["keywords"].dropna():
            try:
                kws = json.loads(kw_json) if isinstance(kw_json, str) else kw_json
                if isinstance(kws, list):
                    all_keywords.extend(kws)
            except (json.JSONDecodeError, TypeError):
                pass

        if all_keywords:
            kw_series = pd.Series(all_keywords).value_counts().head(15)
            fig_kw = px.bar(
                x=kw_series.values, y=kw_series.index,
                orientation="h",
                title="高頻關鍵詞 Top 15",
                labels={"x": "出現次數", "y": "關鍵詞"},
            )
            fig_kw.update_layout(template="twstock", height=350, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("尚無關鍵詞資料")

with st.expander("最新輿論摘要"):
    latest_articles = sentiment_df.sort_values("date", ascending=False).head(10)

    for _, row in latest_articles.iterrows():
        label = row.get("sentiment_label", "neutral")
        icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(label, "⚪")
        score = row.get("sentiment_score", 0)
        source = row.get("source", "")
        title = row.get("title", "（無標題）")

        st.markdown(f"{icon} **[{source}]** {title} (score: {score:.2f})")
