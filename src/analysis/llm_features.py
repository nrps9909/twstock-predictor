"""LLM Embedding 特徵模組

將新聞/情緒文本用 LLM 產生 embedding，再用 PCA 降維至 10 維作為模型特徵。
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import settings

logger = logging.getLogger(__name__)


class LLMFeatureExtractor:
    """LLM embedding → PCA 降維特徵

    流程：
    1. 收集每日新聞/社群文本
    2. 用 Claude 或 OpenAI embedding API 轉換為向量
    3. PCA 降維至 n_components 維
    4. 與其他特徵合併
    """

    def __init__(
        self,
        n_components: int = 10,
        api_key: str | None = None,
        provider: str = "anthropic",
    ):
        self.n_components = n_components
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.provider = provider
        self.pca = None

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """取得文本 embedding

        Args:
            texts: 文本列表

        Returns:
            (n_texts, embedding_dim) ndarray
        """
        if not texts:
            return np.array([])

        if self.provider == "openai" and settings.OPENAI_API_KEY:
            return self._openai_embeddings(texts)

        # Fallback: 用 TF-IDF 作為簡易 embedding
        return self._tfidf_embeddings(texts)

    def _openai_embeddings(self, texts: list[str]) -> np.ndarray:
        """用 OpenAI embedding API"""
        try:
            import httpx
            client = httpx.Client(timeout=30)
            resp = client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                json={
                    "model": "text-embedding-3-small",
                    "input": texts[:100],  # 限制批次大小
                },
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = [item["embedding"] for item in data["data"]]
            return np.array(embeddings)
        except Exception as e:
            logger.warning("OpenAI embedding 失敗，使用 TF-IDF fallback: %s", e)
            return self._tfidf_embeddings(texts)

    def _tfidf_embeddings(self, texts: list[str]) -> np.ndarray:
        """簡易 TF-IDF embedding（不需 API）"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=100, analyzer="char_wb", ngram_range=(2, 4))
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            return tfidf_matrix.toarray()
        except Exception:
            return np.zeros((len(texts), 100))

    def fit_transform(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """取得 embedding 並 PCA 降維

        Returns:
            (n_texts, n_components)
        """
        from sklearn.decomposition import PCA

        embeddings = self.get_embeddings(texts)
        if embeddings.size == 0:
            return np.array([])

        n_samples = embeddings.shape[0]
        n_comp = min(self.n_components, n_samples, embeddings.shape[1])

        self.pca = PCA(n_components=n_comp)
        reduced = self.pca.fit_transform(embeddings)

        # 補齊至 n_components 維
        if reduced.shape[1] < self.n_components:
            padding = np.zeros((n_samples, self.n_components - reduced.shape[1]))
            reduced = np.hstack([reduced, padding])

        logger.info(
            "LLM 特徵: %d 文本 → %d 維 (explained variance: %.2f%%)",
            n_samples, self.n_components,
            sum(self.pca.explained_variance_ratio_) * 100,
        )
        return reduced

    def transform(self, texts: list[str]) -> np.ndarray:
        """用已 fit 的 PCA 轉換新文本"""
        if self.pca is None:
            raise RuntimeError("請先呼叫 fit_transform()")

        embeddings = self.get_embeddings(texts)
        if embeddings.size == 0:
            return np.zeros((len(texts), self.n_components))

        reduced = self.pca.transform(embeddings)
        if reduced.shape[1] < self.n_components:
            padding = np.zeros((len(texts), self.n_components - reduced.shape[1]))
            reduced = np.hstack([reduced, padding])
        return reduced

    def create_daily_features(
        self,
        sentiment_df: pd.DataFrame,
        text_col: str = "title",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """將每日文本彙整為 LLM 特徵

        Args:
            sentiment_df: 含文本和日期的 DataFrame
            text_col: 文本欄位名
            date_col: 日期欄位名

        Returns:
            DataFrame(date, llm_feat_0, ..., llm_feat_9)
        """
        if sentiment_df.empty:
            return pd.DataFrame()

        # 每日文本合併
        daily_texts = sentiment_df.groupby(date_col)[text_col].apply(
            lambda x: " ".join(x.dropna().astype(str))
        ).reset_index()

        texts = daily_texts[text_col].tolist()
        features = self.fit_transform(texts)

        if features.size == 0:
            return pd.DataFrame()

        feat_df = pd.DataFrame(
            features,
            columns=[f"llm_feat_{i}" for i in range(features.shape[1])],
        )
        feat_df[date_col] = daily_texts[date_col].values

        return feat_df
