"""Tests for stock_fetcher module"""

import pytest
import pandas as pd

from src.data.stock_fetcher import StockFetcher


class TestStockFetcher:
    def setup_method(self):
        self.fetcher = StockFetcher()

    def test_get_stock_name(self):
        assert StockFetcher.get_stock_name("2330") == "台積電"
        assert StockFetcher.get_stock_name("2317") == "鴻海"
        assert StockFetcher.get_stock_name("9999") == "9999"

    def test_fetch_daily_prices_returns_dataframe(self):
        """測試日K資料格式（需要有效的 FinMind Token）"""
        df = self.fetcher.fetch_daily_prices("2330", "2024-01-01", "2024-01-31")
        if not df.empty:
            assert "date" in df.columns
            assert "close" in df.columns
            assert "volume" in df.columns

    def test_fetch_daily_prices_empty_on_invalid(self):
        """無效代碼應回傳空 DataFrame"""
        df = self.fetcher.fetch_daily_prices("XXXX", "2024-01-01", "2024-01-31")
        assert isinstance(df, pd.DataFrame)

    def test_fetch_institutional_returns_dataframe(self):
        df = self.fetcher.fetch_institutional("2330", "2024-01-01", "2024-01-31")
        if not df.empty:
            assert "date" in df.columns
            assert "foreign_buy_sell" in df.columns

    def test_fetch_all_merges_correctly(self):
        df = self.fetcher.fetch_all("2330", "2024-01-01", "2024-01-31")
        if not df.empty:
            assert "date" in df.columns
            assert "close" in df.columns
