"""全市場掃描引擎 — SSE 串流 + 20 因子評分 + 體制權重 + 多維度信心"""

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import AsyncGenerator

import numpy as np
import pandas as pd

from src.utils.constants import STOCK_LIST
from src.utils.config import settings
from src.data.twse_scanner import TWSEScanner
from src.data.stock_fetcher import StockFetcher
from src.db.database import (
    get_stock_prices, upsert_stock_prices,
    save_market_scan, get_latest_market_scan,
    save_factor_ic_records,
)
from src.analysis.technical import TechnicalAnalyzer

logger = logging.getLogger(__name__)

MODEL_DIR = settings.PROJECT_ROOT / "models"


# ═══════════════════════════════════════════════════════
# FactorResult dataclass
# ═══════════════════════════════════════════════════════


@dataclass
class FactorResult:
    """單一因子的計算結果"""
    name: str              # e.g. "technical_signal"
    score: float           # 0.0-1.0 (0.5=中性)
    available: bool        # True=有真實資料
    freshness: float       # 0.0-1.0 (1.0=今天)
    components: dict = field(default_factory=dict)  # 子因子明細
    raw_value: float | None = None  # IC 追蹤用原始值


# ═══════════════════════════════════════════════════════
# Phase 3: Weight Engine — 基礎權重 + 體制乘數 (20 因子)
# ═══════════════════════════════════════════════════════

BASE_WEIGHTS = {
    # 短期 (39%)
    "foreign_flow":       0.11,  # 外資籌碼流向
    "technical_signal":   0.08,  # 技術訊號聚合
    "short_momentum":     0.07,  # 短期動能
    "trust_flow":         0.05,  # 投信籌碼流向
    "volume_anomaly":     0.04,  # 量能異常
    "margin_sentiment":   0.04,  # 融資融券情緒
    # 中期 (32%)
    "trend_momentum":     0.07,  # 中期趨勢動能
    "revenue_momentum":   0.04,  # 月營收動能
    "institutional_sync": 0.04,  # 法人同步性
    "volatility_regime":  0.04,  # 波動率狀態
    "news_sentiment":     0.03,  # 新聞情緒
    "global_context":     0.03,  # 國際市場連動
    "margin_quality":     0.04,  # 季報毛利率趨勢
    "sector_rotation":    0.03,  # 產業資金輪動
    # 長期 (29%)
    "ml_ensemble":        0.07,  # ML 模型預測
    "fundamental_value":  0.06,  # 基本面價值
    "liquidity_quality":  0.04,  # 流動性品質
    "macro_risk":         0.04,  # 宏觀風險環境
    "export_momentum":    0.04,  # 台灣出口動能
    "us_manufacturing":   0.04,  # 美國製造業景氣
}  # sum = 1.00

REGIME_MULTIPLIERS = {
    "bull": {
        "foreign_flow": 1.0,   "technical_signal": 1.1, "short_momentum": 1.3,
        "trust_flow": 1.0,     "volume_anomaly": 1.1,   "margin_sentiment": 0.8,
        "trend_momentum": 1.3, "revenue_momentum": 1.0, "institutional_sync": 1.0,
        "volatility_regime": 0.7, "news_sentiment": 0.8, "global_context": 1.0,
        "margin_quality": 0.8, "sector_rotation": 1.2,
        "ml_ensemble": 1.0,   "fundamental_value": 0.8, "liquidity_quality": 0.8,
        "macro_risk": 0.8,    "export_momentum": 1.0,   "us_manufacturing": 0.8,
    },
    "bear": {
        "foreign_flow": 1.3,   "technical_signal": 0.8, "short_momentum": 0.5,
        "trust_flow": 1.2,     "volume_anomaly": 0.9,   "margin_sentiment": 1.5,
        "trend_momentum": 0.5, "revenue_momentum": 1.0, "institutional_sync": 1.2,
        "volatility_regime": 1.5, "news_sentiment": 1.0, "global_context": 1.2,
        "margin_quality": 1.2, "sector_rotation": 0.8,
        "ml_ensemble": 0.8,   "fundamental_value": 1.2, "liquidity_quality": 1.3,
        "macro_risk": 1.3,    "export_momentum": 1.2,   "us_manufacturing": 1.3,
    },
    "sideways": {
        "foreign_flow": 1.0,   "technical_signal": 1.3, "short_momentum": 0.8,
        "trust_flow": 1.0,     "volume_anomaly": 1.2,   "margin_sentiment": 1.0,
        "trend_momentum": 0.7, "revenue_momentum": 1.0, "institutional_sync": 1.0,
        "volatility_regime": 1.2, "news_sentiment": 1.2, "global_context": 1.0,
        "margin_quality": 1.0, "sector_rotation": 1.3,
        "ml_ensemble": 1.0,   "fundamental_value": 1.0, "liquidity_quality": 1.0,
        "macro_risk": 1.0,    "export_momentum": 1.0,   "us_manufacturing": 1.0,
    },
}

# ═══════════════════════════════════════════════════════
# 產業分類表 (sector_rotation 因子用)
# ═══════════════════════════════════════════════════════

STOCK_SECTOR = {
    # semiconductor
    "2330": "semiconductor", "2303": "semiconductor", "2454": "semiconductor",
    "3711": "semiconductor", "2379": "semiconductor", "3034": "semiconductor",
    "6770": "semiconductor", "2344": "semiconductor", "3529": "semiconductor",
    "5274": "semiconductor", "6505": "semiconductor", "3443": "semiconductor",
    "2449": "semiconductor", "3661": "semiconductor", "5347": "semiconductor",
    # electronics
    "2317": "electronics", "2382": "electronics", "2308": "electronics",
    "2301": "electronics", "3231": "electronics", "2395": "electronics",
    "2356": "electronics", "3044": "electronics", "2353": "electronics",
    "2327": "electronics", "6669": "electronics", "3706": "electronics",
    # finance
    "2881": "finance", "2882": "finance", "2886": "finance",
    "2884": "finance", "2891": "finance", "2892": "finance",
    "2880": "finance", "2883": "finance", "2885": "finance",
    "2887": "finance", "5880": "finance", "2890": "finance",
    # telecom
    "2412": "telecom", "3045": "telecom", "4904": "telecom",
    # traditional
    "1301": "traditional", "1303": "traditional", "1326": "traditional",
    "2002": "traditional", "1402": "traditional", "2105": "traditional",
    # shipping
    "2603": "shipping", "2609": "shipping", "2615": "shipping", "2618": "shipping",
    # biotech
    "6446": "biotech", "4743": "biotech", "1760": "biotech",
    # green_energy
    "6488": "green_energy", "3481": "green_energy",
}

DEFAULT_SECTOR = "other"


def _compute_weights(factors: list[FactorResult], regime: str) -> dict[str, float]:
    """計算各因子最終權重（含體制調整 + 缺資料重分配）"""
    multipliers = REGIME_MULTIPLIERS.get(regime, REGIME_MULTIPLIERS["sideways"])
    available = [f for f in factors if f.available]
    if not available:
        return {f.name: 1.0 / len(factors) for f in factors}

    raw = {}
    for f in available:
        base = BASE_WEIGHTS.get(f.name, 0.03)
        mult = multipliers.get(f.name, 1.0)
        raw[f.name] = base * mult

    total = sum(raw.values())
    if total == 0:
        return {f.name: 1.0 / len(available) for f in available}
    return {k: v / total for k, v in raw.items()}


# ═══════════════════════════════════════════════════════
# Global/Macro data caches (fetched once per day)
# ═══════════════════════════════════════════════════════

_global_cache: dict = {"date": None, "data": None}
_macro_cache: dict = {"date": None, "data": None}


def _fetch_global_market_data() -> dict:
    """取得前一交易日全球市場數據 (yfinance), 每日快取

    包含: SOX, TSM 日報酬 + EWT 1d/20d/60d 報酬 (出口動能用)
    """
    today = date.today()
    if _global_cache["date"] == today and _global_cache["data"] is not None:
        return _global_cache["data"]

    result = {}
    try:
        import yfinance as yf
        for ticker, key in [("^SOX", "sox"), ("TSM", "tsm")]:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="5d")
                if len(hist) >= 2:
                    prev_close = float(hist["Close"].iloc[-2])
                    last_close = float(hist["Close"].iloc[-1])
                    result[f"{key}_return"] = (last_close - prev_close) / prev_close
                else:
                    result[f"{key}_return"] = 0.0
            except Exception:
                result[f"{key}_return"] = 0.0

        # EWT (iShares MSCI Taiwan ETF) — 出口動能代理指標
        try:
            ewt = yf.Ticker("EWT")
            hist = ewt.history(period="90d")
            if len(hist) >= 2:
                last_close = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[-2])
                result["ewt_return_1d"] = (last_close - prev_close) / prev_close
            else:
                result["ewt_return_1d"] = 0.0
            if len(hist) >= 21:
                close_20d_ago = float(hist["Close"].iloc[-21])
                result["ewt_return_20d"] = (last_close - close_20d_ago) / close_20d_ago
            else:
                result["ewt_return_20d"] = 0.0
            if len(hist) >= 61:
                close_60d_ago = float(hist["Close"].iloc[-61])
                result["ewt_return_60d"] = (last_close - close_60d_ago) / close_60d_ago
            else:
                result["ewt_return_60d"] = 0.0
        except Exception:
            result["ewt_return_1d"] = 0.0
            result["ewt_return_20d"] = 0.0
            result["ewt_return_60d"] = 0.0

    except Exception:
        result["sox_return"] = 0.0
        result["tsm_return"] = 0.0
        result["ewt_return_1d"] = 0.0
        result["ewt_return_20d"] = 0.0
        result["ewt_return_60d"] = 0.0

    _global_cache["date"] = today
    _global_cache["data"] = result
    return result


def _fetch_macro_data() -> dict:
    """取得宏觀風險數據 (yfinance), 每日快取"""
    today = date.today()
    if _macro_cache["date"] == today and _macro_cache["data"] is not None:
        return _macro_cache["data"]

    result: dict = {}
    try:
        import yfinance as yf

        # VIX
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")
            if len(hist) >= 1:
                result["vix"] = float(hist["Close"].iloc[-1])
            else:
                result["vix"] = 20.0
        except Exception:
            result["vix"] = 20.0

        # USD/TWD 匯率 30 日趨勢
        try:
            fx = yf.Ticker("TWD=X")
            hist = fx.history(period="35d")
            if len(hist) >= 20:
                recent = float(hist["Close"].tail(5).mean())
                prior = float(hist["Close"].tail(30).head(10).mean())
                result["usdtwd_trend"] = (recent - prior) / prior
            else:
                result["usdtwd_trend"] = 0.0
        except Exception:
            result["usdtwd_trend"] = 0.0

        # 美國 10Y 殖利率 30 日變化
        try:
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="35d")
            if len(hist) >= 20:
                recent = float(hist["Close"].tail(5).mean())
                prior = float(hist["Close"].tail(30).head(10).mean())
                result["tnx_change"] = recent - prior
            else:
                result["tnx_change"] = 0.0
        except Exception:
            result["tnx_change"] = 0.0

        # XLI (工業 ETF) — 製造業景氣代理指標
        try:
            xli = yf.Ticker("XLI")
            hist = xli.history(period="250d")
            if len(hist) >= 21:
                last_close = float(hist["Close"].iloc[-1])
                close_20d_ago = float(hist["Close"].iloc[-21])
                result["xli_return_20d"] = (last_close - close_20d_ago) / close_20d_ago
            else:
                result["xli_return_20d"] = 0.0
            if len(hist) >= 200:
                sma200 = float(hist["Close"].tail(200).mean())
                last_close = float(hist["Close"].iloc[-1])
                result["xli_vs_sma200"] = (last_close - sma200) / sma200
            else:
                result["xli_vs_sma200"] = 0.0
        except Exception:
            result["xli_return_20d"] = 0.0
            result["xli_vs_sma200"] = 0.0

        # XLI/SPY 比率趨勢 — 製造業相對強度
        try:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="35d")
            xli_short = yf.Ticker("XLI").history(period="35d")
            if len(spy_hist) >= 21 and len(xli_short) >= 21:
                xli_now = float(xli_short["Close"].iloc[-1])
                spy_now = float(spy_hist["Close"].iloc[-1])
                xli_20d = float(xli_short["Close"].iloc[-21])
                spy_20d = float(spy_hist["Close"].iloc[-21])
                ratio_now = xli_now / spy_now if spy_now > 0 else 1.0
                ratio_20d = xli_20d / spy_20d if spy_20d > 0 else 1.0
                result["xli_spy_ratio_trend"] = (ratio_now - ratio_20d) / ratio_20d if ratio_20d > 0 else 0.0
            else:
                result["xli_spy_ratio_trend"] = 0.0
        except Exception:
            result["xli_spy_ratio_trend"] = 0.0

    except Exception:
        result.setdefault("vix", 20.0)
        result.setdefault("usdtwd_trend", 0.0)
        result.setdefault("tnx_change", 0.0)
        result.setdefault("xli_return_20d", 0.0)
        result.setdefault("xli_vs_sma200", 0.0)
        result.setdefault("xli_spy_ratio_trend", 0.0)

    _macro_cache["date"] = today
    _macro_cache["data"] = result
    return result


async def _fetch_revenue_batch(stock_ids: list[str]) -> dict[str, pd.DataFrame]:
    """批次取得最近15個月營收 (FinMind TaiwanStockMonthRevenue)"""
    result: dict[str, pd.DataFrame] = {}
    try:
        fetcher = StockFetcher()
        start = (date.today() - timedelta(days=450)).strftime("%Y-%m-%d")
        for sid in stock_ids:
            try:
                data = fetcher._query_finmind("TaiwanStockMonthRevenue", sid, start)
                if data:
                    result[sid] = pd.DataFrame(data)
            except Exception:
                pass
    except Exception:
        pass
    return result


# ═══════════════════════════════════════════════════════
# Phase 2: 20 Factor Computers
# ═══════════════════════════════════════════════════════


def _compute_foreign_flow(trust_info: dict, df: pd.DataFrame,
                          avg_vol_20d: float) -> FactorResult:
    """外資籌碼流向 (13%) — 標準化淨買超 + 連續天數 + 加速度 + 異常偵測"""
    foreign_net_5d = trust_info.get("foreign_cumulative", 0)
    foreign_consecutive = trust_info.get("foreign_consecutive_days", 0)

    has_data = bool(trust_info) or (
        not df.empty and "foreign_buy_sell" in df.columns and df["foreign_buy_sell"].notna().any()
    )
    if not has_data:
        return FactorResult("foreign_flow", 0.5, False, 0.0)

    components = {}

    # 1. 標準化淨買超 (40%)
    if avg_vol_20d > 0:
        normalized = foreign_net_5d / avg_vol_20d
        net_score = max(0, min(1, 0.5 + normalized * 0.2))
    else:
        net_score = 0.5
    components["net_normalized"] = round(net_score, 4)

    # 2. 連續天數 bonus (20%)
    consec_score = min(0.5 + foreign_consecutive * 0.1, 1.0)
    components["consecutive"] = round(consec_score, 4)

    # 3. 加速度 (20%): 近5日 vs 前5日
    accel_score = 0.5
    if not df.empty and len(df) >= 10 and "foreign_buy_sell" in df.columns:
        fbs = df["foreign_buy_sell"].fillna(0)
        recent = float(fbs.tail(5).sum())
        prior = float(fbs.iloc[-10:-5].sum())
        if abs(prior) > 50:
            accel = (recent - prior) / abs(prior)
            accel_score = max(0, min(1, 0.5 + accel * 0.25))
        else:
            accel_score = 0.6 if recent > 0 else 0.4
    components["acceleration"] = round(accel_score, 4)

    # 4. 異常大量偵測 (20%): Z-score
    anomaly_score = 0.5
    if not df.empty and len(df) >= 60 and "foreign_buy_sell" in df.columns:
        fbs_60 = df["foreign_buy_sell"].tail(60).dropna()
        if len(fbs_60) >= 20:
            mu, sigma = float(fbs_60.mean()), float(fbs_60.std())
            if sigma > 0:
                today_fbs = float(fbs_60.iloc[-1])
                z = (today_fbs - mu) / sigma
                anomaly_score = max(0, min(1, 0.5 + z * 0.15))
    components["anomaly_z"] = round(anomaly_score, 4)

    total = net_score * 0.40 + consec_score * 0.20 + accel_score * 0.20 + anomaly_score * 0.20
    return FactorResult("foreign_flow", round(total, 4), True, 1.0,
                        components, raw_value=total)


def _compute_technical_signal(signals: dict, df_tech: pd.DataFrame) -> FactorResult:
    """技術訊號聚合 (9%) — 訊號+ADX+MA排列+OBV背離 (沿用 technical_trend 邏輯)"""
    if not signals or df_tech.empty:
        return FactorResult("technical_signal", 0.5, False, 0.0)

    components = {}

    # 1. Signal score 40%
    summary = signals.get("summary", {})
    raw = summary.get("raw_score", 0)
    max_s = summary.get("max_score", 5)
    signal_score = (raw / max_s + 1) / 2 if max_s != 0 else 0.5
    components["signal"] = round(signal_score, 4)

    # 2. ADX trend strength 20%
    adx_val = df_tech["adx"].iloc[-1] if "adx" in df_tech.columns else None
    if adx_val is not None and not pd.isna(adx_val):
        if adx_val > 40:
            adx_score = 0.9
        elif adx_val > 25:
            adx_score = 0.7
        else:
            adx_score = 0.4
        if adx_val > 25 and signal_score > 0.5:
            adx_score = min(adx_score + 0.1, 1.0)
        elif adx_val > 25 and signal_score < 0.5:
            adx_score = max(adx_score - 0.1, 0.0)
    else:
        adx_score = 0.5
    components["adx"] = round(adx_score, 4)

    # 3. MA alignment 20%
    try:
        last = df_tech.iloc[-1]
        close = last.get("close", 0)
        sma5 = last.get("sma_5", 0)
        sma20 = last.get("sma_20", 0)
        sma60 = last.get("sma_60", 0)
        if close and sma5 and sma20 and sma60:
            if close > sma5 > sma20 > sma60:
                ma_score = 1.0
            elif close > sma5 > sma20:
                ma_score = 0.8
            elif close > sma20:
                ma_score = 0.65
            elif close < sma5 < sma20 < sma60:
                ma_score = 0.0
            elif close < sma5 < sma20:
                ma_score = 0.2
            elif close < sma20:
                ma_score = 0.35
            else:
                ma_score = 0.5
        else:
            ma_score = 0.5
    except Exception:
        ma_score = 0.5
    components["ma_alignment"] = round(ma_score, 4)

    # 4. OBV divergence 20%
    obv_score = 0.5
    if "obv" in df_tech.columns and len(df_tech) >= 20:
        try:
            close_20d = df_tech["close"].tail(20)
            obv_20d = df_tech["obv"].tail(20)
            price_up = close_20d.iloc[-1] > close_20d.iloc[0]
            obv_up = obv_20d.iloc[-1] > obv_20d.iloc[0]
            if price_up and not obv_up:
                obv_score = 0.3
            elif not price_up and obv_up:
                obv_score = 0.7
            elif price_up and obv_up:
                obv_score = 0.65
            else:
                obv_score = 0.35
        except Exception:
            pass
    components["obv_divergence"] = round(obv_score, 4)

    total = signal_score * 0.4 + adx_score * 0.2 + ma_score * 0.2 + obv_score * 0.2
    return FactorResult("technical_signal", round(total, 4), True, 1.0,
                        components, raw_value=total)


def _compute_short_momentum(df: pd.DataFrame) -> FactorResult:
    """短期動能 (8%) — 1-5天報酬 + 均線乖離率"""
    if df.empty or len(df) < 5:
        return FactorResult("short_momentum", 0.5, False, 0.0)

    close = df["close"].dropna()
    if len(close) < 5:
        return FactorResult("short_momentum", 0.5, False, 0.0)

    components = {}

    def ret_to_score(r, scale):
        return max(0, min(1, 0.5 + r * scale))

    # 1. Multi-timeframe short-term returns (60%)
    ret_1d = (close.iloc[-1] / close.iloc[-2]) - 1 if len(close) >= 2 else 0
    ret_3d = (close.iloc[-1] / close.iloc[-4]) - 1 if len(close) >= 4 else 0
    ret_5d = (close.iloc[-1] / close.iloc[-6]) - 1 if len(close) >= 6 else 0

    s1 = ret_to_score(ret_1d, 10.0)
    s3 = ret_to_score(ret_3d, 5.0)
    s5 = ret_to_score(ret_5d, 4.0)
    mtf = s1 * 0.30 + s3 * 0.35 + s5 * 0.35

    components["return_1d"] = round(float(ret_1d), 4)
    components["return_3d"] = round(float(ret_3d), 4)
    components["return_5d"] = round(float(ret_5d), 4)
    components["mtf_score"] = round(mtf, 4)

    # 2. 均線乖離率 (40%)
    if len(close) >= 5:
        sma3 = float(close.tail(3).mean())
        sma5 = float(close.tail(5).mean())
        c = float(close.iloc[-1])
        bias3 = (c - sma3) / sma3 if sma3 > 0 else 0
        bias5 = (c - sma5) / sma5 if sma5 > 0 else 0
        bias_score = max(0, min(1, 0.5 + (bias3 * 0.5 + bias5 * 0.5) * 8.0))
    else:
        bias_score = 0.5
    components["bias"] = round(bias_score, 4)

    total = mtf * 0.60 + bias_score * 0.40
    return FactorResult("short_momentum", round(total, 4), True, 1.0,
                        components, raw_value=total)


def _compute_trust_flow(trust_info: dict, df: pd.DataFrame,
                        avg_vol_20d: float) -> FactorResult:
    """投信籌碼流向 (6%) — 標準化淨買超 + 連續天數 + 加速度"""
    trust_net_5d = trust_info.get("trust_cumulative", 0)
    trust_consecutive = trust_info.get("trust_consecutive_days", 0)

    has_data = bool(trust_info) or (
        not df.empty and "trust_buy_sell" in df.columns and df["trust_buy_sell"].notna().any()
    )
    if not has_data:
        return FactorResult("trust_flow", 0.5, False, 0.0)

    components = {}

    # 1. 標準化淨買超 (40%)
    if avg_vol_20d > 0:
        normalized = trust_net_5d / avg_vol_20d
        net_score = max(0, min(1, 0.5 + normalized * 0.25))
    else:
        net_score = 0.5
    components["net_normalized"] = round(net_score, 4)

    # 2. 連續天數 (30%)
    consec_score = min(0.5 + trust_consecutive * 0.1, 1.0)
    components["consecutive"] = round(consec_score, 4)

    # 3. 加速度 (30%)
    accel_score = 0.5
    if not df.empty and len(df) >= 10 and "trust_buy_sell" in df.columns:
        tbs = df["trust_buy_sell"].fillna(0)
        recent = float(tbs.tail(5).sum())
        prior = float(tbs.iloc[-10:-5].sum())
        if abs(prior) > 20:
            accel = (recent - prior) / abs(prior)
            accel_score = max(0, min(1, 0.5 + accel * 0.25))
        else:
            accel_score = 0.6 if recent > 0 else 0.4
    components["acceleration"] = round(accel_score, 4)

    total = net_score * 0.40 + consec_score * 0.30 + accel_score * 0.30
    return FactorResult("trust_flow", round(total, 4), True, 1.0,
                        components, raw_value=total)


def _compute_volume_anomaly(df: pd.DataFrame, df_tech: pd.DataFrame) -> FactorResult:
    """量能異常 (5%) — 放大率 + 量價一致性 + OBV趨勢"""
    if df.empty or len(df) < 20:
        return FactorResult("volume_anomaly", 0.5, False, 0.0)

    vol = df["volume"].dropna()
    close = df["close"].dropna()
    if len(vol) < 20 or len(close) < 6:
        return FactorResult("volume_anomaly", 0.5, False, 0.0)

    components = {}

    # 1. 量能放大率 (50%) — 方向感知
    vol_5d = float(vol.tail(5).mean())
    vol_20d = float(vol.tail(20).mean())
    vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1.0
    ret_5d = (float(close.iloc[-1]) / float(close.iloc[-6])) - 1 if len(close) >= 6 else 0

    if ret_5d > 0:
        expansion_score = min(0.5 + (vol_ratio - 1) * 0.3, 0.95)
    else:
        expansion_score = max(0.5 - (vol_ratio - 1) * 0.3, 0.05)
    components["expansion"] = round(expansion_score, 4)
    components["vol_ratio"] = round(vol_ratio, 4)

    # 2. 量價一致性 (30%)
    if len(close) >= 6 and len(vol) >= 6:
        consistency_count = 0
        for i in range(-5, 0):
            p_chg = float(close.iloc[i]) - float(close.iloc[i - 1])
            v_chg = float(vol.iloc[i]) - float(vol.iloc[i - 1])
            if (p_chg > 0 and v_chg > 0) or (p_chg < 0 and v_chg < 0):
                consistency_count += 1
        consistency_score = 0.3 + consistency_count * 0.14
    else:
        consistency_score = 0.5
    components["consistency"] = round(consistency_score, 4)

    # 3. OBV 趨勢確認 (20%)
    obv_score = 0.5
    if "obv" in df_tech.columns and len(df_tech) >= 20:
        obv = df_tech["obv"].dropna()
        if len(obv) >= 20:
            obv_5d = float(obv.tail(5).mean())
            obv_20d = float(obv.tail(20).mean())
            if obv_5d > obv_20d * 1.02:
                obv_score = 0.7
            elif obv_5d < obv_20d * 0.98:
                obv_score = 0.3
    components["obv_trend"] = round(obv_score, 4)

    total = expansion_score * 0.50 + consistency_score * 0.30 + obv_score * 0.20
    return FactorResult("volume_anomaly", round(total, 4), True, 1.0,
                        components, raw_value=total)


def _compute_margin_sentiment(df: pd.DataFrame) -> FactorResult:
    """融資融券情緒 (5%) — 反向指標 (沿用 margin_retail 邏輯)"""
    if df.empty or "margin_balance" not in df.columns:
        return FactorResult("margin_sentiment", 0.5, False, 0.0)

    margin = df["margin_balance"].dropna()
    short = df["short_balance"].dropna() if "short_balance" in df.columns else pd.Series(dtype=float)

    if len(margin) < 5:
        return FactorResult("margin_sentiment", 0.5, False, 0.0)

    components = {}

    # 1. Margin trend 50% (INVERSE)
    recent_margin = float(margin.iloc[-1])
    margin_5d_ago = float(margin.iloc[-6]) if len(margin) >= 6 else float(margin.iloc[0])
    if margin_5d_ago > 0:
        margin_change = (recent_margin - margin_5d_ago) / margin_5d_ago
        margin_trend_score = max(0.0, min(1.0, 0.5 - margin_change * 3.0))
    else:
        margin_trend_score = 0.5
    components["margin_trend"] = round(margin_trend_score, 4)

    # 2. Margin utilization 30%
    if len(margin) >= 20:
        max_20d = float(margin.tail(20).max())
        if max_20d > 0:
            utilization = recent_margin / max_20d
            if utilization > 0.8:
                util_score = 0.2
            elif utilization > 0.6:
                util_score = 0.35
            elif utilization < 0.3:
                util_score = 0.7
            else:
                util_score = 0.5
        else:
            util_score = 0.5
    else:
        util_score = 0.5
    components["utilization"] = round(util_score, 4)

    # 3. Short/Margin ratio 20%
    if not short.empty and len(short) >= 1 and recent_margin > 0:
        short_ratio = float(short.iloc[-1]) / recent_margin
        if short_ratio > 0.20:
            short_score = 0.7
        elif short_ratio > 0.10:
            short_score = 0.6
        else:
            short_score = 0.5
    else:
        short_score = 0.5
    components["short_ratio"] = round(short_score, 4)

    total = margin_trend_score * 0.50 + util_score * 0.30 + short_score * 0.20
    return FactorResult("margin_sentiment", round(total, 4), True, 0.9,
                        components, raw_value=total)


def _compute_trend_momentum(df: pd.DataFrame, df_tech: pd.DataFrame) -> FactorResult:
    """中期趨勢動能 (8%) — 20d/60d報酬 + MA排列 + ADX確認"""
    if df.empty or len(df) < 21:
        return FactorResult("trend_momentum", 0.5, False, 0.0)

    close = df["close"].dropna()
    if len(close) < 21:
        return FactorResult("trend_momentum", 0.5, False, 0.0)

    components = {}

    # 1. 中期報酬 (40%)
    ret_20d = (float(close.iloc[-1]) / float(close.iloc[-21])) - 1
    ret_60d = (float(close.iloc[-1]) / float(close.iloc[-61])) - 1 if len(close) >= 61 else 0

    s20 = max(0, min(1, 0.5 + ret_20d * 3.0))
    s60 = max(0, min(1, 0.5 + ret_60d * 1.5))
    ret_score = s20 * 0.6 + s60 * 0.4
    components["return_20d"] = round(float(ret_20d), 4)
    components["return_60d"] = round(float(ret_60d), 4)
    components["ret_score"] = round(ret_score, 4)

    # 2. MA 排列 (30%)
    if len(close) >= 60:
        c = float(close.iloc[-1])
        sma5 = float(close.tail(5).mean())
        sma20 = float(close.tail(20).mean())
        sma60 = float(close.tail(60).mean())
        if c > sma5 > sma20 > sma60:
            ma_score = 1.0
        elif c > sma5 > sma20:
            ma_score = 0.8
        elif c > sma20:
            ma_score = 0.65
        elif c < sma5 < sma20 < sma60:
            ma_score = 0.0
        elif c < sma5 < sma20:
            ma_score = 0.2
        elif c < sma20:
            ma_score = 0.35
        else:
            ma_score = 0.5
    else:
        ma_score = 0.5
    components["ma_alignment"] = round(ma_score, 4)

    # 3. ADX 趨勢確認 (30%)
    adx_score = 0.5
    if "adx" in df_tech.columns and len(df_tech) >= 5:
        adx = float(df_tech["adx"].iloc[-1])
        if adx > 40:
            adx_score = 0.85
        elif adx > 25:
            adx_score = 0.65
        else:
            adx_score = 0.40
        if adx > 25 and ret_score > 0.55:
            adx_score = min(adx_score + 0.1, 1.0)
        elif adx > 25 and ret_score < 0.45:
            adx_score = max(adx_score - 0.1, 0.0)
    components["adx"] = round(adx_score, 4)

    total = ret_score * 0.40 + ma_score * 0.30 + adx_score * 0.30
    return FactorResult("trend_momentum", round(total, 4), True, 1.0,
                        components, raw_value=total)


def _compute_revenue_momentum(revenue_df: pd.DataFrame | None) -> FactorResult:
    """月營收動能 (5%) — YoY + YoY加速度 + MoM"""
    if revenue_df is None or revenue_df.empty or len(revenue_df) < 13:
        return FactorResult("revenue_momentum", 0.5, False, 0.0)

    components = {}
    rev = revenue_df.sort_values("date").copy()
    latest = float(rev["revenue"].iloc[-1])
    year_ago = float(rev["revenue"].iloc[-13])

    # 1. 最新月營收 YoY (50%)
    yoy = (latest - year_ago) / year_ago if year_ago > 0 else 0.0
    if yoy > 0.30:
        yoy_score = 0.85
    elif yoy > 0.15:
        yoy_score = 0.72
    elif yoy > 0.05:
        yoy_score = 0.60
    elif yoy > -0.05:
        yoy_score = 0.50
    elif yoy > -0.15:
        yoy_score = 0.38
    else:
        yoy_score = 0.22
    components["yoy"] = round(yoy, 4)
    components["yoy_score"] = round(yoy_score, 4)

    # 2. YoY 加速度 (30%)
    accel_score = 0.5
    if len(rev) >= 18:  # need at least 6+12 months lookback
        recent_yoys = []
        for i in [-1, -2, -3]:
            idx = len(rev) + i
            idx_ya = idx - 12
            if idx_ya >= 0:
                r = float(rev["revenue"].iloc[idx])
                r_ya = float(rev["revenue"].iloc[idx_ya])
                recent_yoys.append((r - r_ya) / r_ya if r_ya > 0 else 0)
        prior_yoys = []
        for i in [-4, -5, -6]:
            idx = len(rev) + i
            idx_ya = idx - 12
            if idx_ya >= 0:
                r = float(rev["revenue"].iloc[idx])
                r_ya = float(rev["revenue"].iloc[idx_ya])
                prior_yoys.append((r - r_ya) / r_ya if r_ya > 0 else 0)
        if recent_yoys and prior_yoys:
            accel = np.mean(recent_yoys) - np.mean(prior_yoys)
            accel_score = max(0, min(1, 0.5 + accel * 3.0))
    components["accel"] = round(accel_score, 4)

    # 3. MoM 月增率 (20%)
    mom_score = 0.5
    if len(rev) >= 2:
        prev_month = float(rev["revenue"].iloc[-2])
        if prev_month > 0:
            mom = (latest - prev_month) / prev_month
            mom_score = max(0, min(1, 0.5 + mom * 3.0))
    components["mom"] = round(mom_score, 4)

    total = yoy_score * 0.50 + accel_score * 0.30 + mom_score * 0.20
    return FactorResult("revenue_momentum", round(total, 4), True, 0.8,
                        components, raw_value=total)


def _compute_institutional_sync(trust_info: dict, df: pd.DataFrame) -> FactorResult:
    """法人同步性 (5%) — 外資投信同步 + 三法人合計 + 法人vs散戶"""
    foreign_cum = trust_info.get("foreign_cumulative", 0)
    trust_cum = trust_info.get("trust_cumulative", 0)
    dealer_cum = trust_info.get("dealer_cumulative", 0)

    if not trust_info:
        return FactorResult("institutional_sync", 0.5, False, 0.0)

    components = {}

    # 1. 外資投信同步 (40%)
    if foreign_cum > 0 and trust_cum > 0:
        sync_score = 0.85
    elif foreign_cum < 0 and trust_cum < 0:
        sync_score = 0.15
    elif foreign_cum > 0 and trust_cum < 0:
        sync_score = 0.55
    elif foreign_cum < 0 and trust_cum > 0:
        sync_score = 0.45
    else:
        sync_score = 0.50
    components["foreign_trust_sync"] = round(sync_score, 4)

    # 2. 三法人合計方向 (30%)
    positive_count = sum(1 for x in [foreign_cum, trust_cum, dealer_cum] if x > 0)
    if positive_count == 3:
        direction_score = 0.90
    elif positive_count == 2:
        direction_score = 0.65
    elif positive_count == 1:
        direction_score = 0.35
    else:
        direction_score = 0.10
    components["direction"] = round(direction_score, 4)

    # 3. 法人 vs 散戶背離 (30%)
    institutional_net = foreign_cum + trust_cum
    diverge_score = 0.5
    if not df.empty and len(df) >= 5 and "margin_balance" in df.columns:
        margin = df["margin_balance"].dropna()
        if len(margin) >= 5:
            margin_chg = float(margin.iloc[-1] - margin.iloc[-5])
            if institutional_net > 0 and margin_chg < 0:
                diverge_score = 0.85
            elif institutional_net < 0 and margin_chg > 0:
                diverge_score = 0.15
            elif institutional_net > 0 and margin_chg > 0:
                diverge_score = 0.60
            else:
                diverge_score = 0.40
    components["divergence"] = round(diverge_score, 4)

    total = sync_score * 0.40 + direction_score * 0.30 + diverge_score * 0.30
    return FactorResult("institutional_sync", round(total, 4), True, 1.0,
                        components, raw_value=total)


def _compute_volatility_regime(df: pd.DataFrame, df_tech: pd.DataFrame) -> FactorResult:
    """波動率狀態 (5%) — 沿用 volatility 邏輯"""
    if df.empty or len(df) < 20:
        return FactorResult("volatility_regime", 0.5, False, 0.0)

    close = df["close"].dropna()
    if len(close) < 20:
        return FactorResult("volatility_regime", 0.5, False, 0.0)

    components = {}

    # 1. Low volatility premium 40%
    returns = close.pct_change().dropna()
    vol_20d = float(returns.tail(20).std()) * (252 ** 0.5)
    if vol_20d < 0.20:
        vol_score = 1.0
    elif vol_20d < 0.30:
        vol_score = 0.7
    elif vol_20d < 0.40:
        vol_score = 0.5
    elif vol_20d < 0.55:
        vol_score = 0.3
    else:
        vol_score = 0.0
    components["low_vol_premium"] = round(vol_score, 4)
    components["vol_20d_annualized"] = round(vol_20d, 4)

    # 2. Volatility compression/expansion 30%
    if len(returns) >= 20:
        vol_5d = float(returns.tail(5).std()) * (252 ** 0.5) if len(returns) >= 5 else vol_20d
        vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1.0
        if vol_ratio < 0.7:
            compress_score = 0.65
        elif vol_ratio < 1.0:
            compress_score = 0.55
        elif vol_ratio < 1.5:
            compress_score = 0.45
        else:
            compress_score = 0.3
    else:
        compress_score = 0.5
    components["compression"] = round(compress_score, 4)

    # 3. BB width percentile 30%
    bb_score = 0.5
    if "bb_width" in df_tech.columns and len(df_tech) >= 60:
        bb = df_tech["bb_width"].dropna()
        if len(bb) >= 20:
            current_bb = float(bb.iloc[-1])
            bb_60d = bb.tail(60)
            percentile = float((bb_60d < current_bb).sum() / len(bb_60d))
            bb_score = max(0.0, min(1.0, 1.0 - percentile))
    components["bb_percentile"] = round(bb_score, 4)

    total = vol_score * 0.40 + compress_score * 0.30 + bb_score * 0.30
    return FactorResult("volatility_regime", round(total, 4), True, 1.0,
                        components, raw_value=total)


def _compute_news_sentiment(sentiment_scores: dict, sentiment_df: pd.DataFrame | None,
                            stock_id: str) -> FactorResult:
    """新聞情緒 (4%) — 沿用 sentiment 邏輯"""
    if stock_id not in sentiment_scores:
        return FactorResult("news_sentiment", 0.5, False, 0.0)

    components = {}

    # 1. Source-weighted score 40%
    if sentiment_df is not None and not sentiment_df.empty:
        source_weights = {"cnyes": 0.35, "yahoo": 0.25, "ptt": 0.25, "google": 0.15}
        weighted_sum = 0.0
        weight_total = 0.0
        for source, w in source_weights.items():
            src_df = sentiment_df[sentiment_df["source"] == source] if "source" in sentiment_df.columns else pd.DataFrame()
            if not src_df.empty:
                src_scores = src_df["sentiment_score"].dropna()
                if not src_scores.empty:
                    avg = float(src_scores.mean())
                    weighted_sum += (avg + 1) / 2 * w
                    weight_total += w
        source_score = weighted_sum / weight_total if weight_total > 0 else sentiment_scores[stock_id]
    else:
        source_score = sentiment_scores[stock_id]
    components["source_weighted"] = round(source_score, 4)

    # 2. Sentiment momentum 30%
    momentum_score = 0.5
    if sentiment_df is not None and not sentiment_df.empty and len(sentiment_df) >= 10:
        scores = sentiment_df["sentiment_score"].dropna()
        if len(scores) >= 10:
            recent_5 = float(scores.tail(5).mean())
            early_5 = float(scores.head(5).mean())
            delta = recent_5 - early_5
            momentum_score = max(0.0, min(1.0, 0.5 + delta * 1.5))
    components["momentum"] = round(momentum_score, 4)

    # 3. Engagement anomaly 30%
    engage_score = 0.5
    if sentiment_df is not None and not sentiment_df.empty and "engagement" in sentiment_df.columns:
        engagement = sentiment_df["engagement"].dropna()
        if len(engagement) >= 5:
            recent_engage = float(engagement.tail(5).mean())
            avg_engage = float(engagement.mean())
            if avg_engage > 0:
                ratio = recent_engage / avg_engage
                if ratio > 2.0:
                    engage_score = 0.7 if source_score > 0.5 else 0.3
                elif ratio > 1.3:
                    engage_score = 0.6 if source_score > 0.5 else 0.4
    components["engagement"] = round(engage_score, 4)

    freshness = 0.5
    if sentiment_df is not None and not sentiment_df.empty and "date" in sentiment_df.columns:
        try:
            latest_date = pd.to_datetime(sentiment_df["date"]).max()
            days_old = (pd.Timestamp.now() - latest_date).days
            freshness = max(0.0, min(1.0, 1.0 - days_old / 14))
        except Exception:
            freshness = 0.5

    total = source_score * 0.40 + momentum_score * 0.30 + engage_score * 0.30
    return FactorResult("news_sentiment", round(total, 4), True, round(freshness, 2),
                        components, raw_value=total)


def _compute_global_context(global_data: dict | None) -> FactorResult:
    """國際市場連動 (3%) — SOX + TSM"""
    if global_data is None:
        return FactorResult("global_context", 0.5, False, 0.0)

    components = {}
    sox_ret = global_data.get("sox_return", 0)
    tsm_ret = global_data.get("tsm_return", 0)

    def ret_to_score(r):
        return max(0.05, min(0.95, 0.5 + r * 17.5))

    sox_score = ret_to_score(sox_ret)
    tsm_score = ret_to_score(tsm_ret)

    components["sox_return"] = round(sox_ret, 4)
    components["tsm_return"] = round(tsm_ret, 4)
    components["sox_score"] = round(sox_score, 4)
    components["tsm_score"] = round(tsm_score, 4)

    total = sox_score * 0.60 + tsm_score * 0.40

    available = sox_ret != 0 or tsm_ret != 0
    return FactorResult("global_context", round(total, 4), available, 0.9,
                        components, raw_value=total)


def _compute_ml_ensemble(ml_scores: dict, stock_id: str) -> FactorResult:
    """ML 集成 (8%) — signal 映射"""
    if stock_id not in ml_scores:
        return FactorResult("ml_ensemble", 0.5, False, 0.0)

    score = ml_scores[stock_id]
    return FactorResult("ml_ensemble", round(score, 4), True, 1.0,
                        {"raw_ml_score": round(score, 4)}, raw_value=score)


def _compute_fundamental_value(stock_id: str) -> FactorResult:
    """基本面價值 (7%) — P/E + ROE + 殖利率"""
    components = {}

    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{stock_id}.TW")
        info = ticker.info or {}
    except Exception:
        return FactorResult("fundamental_value", 0.5, False, 0.0)

    # 1. P/E 風險過濾 (40%)
    pe = info.get("trailingPE")
    if pe is not None:
        components["pe_ratio"] = round(pe, 2)
        if pe > 80:
            pe_score = 0.30
        elif pe > 50:
            pe_score = 0.38
        elif pe < 0:
            pe_score = 0.30
        elif pe < 5:
            pe_score = 0.42
        else:
            pe_score = 0.50
    else:
        pe_score = 0.50

    # 2. ROE 穩健度 (35%)
    roe = info.get("returnOnEquity")
    if roe is not None:
        components["roe"] = round(roe, 4)
        if roe > 0.25:
            roe_score = 0.85
        elif roe > 0.15:
            roe_score = 0.72
        elif roe > 0.08:
            roe_score = 0.58
        elif roe > 0:
            roe_score = 0.42
        else:
            roe_score = 0.25
    else:
        roe_score = 0.50

    # 3. 殖利率 (25%)
    div_yield = info.get("dividendYield")
    if div_yield is not None:
        components["dividend_yield"] = round(div_yield, 4)
        if div_yield > 0.06:
            div_score = 0.80
        elif div_yield > 0.04:
            div_score = 0.68
        elif div_yield > 0.02:
            div_score = 0.55
        elif div_yield > 0:
            div_score = 0.45
        else:
            div_score = 0.40
    else:
        div_score = 0.50

    total = pe_score * 0.40 + roe_score * 0.35 + div_score * 0.25

    available = bool(pe is not None or roe is not None or div_yield is not None)
    return FactorResult("fundamental_value", round(total, 4), available,
                        0.7 if available else 0.0, components, raw_value=total)


def _compute_liquidity_quality(df: pd.DataFrame) -> FactorResult:
    """流動性品質 (5%) — 沿用 liquidity 邏輯"""
    if df.empty or len(df) < 20:
        return FactorResult("liquidity_quality", 0.5, False, 0.0)

    vol = df["volume"].dropna()
    if len(vol) < 20:
        return FactorResult("liquidity_quality", 0.5, False, 0.0)

    components = {}

    # 1. Average daily volume 50%
    avg_vol = float(vol.tail(20).mean())
    vol_score = min(avg_vol / 5000, 1.0)
    components["avg_volume_20d"] = round(avg_vol, 0)
    components["vol_score"] = round(vol_score, 4)

    # 2. Volume stability 25%
    vol_std = float(vol.tail(20).std())
    cv = vol_std / avg_vol if avg_vol > 0 else 1.0
    if cv < 0.3:
        stability_score = 0.9
    elif cv < 0.5:
        stability_score = 0.7
    elif cv < 0.8:
        stability_score = 0.5
    else:
        stability_score = 0.3
    components["stability"] = round(stability_score, 4)

    # 3. Spread proxy 25%
    if "high" in df.columns and "low" in df.columns:
        spread_df = ((df["high"] - df["low"]) / df["close"]).dropna().tail(20)
        if not spread_df.empty:
            avg_spread = float(spread_df.mean())
            if avg_spread < 0.01:
                spread_score = 0.9
            elif avg_spread < 0.02:
                spread_score = 0.7
            elif avg_spread < 0.04:
                spread_score = 0.5
            else:
                spread_score = 0.3
        else:
            spread_score = 0.5
    else:
        spread_score = 0.5
    components["spread_proxy"] = round(spread_score, 4)

    total = vol_score * 0.50 + stability_score * 0.25 + spread_score * 0.25
    return FactorResult("liquidity_quality", round(total, 4), True, 1.0,
                        components, raw_value=total)


def _compute_macro_risk(macro_data: dict | None) -> FactorResult:
    """宏觀風險環境 (4%) — VIX + USD/TWD + 美10Y"""
    if macro_data is None:
        return FactorResult("macro_risk", 0.5, False, 0.0)

    components = {}

    # 1. VIX 恐慌指標 (40%)
    vix = macro_data.get("vix", 20)
    if vix < 15:
        vix_score = 0.80
    elif vix < 20:
        vix_score = 0.65
    elif vix < 25:
        vix_score = 0.45
    elif vix < 30:
        vix_score = 0.30
    else:
        vix_score = 0.15
    components["vix"] = round(vix, 2)
    components["vix_score"] = round(vix_score, 4)

    # 2. USD/TWD 趨勢 (30%)
    fx_trend = macro_data.get("usdtwd_trend", 0)
    fx_score = max(0.1, min(0.9, 0.5 - fx_trend * 15.0))
    components["usdtwd_trend"] = round(fx_trend, 4)
    components["fx_score"] = round(fx_score, 4)

    # 3. 美國10Y殖利率變化 (30%)
    tnx_chg = macro_data.get("tnx_change", 0)
    tnx_score = max(0.1, min(0.9, 0.5 - tnx_chg * 1.5))
    components["tnx_change"] = round(tnx_chg, 4)
    components["tnx_score"] = round(tnx_score, 4)

    total = vix_score * 0.40 + fx_score * 0.30 + tnx_score * 0.30

    available = vix != 20.0 or fx_trend != 0 or tnx_chg != 0
    return FactorResult("macro_risk", round(total, 4), available, 0.9,
                        components, raw_value=total)


def _compute_margin_quality(stock_id: str) -> FactorResult:
    """季報毛利率/營益率趨勢 (4%) — yfinance 季報數據"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{stock_id}.TW")

        # Try quarterly income statement first
        try:
            qis = ticker.quarterly_income_stmt
            if qis is not None and not qis.empty and qis.shape[1] >= 2:
                components = {}

                # Parse gross margin from quarterly data
                gross_profit = None
                total_revenue = None
                operating_income = None

                for label in ["Gross Profit", "GrossProfit"]:
                    if label in qis.index:
                        gross_profit = qis.loc[label]
                        break
                for label in ["Total Revenue", "TotalRevenue"]:
                    if label in qis.index:
                        total_revenue = qis.loc[label]
                        break
                for label in ["Operating Income", "OperatingIncome"]:
                    if label in qis.index:
                        operating_income = qis.loc[label]
                        break

                if gross_profit is not None and total_revenue is not None:
                    # Latest quarter gross margin
                    gm_latest = float(gross_profit.iloc[0]) / float(total_revenue.iloc[0]) \
                        if float(total_revenue.iloc[0]) != 0 else 0
                    gm_prev = float(gross_profit.iloc[1]) / float(total_revenue.iloc[1]) \
                        if float(total_revenue.iloc[1]) != 0 else 0

                    gm_qoq_change = gm_latest - gm_prev

                    # YoY if 5+ quarters
                    gm_yoy_change = 0.0
                    if qis.shape[1] >= 5:
                        gm_yoy = float(gross_profit.iloc[4]) / float(total_revenue.iloc[4]) \
                            if float(total_revenue.iloc[4]) != 0 else 0
                        gm_yoy_change = gm_latest - gm_yoy

                    # Scoring: 毛利率趨勢 (60%)
                    trend_score = max(0.0, min(1.0, 0.5 + gm_qoq_change * 8.0 + gm_yoy_change * 4.0))
                    components["gm_latest"] = round(gm_latest, 4)
                    components["gm_qoq_change"] = round(gm_qoq_change, 4)
                    components["gm_yoy_change"] = round(gm_yoy_change, 4)
                    components["trend_score"] = round(trend_score, 4)

                    # 營益率水準 (40%)
                    op_margin = 0.0
                    if operating_income is not None and float(total_revenue.iloc[0]) != 0:
                        op_margin = float(operating_income.iloc[0]) / float(total_revenue.iloc[0])
                    if op_margin > 0.20:
                        level_score = 0.85
                    elif op_margin > 0.10:
                        level_score = 0.70
                    elif op_margin > 0.05:
                        level_score = 0.55
                    elif op_margin > 0:
                        level_score = 0.42
                    else:
                        level_score = 0.25
                    components["op_margin"] = round(op_margin, 4)
                    components["level_score"] = round(level_score, 4)

                    total = trend_score * 0.60 + level_score * 0.40
                    return FactorResult("margin_quality", round(total, 4), True, 0.6,
                                        components, raw_value=total)
        except Exception:
            pass

        # Fallback: ticker.info
        info = ticker.info or {}
        gm = info.get("grossMargins")
        om = info.get("operatingMargins")
        if gm is not None or om is not None:
            components = {}
            gm_score = 0.5
            if gm is not None:
                components["grossMargins"] = round(gm, 4)
                gm_score = max(0.0, min(1.0, 0.3 + gm * 1.0))
            om_score = 0.5
            if om is not None:
                components["operatingMargins"] = round(om, 4)
                if om > 0.20:
                    om_score = 0.85
                elif om > 0.10:
                    om_score = 0.70
                elif om > 0.05:
                    om_score = 0.55
                elif om > 0:
                    om_score = 0.42
                else:
                    om_score = 0.25
            total = gm_score * 0.60 + om_score * 0.40
            return FactorResult("margin_quality", round(total, 4), True, 0.4,
                                components, raw_value=total)

    except Exception:
        pass

    return FactorResult("margin_quality", 0.5, False, 0.0)


def _compute_sector_aggregates(stock_dfs: dict[str, pd.DataFrame],
                                trust_lookup: dict) -> dict[str, dict]:
    """預計算各產業聚合數據 (sector_rotation 因子用)

    Returns: {sector_name: {net_flow, avg_return_20d, breadth}}
    """
    from collections import defaultdict
    sector_stocks = defaultdict(list)

    for sid in stock_dfs:
        sector = STOCK_SECTOR.get(sid, DEFAULT_SECTOR)
        sector_stocks[sector].append(sid)

    sector_data = {}
    all_flows = []
    all_returns = []

    for sector, sids in sector_stocks.items():
        net_flow = 0.0
        returns_20d = []
        positive_flow_count = 0
        total_count = 0

        for sid in sids:
            ti = trust_lookup.get(sid, {})
            foreign_cum = ti.get("foreign_cumulative", 0)
            trust_cum = ti.get("trust_cumulative", 0)
            stock_flow = foreign_cum + trust_cum
            net_flow += stock_flow
            total_count += 1
            if stock_flow > 0:
                positive_flow_count += 1

            df = stock_dfs[sid]
            if not df.empty and len(df) >= 21:
                close = df["close"].dropna()
                if len(close) >= 21:
                    ret = (float(close.iloc[-1]) / float(close.iloc[-21])) - 1
                    returns_20d.append(ret)

        avg_return = float(np.mean(returns_20d)) if returns_20d else 0.0
        breadth = positive_flow_count / total_count if total_count > 0 else 0.5

        sector_data[sector] = {
            "net_flow": net_flow,
            "avg_return_20d": avg_return,
            "breadth": breadth,
            "stock_count": total_count,
        }
        all_flows.append(net_flow)
        all_returns.append(avg_return)

    # Market average
    market_avg_flow = float(np.mean(all_flows)) if all_flows else 0.0
    market_avg_return = float(np.mean(all_returns)) if all_returns else 0.0
    sector_data["_market_avg"] = {
        "net_flow": market_avg_flow,
        "avg_return_20d": market_avg_return,
    }

    return sector_data


def _compute_sector_rotation(stock_id: str, sector_data: dict | None) -> FactorResult:
    """產業資金輪動 (3%) — 產業法人流向/動能/廣度 vs 市場平均"""
    if sector_data is None or not sector_data:
        return FactorResult("sector_rotation", 0.5, False, 0.0)

    sector = STOCK_SECTOR.get(stock_id, DEFAULT_SECTOR)
    s_data = sector_data.get(sector)
    market_avg = sector_data.get("_market_avg", {})

    if s_data is None:
        return FactorResult("sector_rotation", 0.5, False, 0.0)

    components = {}
    mkt_flow = market_avg.get("net_flow", 0)
    mkt_return = market_avg.get("avg_return_20d", 0)

    # 1. 產業法人流向 vs 市場平均 (50%)
    flow_diff = s_data["net_flow"] - mkt_flow
    flow_denom = max(abs(mkt_flow), 1000.0)
    flow_score = max(0.0, min(1.0, 0.5 + (flow_diff / flow_denom) * 0.3))
    components["flow_vs_market"] = round(flow_score, 4)
    components["sector"] = sector

    # 2. 產業相對動能 vs 市場平均 (30%)
    ret_diff = s_data["avg_return_20d"] - mkt_return
    ret_score = max(0.0, min(1.0, 0.5 + ret_diff * 5.0))
    components["return_vs_market"] = round(ret_score, 4)

    # 3. 產業廣度 (20%)
    breadth = s_data["breadth"]
    breadth_score = max(0.0, min(1.0, breadth))
    components["breadth"] = round(breadth_score, 4)

    total = flow_score * 0.50 + ret_score * 0.30 + breadth_score * 0.20
    return FactorResult("sector_rotation", round(total, 4), True, 1.0,
                        components, raw_value=total)


def _compute_export_momentum(global_data: dict | None) -> FactorResult:
    """台灣出口動能 (4%) — EWT 20d/60d 報酬 + EWT vs SOX 相對強度"""
    if global_data is None:
        return FactorResult("export_momentum", 0.5, False, 0.0)

    ewt_1d = global_data.get("ewt_return_1d")
    ewt_20d = global_data.get("ewt_return_20d")
    ewt_60d = global_data.get("ewt_return_60d")
    sox_1d = global_data.get("sox_return", 0)

    if ewt_20d is None:
        return FactorResult("export_momentum", 0.5, False, 0.0)

    components = {}

    # 1. EWT 20d 報酬 (50%)
    s20 = max(0.0, min(1.0, 0.5 + ewt_20d * 4.0))
    components["ewt_return_20d"] = round(ewt_20d, 4)
    components["ewt_20d_score"] = round(s20, 4)

    # 2. EWT 60d 報酬 (30%)
    s60 = 0.5
    if ewt_60d is not None:
        s60 = max(0.0, min(1.0, 0.5 + ewt_60d * 2.0))
        components["ewt_return_60d"] = round(ewt_60d, 4)
    components["ewt_60d_score"] = round(s60, 4)

    # 3. EWT vs SOX 相對強度 (20%)
    ewt_1d_val = ewt_1d if ewt_1d is not None else 0.0
    relative = ewt_1d_val - sox_1d
    s_rel = max(0.0, min(1.0, 0.5 + relative * 10.0))
    components["relative_strength"] = round(s_rel, 4)

    total = s20 * 0.50 + s60 * 0.30 + s_rel * 0.20
    available = ewt_20d != 0 or (ewt_60d is not None and ewt_60d != 0)
    return FactorResult("export_momentum", round(total, 4), available, 0.9,
                        components, raw_value=total)


def _compute_us_manufacturing(macro_data: dict | None) -> FactorResult:
    """美國製造業景氣 (4%) — XLI 20d報酬 + XLI/SPY比率趨勢 + XLI vs 200d SMA"""
    if macro_data is None:
        return FactorResult("us_manufacturing", 0.5, False, 0.0)

    xli_ret = macro_data.get("xli_return_20d")
    xli_sma = macro_data.get("xli_vs_sma200")
    xli_spy = macro_data.get("xli_spy_ratio_trend")

    if xli_ret is None:
        return FactorResult("us_manufacturing", 0.5, False, 0.0)

    components = {}

    # 1. XLI 20d 報酬 (40%)
    s_ret = max(0.0, min(1.0, 0.5 + xli_ret * 4.0))
    components["xli_return_20d"] = round(xli_ret, 4)
    components["xli_ret_score"] = round(s_ret, 4)

    # 2. XLI/SPY 比率趨勢 (40%)
    s_ratio = 0.5
    if xli_spy is not None:
        s_ratio = max(0.0, min(1.0, 0.5 + xli_spy * 8.0))
        components["xli_spy_ratio_trend"] = round(xli_spy, 4)
    components["ratio_score"] = round(s_ratio, 4)

    # 3. XLI vs 200d SMA (20%)
    s_sma = 0.5
    if xli_sma is not None:
        if xli_sma > 0.05:
            s_sma = 0.75
        elif xli_sma > 0:
            s_sma = 0.60
        elif xli_sma > -0.05:
            s_sma = 0.40
        else:
            s_sma = 0.25
        components["xli_vs_sma200"] = round(xli_sma, 4)
    components["sma_score"] = round(s_sma, 4)

    total = s_ret * 0.40 + s_ratio * 0.40 + s_sma * 0.20
    available = xli_ret != 0 or (xli_sma is not None and xli_sma != 0)
    return FactorResult("us_manufacturing", round(total, 4), available, 0.9,
                        components, raw_value=total)


# ── Legacy aliases for backward compatibility with tests/imports ──

def _compute_technical_trend(signals: dict, df_tech: pd.DataFrame) -> FactorResult:
    """Legacy alias → technical_signal"""
    r = _compute_technical_signal(signals, df_tech)
    return FactorResult("technical_trend", r.score, r.available, r.freshness,
                        r.components, r.raw_value)


def _compute_momentum(df: pd.DataFrame) -> FactorResult:
    """Legacy alias → short_momentum"""
    r = _compute_short_momentum(df)
    return FactorResult("momentum", r.score, r.available, r.freshness,
                        r.components, r.raw_value)


def _compute_institutional_flow(trust_info: dict, df: pd.DataFrame) -> FactorResult:
    """Legacy alias → foreign_flow (uses avg_vol_20d=10000 fallback)"""
    avg_vol = 10000.0
    if not df.empty and "volume" in df.columns:
        vol = df["volume"].dropna()
        if len(vol) >= 20:
            avg_vol = float(vol.tail(20).mean())
    r = _compute_foreign_flow(trust_info, df, avg_vol)
    return FactorResult("institutional_flow", r.score, r.available, r.freshness,
                        r.components, r.raw_value)


def _compute_margin_retail(df: pd.DataFrame) -> FactorResult:
    """Legacy alias → margin_sentiment"""
    r = _compute_margin_sentiment(df)
    return FactorResult("margin_retail", r.score, r.available, r.freshness,
                        r.components, r.raw_value)


def _compute_volatility(df: pd.DataFrame, df_tech: pd.DataFrame) -> FactorResult:
    """Legacy alias → volatility_regime"""
    r = _compute_volatility_regime(df, df_tech)
    return FactorResult("volatility", r.score, r.available, r.freshness,
                        r.components, r.raw_value)


def _compute_sentiment(sentiment_scores: dict, sentiment_df: pd.DataFrame | None,
                       stock_id: str) -> FactorResult:
    """Legacy alias → news_sentiment"""
    r = _compute_news_sentiment(sentiment_scores, sentiment_df, stock_id)
    return FactorResult("sentiment", r.score, r.available, r.freshness,
                        r.components, r.raw_value)


def _compute_liquidity(df: pd.DataFrame) -> FactorResult:
    """Legacy alias → liquidity_quality"""
    r = _compute_liquidity_quality(df)
    return FactorResult("liquidity", r.score, r.available, r.freshness,
                        r.components, r.raw_value)


def _compute_value_quality(stock_id: str) -> FactorResult:
    """Legacy alias → fundamental_value"""
    r = _compute_fundamental_value(stock_id)
    return FactorResult("value_quality", r.score, r.available, r.freshness,
                        r.components, r.raw_value)


# ═══════════════════════════════════════════════════════
# Phase 4: Confidence Calculator + Risk Discount
# ═══════════════════════════════════════════════════════

# Risk discount thresholds
RISK_DISCOUNTS = [
    # (condition_fn, discount, description)
    # These are evaluated per-stock using factor results and raw data
]


def _compute_confidence(factors: list[FactorResult], weights: dict[str, float],
                        total_score: float, df: pd.DataFrame) -> dict:
    """多維度信心計算 + 風險折扣

    confidence = (agreement*0.30 + strength*0.30 + coverage*0.25 + freshness*0.15) × risk_discount
    """
    available_factors = [f for f in factors if f.available]

    # 1. Factor agreement 30%
    if available_factors:
        bullish_count = sum(1 for f in available_factors if f.score > 0.55)
        bearish_count = sum(1 for f in available_factors if f.score < 0.45)
        total_available = len(available_factors)
        max_direction = max(bullish_count, bearish_count)
        agreement = max_direction / total_available if total_available > 0 else 0.5
    else:
        agreement = 0.0

    # 2. Signal strength 30%
    strength = abs(total_score - 0.5) * 2

    # 3. Data coverage 25%: sum of base weights for available factors
    coverage = sum(BASE_WEIGHTS.get(f.name, 0) for f in available_factors)

    # 4. Data freshness 15%: weighted avg of freshness
    if available_factors:
        total_w = sum(weights.get(f.name, 0) for f in available_factors)
        if total_w > 0:
            freshness = sum(f.freshness * weights.get(f.name, 0) for f in available_factors) / total_w
        else:
            freshness = 0.5
    else:
        freshness = 0.0

    raw_confidence = agreement * 0.30 + strength * 0.30 + coverage * 0.25 + freshness * 0.15

    # Risk discount
    risk_discount = 1.0

    if not df.empty and len(df) >= 20:
        close = df["close"].dropna()
        vol_col = df["volume"].dropna()

        # High volatility discount
        if len(close) >= 20:
            returns = close.pct_change().dropna()
            vol_annual = float(returns.tail(20).std()) * (252 ** 0.5)
            if vol_annual > 0.60:
                risk_discount *= 0.70
            elif vol_annual > 0.40:
                risk_discount *= 0.85

        # Low volume discount
        if len(vol_col) >= 20:
            avg_vol = float(vol_col.tail(20).mean())
            if avg_vol < 200:
                risk_discount *= 0.60
            elif avg_vol < 500:
                risk_discount *= 0.80

        # Margin surge discount
        if "margin_balance" in df.columns:
            margin = df["margin_balance"].dropna()
            if len(margin) >= 6:
                m_now = float(margin.iloc[-1])
                m_5d = float(margin.iloc[-6]) if len(margin) >= 6 else m_now
                if m_5d > 0 and (m_now - m_5d) / m_5d > 0.10:
                    risk_discount *= 0.85

    # P/E discount (from fundamental_value factor)
    fv_factor = next((f for f in factors if f.name == "fundamental_value"), None)
    if fv_factor and fv_factor.available:
        pe = fv_factor.components.get("pe_ratio")
        if pe is not None:
            if pe > 80 or pe < 0:
                risk_discount *= 0.75
            elif pe > 50:
                risk_discount *= 0.85

    # Floor at 0.3
    risk_discount = max(0.3, risk_discount)

    confidence = raw_confidence * risk_discount

    return {
        "confidence": round(confidence, 4),
        "confidence_agreement": round(agreement, 4),
        "confidence_strength": round(strength, 4),
        "confidence_coverage": round(coverage, 4),
        "confidence_freshness": round(freshness, 4),
        "risk_discount": round(risk_discount, 4),
    }


# ═══════════════════════════════════════════════════════
# score_stock() — Main scoring function (20 因子)
# ═══════════════════════════════════════════════════════


def score_stock(
    stock_data: dict,
    df: pd.DataFrame,
    df_tech: pd.DataFrame,
    signals: dict,
    trust_info: dict,
    sentiment_scores: dict,
    sentiment_df: pd.DataFrame | None,
    ml_scores: dict,
    regime: str = "sideways",
    revenue_df: pd.DataFrame | None = None,
    global_data: dict | None = None,
    macro_data: dict | None = None,
    sector_data: dict | None = None,
) -> dict:
    """20 因子多維度評分

    Returns dict with all scores, confidence breakdown, regime, factor_details.
    Backward-compatible with old fields (technical_score, fundamental_score, etc.)
    """
    stock_id = stock_data["stock_id"]

    # Compute avg_vol_20d for institutional flow normalization
    avg_vol_20d = 10000.0
    if not df.empty and "volume" in df.columns:
        vol = df["volume"].dropna()
        if len(vol) >= 20:
            avg_vol_20d = float(vol.tail(20).mean())

    # Compute all 20 factors
    factors = [
        _compute_foreign_flow(trust_info, df, avg_vol_20d),
        _compute_technical_signal(signals, df_tech),
        _compute_short_momentum(df),
        _compute_trust_flow(trust_info, df, avg_vol_20d),
        _compute_volume_anomaly(df, df_tech),
        _compute_margin_sentiment(df),
        _compute_trend_momentum(df, df_tech),
        _compute_revenue_momentum(revenue_df),
        _compute_institutional_sync(trust_info, df),
        _compute_volatility_regime(df, df_tech),
        _compute_news_sentiment(sentiment_scores, sentiment_df, stock_id),
        _compute_global_context(global_data),
        _compute_margin_quality(stock_id),
        _compute_sector_rotation(stock_id, sector_data),
        _compute_ml_ensemble(ml_scores, stock_id),
        _compute_fundamental_value(stock_id),
        _compute_liquidity_quality(df),
        _compute_macro_risk(macro_data),
        _compute_export_momentum(global_data),
        _compute_us_manufacturing(macro_data),
    ]

    # Compute weights with regime adjustment and missing-data redistribution
    weights = _compute_weights(factors, regime)

    # Weighted total score (only available factors contribute)
    total_score = 0.0
    for f in factors:
        w = weights.get(f.name, 0)
        if f.available:
            total_score += f.score * w

    # Determine signal
    if total_score > 0.7:
        signal = "strong_buy"
    elif total_score > 0.6:
        signal = "buy"
    elif total_score < 0.3:
        signal = "strong_sell"
    elif total_score < 0.4:
        signal = "sell"
    else:
        signal = "hold"

    # Confidence
    conf = _compute_confidence(factors, weights, total_score, df)

    # Score coverage
    score_coverage = {f.name: f.available for f in factors}
    effective_coverage = round(sum(
        BASE_WEIGHTS.get(f.name, 0) for f in factors if f.available
    ), 2)

    # Factor details (full transparency)
    factor_details = {}
    for f in factors:
        factor_details[f.name] = {
            "score": f.score,
            "available": f.available,
            "freshness": f.freshness,
            "weight": round(weights.get(f.name, 0), 4),
            "components": f.components,
        }

    # Build reasoning
    reasons = _build_reasoning(factors, trust_info, signals, ml_scores, stock_id)

    # Helper to get factor score by name
    def _fs(name: str) -> float:
        f = next((f for f in factors if f.name == name), None)
        return round(f.score, 4) if f else 0.5

    # Backward compatible sub-scores (map new → old field names)
    return {
        **stock_data,
        "total_score": round(total_score, 4),
        "signal": signal,
        # Backward-compatible score fields
        "technical_score": _fs("technical_signal"),
        "fundamental_score": _fs("institutional_sync"),
        "sentiment_score": _fs("news_sentiment"),
        "ml_score": _fs("ml_ensemble"),
        "momentum_score": _fs("short_momentum"),
        # Mapped factor score fields (backward compat for DB columns)
        "institutional_flow_score": _fs("foreign_flow"),
        "margin_retail_score": _fs("margin_sentiment"),
        "volatility_score": _fs("volatility_regime"),
        "liquidity_score": _fs("liquidity_quality"),
        "value_quality_score": _fs("fundamental_value"),
        # Confidence
        **conf,
        # Coverage
        "score_coverage": score_coverage,
        "effective_coverage": effective_coverage,
        # Regime + details
        "market_regime": regime,
        "factor_details": factor_details,
        # Reasoning
        "reasoning": "；".join(reasons) if reasons else stock_data.get("reasoning", "資料不足"),
        # Internal factors for IC tracking
        "_factors": factors,
    }


def _build_reasoning(factors: list[FactorResult], trust_info: dict,
                     signals: dict, ml_scores: dict, stock_id: str) -> list[str]:
    """Build human-readable reasoning from factors"""
    reasons = []

    # Institutional
    trust_net = trust_info.get("trust_cumulative", 0)
    foreign_net = trust_info.get("foreign_cumulative", 0)
    trust_consec = trust_info.get("trust_consecutive_days", 0)
    foreign_consec = trust_info.get("foreign_consecutive_days", 0)

    if trust_net > 0:
        reasons.append(f"投信{trust_consec}日買超{trust_net:+,.0f}張")
    if foreign_net > 0:
        reasons.append(f"外資{foreign_consec}日買超{foreign_net:+,.0f}張")
    elif foreign_net < 0:
        reasons.append(f"外資賣超{foreign_net:,.0f}張")
    if trust_net > 0 and foreign_net > 0:
        reasons.append("外資+投信同步買超")

    # Technical
    tech_sig = signals.get("summary", {}).get("signal", "")
    if tech_sig:
        sig_labels = {"buy": "技術面偏多", "sell": "技術面偏空", "hold": "技術面中性"}
        reasons.append(sig_labels.get(tech_sig, f"技術 {tech_sig}"))

    # ML
    if stock_id in ml_scores:
        ml_val = ml_scores[stock_id]
        ml_label = "ML看多" if ml_val > 0.6 else ("ML看空" if ml_val < 0.4 else "ML中性")
        reasons.append(ml_label)

    # Margin warning
    margin_factor = next((f for f in factors if f.name == "margin_sentiment"), None)
    if margin_factor and margin_factor.available and margin_factor.score < 0.35:
        reasons.append("融資偏高警示")

    # Volatility warning
    vol_factor = next((f for f in factors if f.name == "volatility_regime"), None)
    if vol_factor and vol_factor.available and vol_factor.score < 0.3:
        reasons.append("高波動風險")

    # Margin quality
    mq_factor = next((f for f in factors if f.name == "margin_quality"), None)
    if mq_factor and mq_factor.available:
        if mq_factor.score > 0.7:
            reasons.append("毛利率擴張")
        elif mq_factor.score < 0.35:
            reasons.append("毛利率壓縮警示")

    # Sector rotation
    sr_factor = next((f for f in factors if f.name == "sector_rotation"), None)
    if sr_factor and sr_factor.available:
        sector = sr_factor.components.get("sector", "")
        if sr_factor.score > 0.65:
            reasons.append(f"{sector}產業資金流入")
        elif sr_factor.score < 0.35:
            reasons.append(f"{sector}產業資金流出")

    # Export momentum
    em_factor = next((f for f in factors if f.name == "export_momentum"), None)
    if em_factor and em_factor.available:
        if em_factor.score > 0.65:
            reasons.append("出口動能強勁")
        elif em_factor.score < 0.35:
            reasons.append("出口動能疲弱")

    # US manufacturing
    um_factor = next((f for f in factors if f.name == "us_manufacturing"), None)
    if um_factor and um_factor.available:
        if um_factor.score > 0.65:
            reasons.append("美製造業景氣擴張")
        elif um_factor.score < 0.35:
            reasons.append("美製造業景氣衰退")

    return reasons


# ═══════════════════════════════════════════════════════
# Helper: SSE event formatter + rank recommendations
# ═══════════════════════════════════════════════════════


def _event(step: str, status: str, progress: int, message: str = "", data: dict | None = None) -> str:
    """格式化 SSE 事件"""
    payload = {
        "step": step,
        "status": status,
        "progress": progress,
        "message": message,
    }
    if data is not None:
        payload["data"] = data
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def rank_recommendations(results: list[dict]) -> dict:
    """分類為 buy/sell/hold 推薦"""
    buy_recs = [r for r in results if r["signal"] in ("buy", "strong_buy")]
    sell_recs = [r for r in results if r["signal"] in ("sell", "strong_sell")]
    hold_recs = [r for r in results if r["signal"] == "hold"]

    buy_recs.sort(key=lambda x: x["total_score"], reverse=True)
    sell_recs.sort(key=lambda x: x["total_score"])
    hold_recs.sort(key=lambda x: x["total_score"], reverse=True)

    return {
        "buy_recommendations": buy_recs[:10],
        "sell_recommendations": sell_recs[:10],
        "hold": hold_recs,
    }


# ═══════════════════════════════════════════════════════
# run_market_scan() — Main scan engine
# ═══════════════════════════════════════════════════════


async def run_market_scan(top_n: int = 40) -> AsyncGenerator[str, None]:
    """全市場掃描 SSE 串流

    Steps:
    1. UNIVERSE    — TWSE T86 取得投信買超母體
    2. FETCH_DATA  — 批次抓取各股最新數據
    3. TECHNICAL   — 批次計算技術指標
    3.5 REGIME     — HMM 體制偵測
    4. SENTIMENT   — 全市場情緒
    5. ML_PREDICT  — 對有模型的股票跑 ML 預測
    5.5 GLOBAL     — 全球市場 / 宏觀 / 月營收數據
    5.7 SECTOR     — 產業聚合計算
    6. SCORE_RANK  — 20 因子評分 + 排序
    7. DONE        — 儲存 + IC 追蹤 + 回傳推薦清單
    """
    today = date.today()
    scanner = TWSEScanner()
    fetcher = StockFetcher()
    analyzer = TechnicalAnalyzer()

    # ── Step 1: UNIVERSE ──────────────────────────────────
    yield _event("universe", "running", 5, "掃描投信買超母體...")

    try:
        trust_data = await asyncio.to_thread(
            scanner.get_trust_top_stocks, days=5, top_n=top_n
        )

        universe_map = {}
        for s in trust_data:
            universe_map[s["stock_id"]] = s["stock_name"]
        for sid, sname in STOCK_LIST.items():
            if sid not in universe_map:
                universe_map[sid] = sname

        trust_lookup = {s["stock_id"]: s for s in trust_data}

        yield _event("universe", "done", 15,
                     f"投信買超 {len(trust_data)} 支 + 基本清單 = 共 {len(universe_map)} 支",
                     {"count": len(universe_map), "trust_count": len(trust_data)})
    except Exception as e:
        logger.error("TWSE scanner failed: %s", e)
        universe_map = dict(STOCK_LIST)
        trust_lookup = {}
        yield _event("universe", "error", 15,
                     f"TWSE 掃描失敗，使用基本清單 ({len(universe_map)} 支): {e}")

    stock_ids = list(universe_map.keys())

    # ── Step 2: FETCH_DATA ────────────────────────────────
    yield _event("fetch_data", "running", 20, f"批次抓取 {len(stock_ids)} 支股票資料...")

    stock_dfs = {}
    fetch_start = (today - timedelta(days=120)).isoformat()
    end_str = today.isoformat()

    async def _fetch_one(sid):
        try:
            df = await asyncio.to_thread(get_stock_prices, sid)
            if df.empty or len(df) < 20:
                new_df = await asyncio.to_thread(
                    fetcher.fetch_all, sid, fetch_start, end_str)
                if not new_df.empty:
                    await asyncio.to_thread(upsert_stock_prices, new_df, sid)
                    return sid, new_df
            elif not df.empty:
                latest = df["date"].max()
                if isinstance(latest, date) and (today - latest).days >= 1:
                    new_df = await asyncio.to_thread(
                        fetcher.fetch_all, sid, fetch_start, end_str)
                    if not new_df.empty:
                        await asyncio.to_thread(upsert_stock_prices, new_df, sid)
                        return sid, new_df
            return sid, df
        except Exception as e:
            logger.warning("Fetch failed for %s: %s", sid, e)
            df = await asyncio.to_thread(get_stock_prices, sid)
            return sid, df

    fetched = 0
    batch_size = 5
    for i in range(0, len(stock_ids), batch_size):
        batch = stock_ids[i:i + batch_size]
        tasks = [_fetch_one(sid) for sid in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                continue
            sid, df = result
            if not df.empty:
                stock_dfs[sid] = df
                fetched += 1

        progress = 20 + int((i + len(batch)) / len(stock_ids) * 20)
        yield _event("fetch_data", "running", min(progress, 40),
                     f"已抓取 {fetched}/{len(stock_ids)} 支...")

    yield _event("fetch_data", "done", 40,
                 f"資料抓取完成: {fetched}/{len(stock_ids)} 支有資料")

    # ── Step 3: TECHNICAL ─────────────────────────────────
    yield _event("technical", "running", 45, "批次計算技術指標...")

    tech_results = {}    # stock_id -> signals dict
    tech_dfs = {}        # stock_id -> df_tech DataFrame
    for sid, df in stock_dfs.items():
        try:
            df_tech = analyzer.compute_all(df)
            signals = analyzer.get_signals(df_tech)
            tech_results[sid] = signals
            tech_dfs[sid] = df_tech
        except Exception as e:
            logger.warning("Technical analysis failed for %s: %s", sid, e)

    yield _event("technical", "done", 52,
                 f"技術分析完成: {len(tech_results)} 支")

    # ── Step 3.5: REGIME DETECTION ────────────────────────
    yield _event("regime", "running", 54, "偵測市場體制...")

    regime = "sideways"
    try:
        from src.models.ensemble import HMMStateDetector
        # Use 2330 (TSMC) as TAIEX proxy
        taiex_proxy_df = stock_dfs.get("2330")
        if taiex_proxy_df is not None and len(taiex_proxy_df) >= 60:
            close = taiex_proxy_df["close"].dropna()
            returns = close.pct_change().dropna().values
            if len(returns) >= 60:
                hmm = HMMStateDetector(n_states=3)
                hmm.fit(returns)
                state = hmm.predict_state(returns)
                regime = state.state_name
                logger.info("HMM regime detected: %s", regime)
    except Exception as e:
        logger.warning("HMM regime detection failed, using sideways: %s", e)

    yield _event("regime", "done", 56, f"市場體制: {regime}")

    # ── Step 4: SENTIMENT ─────────────────────────────────
    yield _event("sentiment", "running", 58, "計算市場情緒...")

    from src.db.database import get_sentiment
    sentiment_scores = {}
    sentiment_dfs = {}  # stock_id -> DataFrame for per-stock sentiment details
    for sid in stock_dfs:
        try:
            sent_df = await asyncio.to_thread(
                get_sentiment, sid, today - timedelta(days=14), today)
            if not sent_df.empty:
                sentiment_dfs[sid] = sent_df
                scores = sent_df["sentiment_score"].dropna()
                if not scores.empty:
                    avg = float(scores.mean())
                    sentiment_scores[sid] = (avg + 1) / 2
        except Exception:
            pass

    yield _event("sentiment", "done", 62,
                 f"情緒資料: {len(sentiment_scores)} 支有情緒分數")

    # ── Step 5: ML_PREDICT ────────────────────────────────
    yield _event("ml_predict", "running", 65, "檢查 ML 模型...")

    ml_scores = {}
    for sid in stock_dfs:
        lstm_path = MODEL_DIR / f"{sid}_lstm.pt"
        xgb_path = MODEL_DIR / f"{sid}_xgb.json"
        if lstm_path.exists() and xgb_path.exists():
            try:
                from src.models.trainer import ModelTrainer
                trainer = ModelTrainer(sid)
                await asyncio.to_thread(trainer.load_models)
                pred_start = (today - timedelta(days=200)).isoformat()
                result = await asyncio.to_thread(
                    trainer.predict, start_date=pred_start, end_date=end_str)
                if result is not None:
                    sig_map = {"strong_buy": 0.9, "buy": 0.75,
                               "hold": 0.5, "sell": 0.25, "strong_sell": 0.1}
                    ml_scores[sid] = sig_map.get(result.signal, 0.5)
            except Exception as e:
                logger.warning("ML predict failed for %s: %s", sid, e)

    yield _event("ml_predict", "done", 75,
                 f"ML 預測: {len(ml_scores)} 支有模型")

    # ── Step 5.5: GLOBAL/MACRO/REVENUE ───────────────────
    yield _event("global_data", "running", 76, "取得全球市場/宏觀/營收數據...")

    global_data = None
    macro_data = None
    revenue_lookup: dict[str, pd.DataFrame] = {}
    try:
        global_data = await asyncio.to_thread(_fetch_global_market_data)
    except Exception as e:
        logger.warning("Global data fetch failed: %s", e)
    try:
        macro_data = await asyncio.to_thread(_fetch_macro_data)
    except Exception as e:
        logger.warning("Macro data fetch failed: %s", e)
    try:
        revenue_lookup = await _fetch_revenue_batch(stock_ids[:50])
    except Exception as e:
        logger.warning("Revenue batch fetch failed: %s", e)

    yield _event("global_data", "done", 78,
                 f"全球/宏觀數據完成, 月營收 {len(revenue_lookup)} 支")

    # ── Step 5.7: SECTOR AGGREGATION ──────────────────────
    sector_data = None
    try:
        sector_data = _compute_sector_aggregates(stock_dfs, trust_lookup)
    except Exception as e:
        logger.warning("Sector aggregation failed: %s", e)

    # ── Step 6: SCORE_RANK ────────────────────────────────
    yield _event("score_rank", "running", 78, "20 因子評分...")

    scored_results = []
    for sid in stock_dfs:
        df = stock_dfs[sid]
        if df.empty:
            continue

        latest = df.iloc[-1]
        current_price = float(latest["close"]) if pd.notna(latest.get("close")) else 0

        # Price change
        if len(df) >= 2:
            prev = df.iloc[-2]
            pct = ((current_price - float(prev["close"])) / float(prev["close"]) * 100
                   if prev["close"] and current_price else 0)
        else:
            pct = 0

        trust_info = trust_lookup.get(sid, {})

        stock_data = {
            "stock_id": sid,
            "stock_name": universe_map.get(sid, sid),
            "current_price": round(current_price, 2),
            "price_change_pct": round(pct, 2),
            "foreign_net_5d": trust_info.get("foreign_cumulative", 0),
            "trust_net_5d": trust_info.get("trust_cumulative", 0),
            "dealer_net_5d": trust_info.get("dealer_cumulative", 0),
        }

        scored = score_stock(
            stock_data=stock_data,
            df=df,
            df_tech=tech_dfs.get(sid, pd.DataFrame()),
            signals=tech_results.get(sid, {}),
            trust_info=trust_info,
            sentiment_scores=sentiment_scores,
            sentiment_df=sentiment_dfs.get(sid),
            ml_scores=ml_scores,
            regime=regime,
            revenue_df=revenue_lookup.get(sid),
            global_data=global_data,
            macro_data=macro_data,
            sector_data=sector_data,
        )
        scored_results.append(scored)

    # Sort by total_score descending and assign ranking
    scored_results.sort(key=lambda x: x["total_score"], reverse=True)
    for i, r in enumerate(scored_results):
        r["ranking"] = i + 1

    recommendations = rank_recommendations(scored_results)

    yield _event("score_rank", "done", 90,
                 f"評分完成: {len(scored_results)} 支 "
                 f"(BUY {len(recommendations['buy_recommendations'])}, "
                 f"SELL {len(recommendations['sell_recommendations'])})")

    # ── Step 7: DONE ──────────────────────────────────────
    yield _event("done", "running", 93, "儲存掃描結果...")

    # Save to DB
    scan_records = []
    for r in scored_results:
        rec = {
            "scan_date": today,
            "stock_id": r["stock_id"],
            "stock_name": r["stock_name"],
            "current_price": r["current_price"],
            "price_change_pct": r["price_change_pct"],
            "signal": r["signal"],
            "confidence": r["confidence"],
            "total_score": r["total_score"],
            "technical_score": r["technical_score"],
            "fundamental_score": r["fundamental_score"],
            "sentiment_score": r["sentiment_score"],
            "ml_score": r["ml_score"],
            "momentum_score": r["momentum_score"],
            "institutional_flow_score": r.get("institutional_flow_score"),
            "margin_retail_score": r.get("margin_retail_score"),
            "volatility_score": r.get("volatility_score"),
            "liquidity_score": r.get("liquidity_score"),
            "value_quality_score": r.get("value_quality_score"),
            "foreign_net_5d": r["foreign_net_5d"],
            "trust_net_5d": r["trust_net_5d"],
            "dealer_net_5d": r.get("dealer_net_5d", 0),
            "ranking": r["ranking"],
            "reasoning": r["reasoning"],
            "score_coverage": r.get("score_coverage"),
            "effective_coverage": r.get("effective_coverage"),
            "confidence_agreement": r.get("confidence_agreement"),
            "confidence_strength": r.get("confidence_strength"),
            "confidence_coverage": r.get("confidence_coverage"),
            "confidence_freshness": r.get("confidence_freshness"),
            "risk_discount": r.get("risk_discount"),
            "market_regime": r.get("market_regime"),
            "factor_details": r.get("factor_details"),
        }
        scan_records.append(rec)

    try:
        await asyncio.to_thread(save_market_scan, scan_records)
    except Exception as e:
        logger.error("Save market scan failed: %s", e)

    # IC tracking: save factor scores
    yield _event("done", "running", 95, "儲存因子 IC 追蹤...")
    try:
        ic_records = []
        for r in scored_results:
            factors = r.get("_factors", [])
            for f in factors:
                if f.available:
                    ic_records.append({
                        "record_date": today,
                        "stock_id": r["stock_id"],
                        "factor_name": f.name,
                        "factor_score": f.score,
                    })
        if ic_records:
            await asyncio.to_thread(save_factor_ic_records, ic_records)
    except Exception as e:
        logger.error("IC tracking save failed: %s", e)

    # Generate alerts
    alert_count = 0
    try:
        from api.services.alert_service import generate_alerts_from_scan
        new_alerts = await asyncio.to_thread(
            generate_alerts_from_scan, scored_results, today)
        alert_count = len(new_alerts)
    except Exception as e:
        logger.error("Alert generation failed: %s", e)

    # Clean internal fields before sending to client
    for r in scored_results:
        r.pop("_factors", None)
        r.pop("_has_sentiment", None)
        r.pop("_has_ml", None)

    yield _event("done", "done", 100, "掃描完成", {
        "scan_date": str(today),
        "total_stocks": len(scored_results),
        "alert_count": alert_count,
        "market_regime": regime,
        "stocks": scored_results,
        "buy_recommendations": recommendations["buy_recommendations"][:5],
        "sell_recommendations": recommendations["sell_recommendations"][:5],
    })


async def get_market_overview() -> dict:
    """讀取 DB 最新掃描結果"""
    results = await asyncio.to_thread(get_latest_market_scan)

    if not results:
        return {
            "scan_date": None,
            "stocks": [],
            "buy_recommendations": [],
            "sell_recommendations": [],
        }

    buy_recs = [r for r in results if r["signal"] in ("buy", "strong_buy")]
    sell_recs = [r for r in results if r["signal"] in ("sell", "strong_sell")]

    buy_recs.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    sell_recs.sort(key=lambda x: x.get("total_score", 0))

    return {
        "scan_date": results[0]["scan_date"] if results else None,
        "stocks": results,
        "buy_recommendations": buy_recs[:5],
        "sell_recommendations": sell_recs[:5],
    }
