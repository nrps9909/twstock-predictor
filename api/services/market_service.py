"""全市場掃描引擎 — SSE 串流 + 20 因子評分 + 體制權重 + 多維度信心"""

import asyncio
import calendar
import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from io import StringIO
from typing import AsyncGenerator

import numpy as np
import pandas as pd

from src.utils.constants import STOCK_LIST
from src.utils.config import settings
from src.data.twse_scanner import TWSEScanner
from src.data.stock_fetcher import StockFetcher
from src.db.database import (
    get_stock_prices,
    upsert_stock_prices,
    save_market_scan,
    get_latest_market_scan,
    save_factor_ic_records,
    get_all_factor_ic_summary,
    upsert_data_cache,
    get_data_cache,
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

    name: str  # e.g. "technical_signal"
    score: float  # 0.0-1.0 (0.5=中性)
    available: bool  # True=有真實資料
    freshness: float  # 0.0-1.0 (1.0=今天)
    components: dict = field(default_factory=dict)  # 子因子明細
    raw_value: float | None = None  # IC 追蹤用原始值


# ═══════════════════════════════════════════════════════
# Phase 3: Weight Engine — 基礎權重 + 體制乘數 (20 因子)
# ═══════════════════════════════════════════════════════

BASE_WEIGHTS = {
    # Short-term (38%) — decorrelated composite factors
    "composite_institutional": 0.15,  # foreign+trust+sync (decorrelated)
    "technical_signal": 0.08,  # technical signal aggregate
    "multi_scale_momentum": 0.10,  # short+trend momentum (decorrelated)
    "volume_anomaly": 0.05,  # volume anomaly
    # Mid-term (28%)
    "margin_sentiment": 0.04,  # margin contrarian
    "revenue_momentum": 0.05,  # monthly revenue YoY
    "volatility_regime": 0.04,  # volatility state
    "news_sentiment": 0.04,  # news sentiment
    "global_macro": 0.08,  # global+us_mfg+tw_etf (decorrelated)
    "margin_quality": 0.04,  # quarterly margin trend
    "sector_rotation": 0.03,  # sector rotation
    # Long-term (34%)
    "ml_ensemble": 0.15,  # ML model prediction (boosted)
    "fundamental_value": 0.06,  # PE/PB/ROE/yield
    "liquidity_quality": 0.04,  # liquidity quality
    "macro_risk": 0.05,  # macro risk environment
}  # sum = 1.00

# Legacy weight mapping for backward compatibility
_LEGACY_FACTOR_NAMES = {
    "foreign_flow": "composite_institutional",
    "trust_flow": "composite_institutional",
    "institutional_sync": "composite_institutional",
    "short_momentum": "multi_scale_momentum",
    "trend_momentum": "multi_scale_momentum",
    "global_context": "global_macro",
    "us_manufacturing": "global_macro",
    "taiwan_etf_momentum": "global_macro",
}

REGIME_MULTIPLIERS = {
    "bull": {
        "composite_institutional": 1.0,
        "technical_signal": 1.1,
        "multi_scale_momentum": 1.3,
        "volume_anomaly": 1.1,
        "margin_sentiment": 0.8,
        "revenue_momentum": 1.0,
        "volatility_regime": 0.7,
        "news_sentiment": 0.8,
        "global_macro": 1.0,
        "margin_quality": 0.8,
        "sector_rotation": 1.2,
        "ml_ensemble": 1.0,
        "fundamental_value": 0.8,
        "liquidity_quality": 0.8,
        "macro_risk": 0.8,
    },
    "bear": {
        "composite_institutional": 1.3,
        "technical_signal": 0.8,
        "multi_scale_momentum": 0.5,
        "volume_anomaly": 0.9,
        "margin_sentiment": 1.5,
        "revenue_momentum": 1.0,
        "volatility_regime": 1.5,
        "news_sentiment": 1.0,
        "global_macro": 1.2,
        "margin_quality": 1.2,
        "sector_rotation": 0.8,
        "ml_ensemble": 0.8,
        "fundamental_value": 1.2,
        "liquidity_quality": 1.3,
        "macro_risk": 1.3,
    },
    "sideways": {
        "composite_institutional": 1.0,
        "technical_signal": 1.3,
        "multi_scale_momentum": 0.8,
        "volume_anomaly": 1.2,
        "margin_sentiment": 1.0,
        "revenue_momentum": 1.0,
        "volatility_regime": 1.2,
        "news_sentiment": 1.2,
        "global_macro": 1.0,
        "margin_quality": 1.0,
        "sector_rotation": 1.3,
        "ml_ensemble": 1.0,
        "fundamental_value": 1.0,
        "liquidity_quality": 1.0,
        "macro_risk": 1.0,
    },
}

# ═══════════════════════════════════════════════════════
# 產業分類表 (sector_rotation 因子用)
# ═══════════════════════════════════════════════════════

STOCK_SECTOR = {
    # semiconductor
    "2330": "semiconductor",
    "2303": "semiconductor",
    "2454": "semiconductor",
    "3711": "semiconductor",
    "2379": "semiconductor",
    "3034": "semiconductor",
    "6770": "semiconductor",
    "2344": "semiconductor",
    "3529": "semiconductor",
    "5274": "semiconductor",
    "6505": "semiconductor",
    "3443": "semiconductor",
    "2449": "semiconductor",
    "3661": "semiconductor",
    "5347": "semiconductor",
    # electronics
    "2317": "electronics",
    "2382": "electronics",
    "2308": "electronics",
    "2301": "electronics",
    "3231": "electronics",
    "2395": "electronics",
    "2356": "electronics",
    "3044": "electronics",
    "2353": "electronics",
    "2327": "electronics",
    "6669": "electronics",
    "3706": "electronics",
    # finance
    "2881": "finance",
    "2882": "finance",
    "2886": "finance",
    "2884": "finance",
    "2891": "finance",
    "2892": "finance",
    "2880": "finance",
    "2883": "finance",
    "2885": "finance",
    "2887": "finance",
    "5880": "finance",
    "2890": "finance",
    # telecom
    "2412": "telecom",
    "3045": "telecom",
    "4904": "telecom",
    # traditional
    "1301": "traditional",
    "1303": "traditional",
    "1326": "traditional",
    "2002": "traditional",
    "1402": "traditional",
    "2105": "traditional",
    # shipping
    "2603": "shipping",
    "2609": "shipping",
    "2615": "shipping",
    "2618": "shipping",
    # biotech
    "6446": "biotech",
    "4743": "biotech",
    "1760": "biotech",
    # green_energy
    "6488": "green_energy",
    "3481": "green_energy",
}

DEFAULT_SECTOR = "other"

# Sector-specific weights for global_context factor: (SOX, TSM, ASML, EWT)
SECTOR_GLOBAL_WEIGHTS = {
    "semiconductor": (0.40, 0.25, 0.20, 0.15),
    "electronics": (0.25, 0.20, 0.10, 0.45),
    "finance": (0.10, 0.10, 0.05, 0.75),
    "telecom": (0.15, 0.10, 0.05, 0.70),
    "traditional": (0.10, 0.10, 0.05, 0.75),
    "shipping": (0.10, 0.10, 0.05, 0.75),
    "biotech": (0.15, 0.15, 0.05, 0.65),
    "green_energy": (0.15, 0.15, 0.10, 0.60),
}
DEFAULT_GLOBAL_WEIGHTS = (0.30, 0.20, 0.15, 0.35)

# Factor scaling constants (derived from historical return distributions)
SCALE_1D_RETURN = 10.0  # ±5% daily → 0/1
SCALE_3D_RETURN = 5.0  # ±10% → 0/1
SCALE_5D_RETURN = 4.0  # ±12.5% → 0/1
SCALE_BIAS = 8.0  # ±6.25% bias → 0/1
SCALE_GLOBAL_5D = 4.0  # ±12.5% global 5d return → 0/1
SCALE_RELATIVE = 10.0  # ±5% relative strength → 0/1
SCALE_GLOBAL_1D = 17.5  # legacy 1d return scale (kept for non-global uses)


_ic_weights_cache: dict = {"date": None, "weights": None}


def _ic_sigmoid(icir: float, ic_mean: float) -> float:
    """Continuous IC-driven weight multiplier using sigmoid mapping.

    Maps ICIR to a smooth multiplier in [0.4, 1.4]:
    - Strong negative IC (< -0.03): floor at 0.4x
    - Neutral ICIR (0): multiplier ≈ 1.0
    - Strong positive ICIR (>0.5): ceiling at 1.4x

    This replaces the old 3-level discrete mapping.
    """
    if ic_mean < -0.03:
        return max(0.4, 1.0 / (1.0 + np.exp(-icir * 4.0)))
    return 0.7 + 0.7 / (1.0 + np.exp(-icir * 4.0))


def _get_ic_adjusted_weights() -> dict[str, float] | None:
    """Fetch IC-adjusted base weights (cached daily).

    Applies continuous sigmoid IC feedback as a third-layer multiplier.
    Returns None if insufficient IC data (< 30 dates per factor).
    """
    from datetime import date as _date

    today = _date.today()
    if _ic_weights_cache["date"] == today and _ic_weights_cache["weights"] is not None:
        return _ic_weights_cache["weights"]

    try:
        ic_summary = get_all_factor_ic_summary(min_samples=30)
        if not ic_summary:
            return None

        # Map legacy factor names to new composite names
        mapped_summary: dict[str, list[dict]] = {}
        for factor, stats in ic_summary.items():
            target = _LEGACY_FACTOR_NAMES.get(factor, factor)
            if target not in BASE_WEIGHTS:
                continue
            mapped_summary.setdefault(target, []).append(stats)

        adjusted = dict(BASE_WEIGHTS)
        for factor, stats_list in mapped_summary.items():
            avg_icir = np.mean([s.get("icir", 0) for s in stats_list])
            avg_ic = np.mean([s.get("ic_mean", 0) for s in stats_list])
            adjusted[factor] *= _ic_sigmoid(avg_icir, avg_ic)

        total = sum(adjusted.values())
        if total > 0:
            result = {k: v / total for k, v in adjusted.items()}
        else:
            result = None

        _ic_weights_cache["date"] = today
        _ic_weights_cache["weights"] = result
        return result
    except Exception as e:
        logger.debug("IC weight adjustment failed: %s", e)
        return None


def _compute_weights(factors: list[FactorResult], regime: str) -> dict[str, float]:
    """計算各因子最終權重（BASE × REGIME × IC，含缺資料重分配）"""
    multipliers = REGIME_MULTIPLIERS.get(regime, REGIME_MULTIPLIERS["sideways"])
    available = [f for f in factors if f.available]
    if not available:
        return {f.name: 1.0 / len(factors) for f in factors}

    # IC-adjusted base weights (fallback to BASE_WEIGHTS if insufficient data)
    ic_base = _get_ic_adjusted_weights()
    base_weights = ic_base if ic_base is not None else BASE_WEIGHTS

    raw = {}
    for f in available:
        base = base_weights.get(f.name, 0.03)
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

# yfinance fallback ticker mapping — 主要 ticker 失敗時嘗試替代
_FALLBACK_TICKERS = {
    "^SOX": "SOXX",  # SOX ETF 替代
    "EWT": "0050.TW",  # 台灣50替代
    "^VIX": "VIXY",  # VIX ETF 替代
    "XLI": "IYJ",  # 工業 ETF 替代
    "TSM": "2330.TW",  # 台積電本地替代
    "ASML": "ASML.AS",  # ASML 歐洲替代
    "^TNX": "TLT",  # 美債 ETF (反向推算)
    "^FVX": "IEF",  # 中期美債 ETF
}


def _fetch_yfinance_with_fallback(ticker: str, period: str = "5d"):
    """yfinance 抓取 + fallback ticker 機制"""
    import yfinance as yf

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        if hist is not None and len(hist) >= 1:
            return hist
    except Exception as e:
        logger.warning("yfinance %s primary fetch failed: %s", ticker, e)

    # Try fallback
    fallback = _FALLBACK_TICKERS.get(ticker)
    if fallback:
        try:
            t = yf.Ticker(fallback)
            hist = t.history(period=period)
            if hist is not None and len(hist) >= 1:
                logger.info("yfinance %s → fallback %s succeeded", ticker, fallback)
                return hist
        except Exception as e:
            logger.warning(
                "yfinance %s fallback %s also failed: %s", ticker, fallback, e
            )

    return pd.DataFrame()


def _fetch_global_market_data() -> dict:
    """取得前一交易日全球市場數據 (yfinance), DB-first + 每日快取

    包含: SOX, TSM 日報酬 + EWT 1d/20d/60d 報酬 (出口動能用)
    """
    today = date.today()
    if _global_cache["date"] == today and _global_cache["data"] is not None:
        return _global_cache["data"]

    # DB-first: check persistent cache
    cached_json = get_data_cache("global_market", today)
    if cached_json:
        try:
            result = json.loads(cached_json)
            _global_cache["date"] = today
            _global_cache["data"] = result
            logger.info("Global market data loaded from DB cache")
            return result
        except (json.JSONDecodeError, TypeError):
            pass

    result = {}
    try:
        for ticker, key in [("^SOX", "sox"), ("TSM", "tsm")]:
            hist = _fetch_yfinance_with_fallback(ticker, "10d")
            if len(hist) >= 2:
                prev_close = float(hist["Close"].iloc[-2])
                last_close = float(hist["Close"].iloc[-1])
                result[f"{key}_return"] = (last_close - prev_close) / prev_close
                # 5d return for global_context factor
                if len(hist) >= 6:
                    close_5d_ago = float(hist["Close"].iloc[-6])
                    result[f"{key}_5d"] = (last_close - close_5d_ago) / close_5d_ago
                else:
                    result[f"{key}_5d"] = result[f"{key}_return"] * 3
            else:
                result[f"{key}_return"] = 0.0
                result[f"{key}_5d"] = 0.0
                logger.warning("yfinance %s: insufficient data (< 2 rows)", ticker)

        # EWT (iShares MSCI Taiwan ETF) — 出口動能代理指標
        hist = _fetch_yfinance_with_fallback("EWT", "90d")
        if len(hist) >= 2:
            last_close = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2])
            result["ewt_return_1d"] = (last_close - prev_close) / prev_close
        else:
            result["ewt_return_1d"] = 0.0
            last_close = 0.0
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

        # 0050.TW (元大台灣50) — 本地替代 / EWT fallback
        hist = _fetch_yfinance_with_fallback("0050.TW", "90d")
        if len(hist) >= 21:
            tw50_last = float(hist["Close"].iloc[-1])
            close_20d_ago = float(hist["Close"].iloc[-21])
            result["tw50_return_20d"] = (tw50_last - close_20d_ago) / close_20d_ago
        else:
            result["tw50_return_20d"] = 0.0
        if len(hist) >= 61:
            close_60d_ago = float(hist["Close"].iloc[-61])
            result["tw50_return_60d"] = (tw50_last - close_60d_ago) / close_60d_ago
        else:
            result["tw50_return_60d"] = 0.0

        # ASML (半導體設備領先指標)
        hist = _fetch_yfinance_with_fallback("ASML", "10d")
        if len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])
            last_close = float(hist["Close"].iloc[-1])
            result["asml_return"] = (last_close - prev_close) / prev_close
            if len(hist) >= 6:
                close_5d_ago = float(hist["Close"].iloc[-6])
                result["asml_5d"] = (last_close - close_5d_ago) / close_5d_ago
            else:
                result["asml_5d"] = result["asml_return"] * 3
        else:
            result["asml_return"] = 0.0
            result["asml_5d"] = 0.0

    except Exception as e:
        result["sox_return"] = 0.0
        result["sox_5d"] = 0.0
        result["tsm_return"] = 0.0
        result["tsm_5d"] = 0.0
        result["ewt_return_1d"] = 0.0
        result["ewt_return_20d"] = 0.0
        result["ewt_return_60d"] = 0.0
        result["tw50_return_20d"] = 0.0
        result["tw50_return_60d"] = 0.0
        result["asml_return"] = 0.0
        result["asml_5d"] = 0.0
        logger.warning("yfinance global import/init failed: %s", e)

    logger.info(
        "Global market data: SOX=%.4f TSM=%.4f EWT_20d=%.4f TW50_20d=%.4f ASML=%.4f",
        result.get("sox_return", 0),
        result.get("tsm_return", 0),
        result.get("ewt_return_20d", 0),
        result.get("tw50_return_20d", 0),
        result.get("asml_return", 0),
    )
    _global_cache["date"] = today
    _global_cache["data"] = result
    try:
        upsert_data_cache("global_market", today, json.dumps(result))
    except Exception as e:
        logger.warning("Failed to save global_market cache to DB: %s", e)
    return result


def _fetch_macro_data() -> dict:
    """取得宏觀風險數據 (yfinance), DB-first + 每日快取"""
    today = date.today()
    if _macro_cache["date"] == today and _macro_cache["data"] is not None:
        return _macro_cache["data"]

    # DB-first: check persistent cache
    cached_json = get_data_cache("macro_risk", today)
    if cached_json:
        try:
            result = json.loads(cached_json)
            _macro_cache["date"] = today
            _macro_cache["data"] = result
            logger.info("Macro risk data loaded from DB cache")
            return result
        except (json.JSONDecodeError, TypeError):
            pass

    result: dict = {}
    try:
        import yfinance as yf

        # VIX
        hist = _fetch_yfinance_with_fallback("^VIX", "5d")
        if len(hist) >= 1:
            result["vix"] = float(hist["Close"].iloc[-1])
        else:
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
        except Exception as e:
            result["usdtwd_trend"] = 0.0
            logger.warning("yfinance TWD=X fetch failed: %s", e)

        # 美國 10Y 殖利率 30 日變化
        tnx_hist_full = _fetch_yfinance_with_fallback("^TNX", "35d")
        if len(tnx_hist_full) >= 20:
            recent = float(tnx_hist_full["Close"].tail(5).mean())
            prior = float(tnx_hist_full["Close"].tail(30).head(10).mean())
            result["tnx_change"] = recent - prior
        else:
            result["tnx_change"] = 0.0

        # XLI (工業 ETF) — 製造業景氣代理指標
        hist = _fetch_yfinance_with_fallback("XLI", "250d")
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

        # ^FVX (5Y Treasury) — 殖利率曲線近似 (TNX - FVX)
        fvx_hist = _fetch_yfinance_with_fallback("^FVX", "35d")
        if len(fvx_hist) >= 5:
            result["fvx"] = float(fvx_hist["Close"].iloc[-1])
            # Yield curve spread: 10Y - 5Y (positive = normal, negative = inversion)
            if len(tnx_hist_full) >= 1:
                tnx_level = float(tnx_hist_full["Close"].iloc[-1])
                fvx_level = float(fvx_hist["Close"].iloc[-1])
                result["yield_curve_spread"] = tnx_level - fvx_level
                # 30d change in spread
                if len(fvx_hist) >= 25 and len(tnx_hist_full) >= 25:
                    fvx_30d = float(fvx_hist["Close"].tail(30).head(5).mean())
                    tnx_30d = float(tnx_hist_full["Close"].tail(30).head(5).mean())
                    spread_now = tnx_level - fvx_level
                    spread_30d = tnx_30d - fvx_30d
                    result["yield_curve_change"] = spread_now - spread_30d
        else:
            result["yield_curve_spread"] = 0.0

        # HG=F (Copper Futures) — 景氣領先指標
        try:
            copper = yf.Ticker("HG=F")
            hist = copper.history(period="30d")
            if len(hist) >= 21:
                last_close = float(hist["Close"].iloc[-1])
                close_20d_ago = float(hist["Close"].iloc[-21])
                result["copper_return_20d"] = (
                    last_close - close_20d_ago
                ) / close_20d_ago
            elif len(hist) >= 2:
                last_close = float(hist["Close"].iloc[-1])
                first_close = float(hist["Close"].iloc[0])
                result["copper_return_20d"] = (last_close - first_close) / first_close
            else:
                result["copper_return_20d"] = 0.0
        except Exception as e:
            result["copper_return_20d"] = 0.0
            logger.warning("yfinance HG=F (copper) fetch failed: %s", e)

        # XLI/SPY 比率趨勢 — 製造業相對強度
        try:
            spy_hist = _fetch_yfinance_with_fallback("SPY", "35d")
            xli_short = _fetch_yfinance_with_fallback("XLI", "35d")
            if len(spy_hist) >= 21 and len(xli_short) >= 21:
                xli_now = float(xli_short["Close"].iloc[-1])
                spy_now = float(spy_hist["Close"].iloc[-1])
                xli_20d = float(xli_short["Close"].iloc[-21])
                spy_20d = float(spy_hist["Close"].iloc[-21])
                ratio_now = xli_now / spy_now if spy_now > 0 else 1.0
                ratio_20d = xli_20d / spy_20d if spy_20d > 0 else 1.0
                result["xli_spy_ratio_trend"] = (
                    (ratio_now - ratio_20d) / ratio_20d if ratio_20d > 0 else 0.0
                )
            else:
                result["xli_spy_ratio_trend"] = 0.0
        except Exception as e:
            result["xli_spy_ratio_trend"] = 0.0
            logger.warning("yfinance XLI/SPY ratio fetch failed: %s", e)

    except Exception as e:
        result.setdefault("vix", 20.0)
        result.setdefault("usdtwd_trend", 0.0)
        result.setdefault("tnx_change", 0.0)
        result.setdefault("xli_return_20d", 0.0)
        result.setdefault("xli_vs_sma200", 0.0)
        result.setdefault("xli_spy_ratio_trend", 0.0)
        result.setdefault("yield_curve_spread", 0.0)
        result.setdefault("copper_return_20d", 0.0)
        logger.warning("yfinance macro import/init failed: %s", e)

    logger.info(
        "Macro data: VIX=%.1f USDTWD_trend=%.4f TNX_chg=%.4f XLI_20d=%.4f",
        result.get("vix", 20),
        result.get("usdtwd_trend", 0),
        result.get("tnx_change", 0),
        result.get("xli_return_20d", 0),
    )
    _macro_cache["date"] = today
    _macro_cache["data"] = result
    try:
        upsert_data_cache("macro_risk", today, json.dumps(result))
    except Exception as e:
        logger.warning("Failed to save macro_risk cache to DB: %s", e)
    return result


async def _fetch_revenue_batch(stock_ids: list[str]) -> dict[str, pd.DataFrame]:
    """批次取得最近15個月營收 (FinMind TaiwanStockMonthRevenue) — DB-first"""
    today = date.today()
    result: dict[str, pd.DataFrame] = {}
    ids_to_fetch: list[str] = []

    # DB-first: check cache for each stock
    for sid in stock_ids:
        cache_key = f"revenue:{sid}"
        cached_json = get_data_cache(cache_key, today)
        if cached_json:
            try:
                df = pd.read_json(StringIO(cached_json), orient="records")
                if not df.empty:
                    result[sid] = df
                    continue
            except Exception:
                pass
        ids_to_fetch.append(sid)

    if not ids_to_fetch:
        return result

    # Fetch missing from FinMind
    try:
        fetcher = StockFetcher()
        start = (today - timedelta(days=450)).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")
        for sid in ids_to_fetch:
            try:
                data = fetcher._query_finmind(
                    "TaiwanStockMonthRevenue", sid, start, end
                )
                if not data.empty:
                    result[sid] = data
                    # Save to DB cache
                    try:
                        upsert_data_cache(
                            f"revenue:{sid}",
                            today,
                            data.to_json(orient="records", date_format="iso"),
                        )
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    return result


# ═══════════════════════════════════════════════════════
# Phase 2: 20 Factor Computers
# ═══════════════════════════════════════════════════════


def _compute_foreign_flow(
    trust_info: dict,
    df: pd.DataFrame,
    avg_vol_20d: float,
    market_flow_stats: dict | None = None,
) -> FactorResult:
    """外資籌碼流向 (11%) — 標準化淨買超 + 連續天數 + 加速度 + 異常偵測"""
    foreign_net_5d = trust_info.get("foreign_cumulative", 0)
    foreign_consecutive = trust_info.get("foreign_consecutive_days", 0)

    has_data = bool(trust_info) or (
        not df.empty
        and "foreign_buy_sell" in df.columns
        and df["foreign_buy_sell"].notna().any()
    )
    if not has_data:
        return FactorResult("foreign_flow", 0.5, False, 0.0)

    components = {}

    # 1. 標準化淨買超 (40%) with cross-sectional adjustment
    if avg_vol_20d > 0:
        normalized = foreign_net_5d / avg_vol_20d
        # Cross-sectional: subtract market-average flow if available
        if market_flow_stats:
            mkt_mean = market_flow_stats.get("mean_normalized", 0)
            mkt_std = market_flow_stats.get("std_normalized", 0)
            if mkt_std > 0:
                normalized = (normalized - mkt_mean) / mkt_std
                net_score = max(0, min(1, 0.5 + normalized * 0.15))
            else:
                normalized = normalized - mkt_mean
                net_score = max(0, min(1, 0.5 + normalized * 0.2))
            components["cross_sectional_adj"] = True
        else:
            net_score = max(0, min(1, 0.5 + normalized * 0.2))
    else:
        net_score = 0.5
    components["net_normalized"] = round(net_score, 4)

    # 2. 連續天數 (20%): positive=buy streak, negative=sell streak
    consec_score = max(0.0, min(1.0, 0.5 + foreign_consecutive * 0.1))
    components["consecutive"] = round(consec_score, 4)

    # 3. 加速度 (20%): 近5日 vs 前5日
    accel_score = 0.5
    if not df.empty and len(df) >= 10 and "foreign_buy_sell" in df.columns:
        fbs = df["foreign_buy_sell"].fillna(0) / 1000  # convert to lots
        recent = float(fbs.tail(5).sum())
        prior = float(fbs.iloc[-10:-5].sum())
        threshold = max(10, avg_vol_20d * 0.05)
        if abs(prior) > threshold:
            accel = (recent - prior) / abs(prior)
            accel_score = max(0, min(1, 0.5 + accel * 0.25))
        else:
            accel_score = 0.6 if recent > 0 else 0.4
    components["acceleration"] = round(accel_score, 4)

    # 4. 異常大量偵測 (20%): Z-score
    anomaly_score = 0.5
    if not df.empty and len(df) >= 60 and "foreign_buy_sell" in df.columns:
        fbs_60 = (df["foreign_buy_sell"].tail(60) / 1000).dropna()  # lots
        if len(fbs_60) >= 20:
            mu, sigma = float(fbs_60.mean()), float(fbs_60.std())
            if sigma > 0:
                today_fbs = float(fbs_60.iloc[-1])
                z = (today_fbs - mu) / sigma
                anomaly_score = max(0, min(1, 0.5 + z * 0.15))
    components["anomaly_z"] = round(anomaly_score, 4)

    total = (
        net_score * 0.40
        + consec_score * 0.20
        + accel_score * 0.20
        + anomaly_score * 0.20
    )

    # Interaction bonus: when 3+ sub-factors directionally agree, boost conviction
    sub_scores = [net_score, consec_score, accel_score, anomaly_score]
    bullish = sum(1 for s in sub_scores if s > 0.6)
    bearish = sum(1 for s in sub_scores if s < 0.4)
    if bullish >= 3 or bearish >= 3:
        conviction = 1.15
        total = 0.5 + (total - 0.5) * conviction
        total = max(0, min(1, total))
        components["conviction_bonus"] = round(conviction, 2)

    trade_days = trust_info.get("trade_days", 0)
    freshness = min(1.0, trade_days / 5) if trade_days > 0 else 0.5
    return FactorResult(
        "foreign_flow", round(total, 4), True, freshness, components, raw_value=total
    )


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
    return FactorResult(
        "technical_signal", round(total, 4), True, 1.0, components, raw_value=total
    )


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
    return FactorResult(
        "short_momentum", round(total, 4), True, 1.0, components, raw_value=total
    )


def _compute_trust_flow(
    trust_info: dict, df: pd.DataFrame, avg_vol_20d: float
) -> FactorResult:
    """投信籌碼流向 (6%) — 標準化淨買超 + 連續天數 + 加速度"""
    trust_net_5d = trust_info.get("trust_cumulative", 0)
    trust_consecutive = trust_info.get("trust_consecutive_days", 0)

    has_data = bool(trust_info) or (
        not df.empty
        and "trust_buy_sell" in df.columns
        and df["trust_buy_sell"].notna().any()
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

    # 2. 連續天數 (30%): positive=buy streak, negative=sell streak
    consec_score = max(0.0, min(1.0, 0.5 + trust_consecutive * 0.1))
    components["consecutive"] = round(consec_score, 4)

    # 3. 加速度 (30%)
    accel_score = 0.5
    if not df.empty and len(df) >= 10 and "trust_buy_sell" in df.columns:
        tbs = df["trust_buy_sell"].fillna(0) / 1000  # convert to lots
        recent = float(tbs.tail(5).sum())
        prior = float(tbs.iloc[-10:-5].sum())
        threshold = max(10, avg_vol_20d * 0.03)
        if abs(prior) > threshold:
            accel = (recent - prior) / abs(prior)
            accel_score = max(0, min(1, 0.5 + accel * 0.25))
        else:
            accel_score = 0.6 if recent > 0 else 0.4
    components["acceleration"] = round(accel_score, 4)

    total = net_score * 0.40 + consec_score * 0.30 + accel_score * 0.30

    # Quarter-end window dressing decay (3/6/9/12 month, last ~10 trading days)
    today = date.today()
    if today.month in (3, 6, 9, 12) and today.day >= 20:
        days_left = calendar.monthrange(today.year, today.month)[1] - today.day
        decay = 0.5 + (days_left / 11) * 0.5  # decays to 0.5 on last day
        total = 0.5 + (total - 0.5) * decay
        components["quarter_end_decay"] = round(decay, 4)

    return FactorResult(
        "trust_flow", round(total, 4), True, 1.0, components, raw_value=total
    )


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
    ret_5d = (
        (float(close.iloc[-1]) / float(close.iloc[-6])) - 1 if len(close) >= 6 else 0
    )

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
    return FactorResult(
        "volume_anomaly", round(total, 4), True, 1.0, components, raw_value=total
    )


def _compute_margin_sentiment(df: pd.DataFrame) -> FactorResult:
    """融資融券情緒 (5%) — 反向指標 (沿用 margin_retail 邏輯)"""
    if df.empty or "margin_balance" not in df.columns:
        return FactorResult("margin_sentiment", 0.5, False, 0.0)

    margin = df["margin_balance"].dropna()
    short = (
        df["short_balance"].dropna()
        if "short_balance" in df.columns
        else pd.Series(dtype=float)
    )

    if len(margin) < 5:
        return FactorResult("margin_sentiment", 0.5, False, 0.0)

    components = {}

    # 1. Margin trend 50% (INVERSE)
    recent_margin = float(margin.iloc[-1])
    margin_5d_ago = (
        float(margin.iloc[-6]) if len(margin) >= 6 else float(margin.iloc[0])
    )
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
    return FactorResult(
        "margin_sentiment", round(total, 4), True, 0.9, components, raw_value=total
    )


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
    ret_60d = (
        (float(close.iloc[-1]) / float(close.iloc[-61])) - 1 if len(close) >= 61 else 0
    )

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
    return FactorResult(
        "trend_momentum", round(total, 4), True, 1.0, components, raw_value=total
    )


def _compute_revenue_momentum(revenue_df: pd.DataFrame | None) -> FactorResult:
    """月營收動能 (5%) — YoY + YoY加速度 + MoM

    Look-ahead bias fix: Taiwan companies must report monthly revenue by the 10th
    of the following month. Filter out revenue data that wouldn't be publicly
    available yet based on today's date.
    """
    if revenue_df is None or revenue_df.empty or len(revenue_df) < 13:
        return FactorResult("revenue_momentum", 0.5, False, 0.0)

    components = {}
    rev = revenue_df.sort_values("date").copy()

    # Look-ahead bias fix: filter to only publicly available revenue data
    # Revenue for month M is available after M+1 month's 10th day
    today = date.today()
    if "date" in rev.columns:
        rev["date"] = pd.to_datetime(rev["date"])
        # Revenue for a given month is reported by the 10th of the next month
        # E.g., January revenue available after Feb 10
        rev["report_avail_date"] = rev["date"].apply(
            lambda d: (d + pd.offsets.MonthEnd(1) + pd.DateOffset(days=10)).date()
        )
        rev = rev[rev["report_avail_date"] <= today]
        if len(rev) < 13:
            return FactorResult(
                "revenue_momentum",
                0.5,
                False,
                0.0,
                {"note": "insufficient_after_lookahead_filter"},
            )

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
    return FactorResult(
        "revenue_momentum", round(total, 4), True, 0.8, components, raw_value=total
    )


def _compute_institutional_sync(trust_info: dict, df: pd.DataFrame) -> FactorResult:
    """法人同步性 (5%) — 外資投信同步 + 三法人方向 + 法人vs散戶 + 自營商訊號"""
    foreign_cum = trust_info.get("foreign_cumulative", 0)
    trust_cum = trust_info.get("trust_cumulative", 0)
    dealer_cum = trust_info.get("dealer_cumulative", 0)

    if not trust_info:
        return FactorResult("institutional_sync", 0.5, False, 0.0)

    components = {}

    # Avg volume for dealer significance check
    avg_vol = 10000.0
    if not df.empty and "volume" in df.columns:
        vol = df["volume"].dropna()
        if len(vol) >= 20:
            avg_vol = float(vol.tail(20).mean())

    # 1. 外資投信同步 (35%)
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

    # 2. 三法人合計方向 (25%)
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

    # 3. 法人 vs 散戶背離 (20%)
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

    # 4. 自營商訊號 (20%) — large dealer trades as hedging warning
    dealer_score = 0.5
    if abs(dealer_cum) > avg_vol * 0.05:
        # Significant dealer activity
        foreign_sign = 1 if foreign_cum > 0 else (-1 if foreign_cum < 0 else 0)
        dealer_sign = 1 if dealer_cum > 0 else (-1 if dealer_cum < 0 else 0)
        if dealer_sign == foreign_sign:
            dealer_score = 0.7  # all institutions aligned
        else:
            dealer_score = 0.3  # dealer hedging against foreign = warning
    components["dealer_signal"] = round(dealer_score, 4)

    total = (
        sync_score * 0.35
        + direction_score * 0.25
        + diverge_score * 0.20
        + dealer_score * 0.20
    )
    return FactorResult(
        "institutional_sync", round(total, 4), True, 1.0, components, raw_value=total
    )


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
    vol_20d = float(returns.tail(20).std()) * (252**0.5)
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
        vol_5d = (
            float(returns.tail(5).std()) * (252**0.5) if len(returns) >= 5 else vol_20d
        )
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
    return FactorResult(
        "volatility_regime", round(total, 4), True, 1.0, components, raw_value=total
    )


def _compute_news_sentiment(
    sentiment_scores: dict, sentiment_df: pd.DataFrame | None, stock_id: str
) -> FactorResult:
    """新聞情緒 (4%) — 沿用 sentiment 邏輯"""
    if stock_id not in sentiment_scores:
        return FactorResult("news_sentiment", 0.5, False, 0.0)

    components = {}
    weight_total = 0.0

    # 1. Source-weighted score 40%
    if sentiment_df is not None and not sentiment_df.empty:
        # Source credibility weights — 基於來源品質的動態權重
        source_weights = {
            "cnyes": 0.40,  # 鉅亨網：專業財經媒體，可信度最高
            "yahoo": 0.25,
            "yahoo_tw": 0.25,  # Yahoo 股市：綜合來源
            "google": 0.20,
            "google_news": 0.20,  # Google News：聚合來源
            "ptt": 0.15,  # PTT：社群輿情，噪音較高但有獨特訊號
        }
        weighted_sum = 0.0
        weight_total = 0.0
        seen_groups = set()  # Avoid double-counting yahoo/yahoo_tw
        if "source" in sentiment_df.columns:
            for source in sentiment_df["source"].unique():
                w = source_weights.get(source, 0.10)
                group = source.split("_")[0]  # yahoo_tw -> yahoo
                if group in seen_groups:
                    continue
                seen_groups.add(group)
                src_df = sentiment_df[sentiment_df["source"] == source]
                src_scores = src_df["sentiment_score"].dropna()
                if not src_scores.empty:
                    avg = float(src_scores.mean())
                    # Normalize: if scores are in -1~1 range, map to 0~1
                    if avg < 0 or (avg == 0 and src_scores.min() < 0):
                        avg = (avg + 1) / 2
                    weighted_sum += avg * w
                    weight_total += w
        source_score = (
            weighted_sum / weight_total
            if weight_total > 0
            else sentiment_scores[stock_id]
        )
    else:
        source_score = sentiment_scores[stock_id]
    components["source_weighted"] = round(source_score, 4)

    # 2. Sentiment momentum 30%
    momentum_score = 0.5
    if sentiment_df is not None and not sentiment_df.empty:
        scores = sentiment_df["sentiment_score"].dropna()
        if len(scores) >= 6:
            half = len(scores) // 2
            recent = float(scores.tail(half).mean())
            early = float(scores.head(half).mean())
            delta = recent - early
            momentum_score = max(0.0, min(1.0, 0.5 + delta * 1.5))
        elif len(scores) >= 3:
            # Few articles: use overall sentiment direction
            avg = float(scores.mean())
            if avg < 0:
                avg = (avg + 1) / 2
            momentum_score = max(0.0, min(1.0, avg))
    components["momentum"] = round(momentum_score, 4)

    # 3. Engagement anomaly 30%
    engage_score = 0.5
    if (
        sentiment_df is not None
        and not sentiment_df.empty
        and "engagement" in sentiment_df.columns
    ):
        engagement = sentiment_df["engagement"].dropna()
        if len(engagement) >= 3:
            total_engage = float(engagement.sum())
            if total_engage > 0:
                # High engagement amplifies sentiment direction
                avg_per_article = total_engage / len(engagement)
                if avg_per_article > 20:
                    engage_score = 0.7 if source_score > 0.5 else 0.3
                elif avg_per_article > 5:
                    engage_score = 0.6 if source_score > 0.5 else 0.4
            elif len(engagement) >= 5:
                recent_engage = float(engagement.tail(3).mean())
                avg_engage = float(engagement.mean())
                if avg_engage > 0:
                    ratio = recent_engage / avg_engage
                    if ratio > 2.0:
                        engage_score = 0.7 if source_score > 0.5 else 0.3
                    elif ratio > 1.3:
                        engage_score = 0.6 if source_score > 0.5 else 0.4
    components["engagement"] = round(engage_score, 4)

    freshness = 0.5
    if (
        sentiment_df is not None
        and not sentiment_df.empty
        and "date" in sentiment_df.columns
    ):
        try:
            latest_date = pd.to_datetime(sentiment_df["date"]).max()
            days_old = (pd.Timestamp.now() - latest_date).days
            freshness = max(0.0, min(1.0, 1.0 - days_old / 14))
        except Exception:
            freshness = 0.5

    total = source_score * 0.40 + momentum_score * 0.30 + engage_score * 0.30
    # available reflects actual data quality, not just presence of stock_id in scores
    has_real_data = weight_total > 0 or (
        sentiment_df is not None and not sentiment_df.empty and len(sentiment_df) >= 3
    )
    return FactorResult(
        "news_sentiment",
        round(total, 4),
        has_real_data,
        round(freshness, 2),
        components,
        raw_value=total,
    )


def _compute_global_context(global_data: dict | None, sector: str = "") -> FactorResult:
    """國際市場連動 (3%) — SOX + TSM + ASML 5d returns + EWT relative, sector-weighted

    Uses 5-day returns instead of 1-day to reduce noise.
    Fallback: 1d return * 3 as rough 5d proxy when 5d data unavailable.
    """
    if global_data is None:
        return FactorResult("global_context", 0.5, False, 0.0)

    w_sox, w_tsm, w_asml, w_ewt = SECTOR_GLOBAL_WEIGHTS.get(
        sector, DEFAULT_GLOBAL_WEIGHTS
    )

    components = {}

    # Prefer 5d returns, fallback to 1d * 3 as rough proxy
    sox_1d = global_data.get("sox_return", 0)
    sox_ret = global_data.get("sox_5d", sox_1d * 3)
    tsm_ret = global_data.get("tsm_5d", global_data.get("tsm_return", 0) * 3)
    asml_ret = global_data.get("asml_5d", global_data.get("asml_return", 0) * 3)
    ewt_1d = global_data.get("ewt_return_1d", 0)

    def ret_to_score(r):
        return max(0.05, min(0.95, 0.5 + r * SCALE_GLOBAL_5D))

    sox_score = ret_to_score(sox_ret)
    tsm_score = ret_to_score(tsm_ret)
    asml_score = ret_to_score(asml_ret)

    # EWT vs SOX 相對強弱 (1d — higher frequency for relative comparison)
    ewt_relative = ewt_1d - sox_1d
    ewt_rel_score = max(0.05, min(0.95, 0.5 + ewt_relative * SCALE_RELATIVE))

    components["sox_return_5d"] = round(sox_ret, 4)
    components["tsm_return_5d"] = round(tsm_ret, 4)
    components["asml_return_5d"] = round(asml_ret, 4)
    # Backward compat keys
    components["sox_return"] = round(sox_1d, 4)
    components["tsm_return"] = round(global_data.get("tsm_return", 0), 4)
    components["asml_return"] = round(global_data.get("asml_return", 0), 4)
    components["sox_score"] = round(sox_score, 4)
    components["tsm_score"] = round(tsm_score, 4)
    components["asml_score"] = round(asml_score, 4)
    components["ewt_relative_score"] = round(ewt_rel_score, 4)
    components["sector"] = sector or "default"
    components["weights"] = {"sox": w_sox, "tsm": w_tsm, "asml": w_asml, "ewt": w_ewt}

    total = (
        sox_score * w_sox
        + tsm_score * w_tsm
        + asml_score * w_asml
        + ewt_rel_score * w_ewt
    )

    available = sox_ret != 0 or tsm_ret != 0 or asml_ret != 0
    return FactorResult(
        "global_context", round(total, 4), available, 0.9, components, raw_value=total
    )


def _compute_ml_ensemble(ml_scores: dict, stock_id: str) -> FactorResult:
    """ML 集成 (8%) — signal 映射"""
    if stock_id not in ml_scores:
        return FactorResult("ml_ensemble", 0.5, False, 0.0)

    score = ml_scores[stock_id]
    return FactorResult(
        "ml_ensemble",
        round(score, 4),
        True,
        1.0,
        {"raw_ml_score": round(score, 4)},
        raw_value=score,
    )


def _compute_fundamental_value(
    stock_id: str,
    fundamental_data: dict | None = None,
    per_pbr_df: pd.DataFrame | None = None,
) -> FactorResult:
    """基本面價值 (6%) — P/E + P/B + ROE + 殖利率

    Data priority:
        1. FinMind TaiwanStockPER (per_pbr_df) — 每日 P/E, P/B, 殖利率
        2. yfinance ticker.info fallback — trailingPE, returnOnEquity, dividendYield
    """
    components = {}
    pe = None
    pb = None
    roe = None
    div_yield = None
    data_source = "none"

    # ── Primary: FinMind 每日 P/E, P/B, 殖利率 ──
    if per_pbr_df is not None and not per_pbr_df.empty:
        latest = per_pbr_df.iloc[-1]
        if "PER" in per_pbr_df.columns:
            val = latest["PER"]
            if pd.notna(val) and val != 0:
                pe = float(val)
        if "PBR" in per_pbr_df.columns:
            val = latest["PBR"]
            if pd.notna(val) and val != 0:
                pb = float(val)
        if "dividend_yield" in per_pbr_df.columns:
            val = latest["dividend_yield"]
            if pd.notna(val):
                div_yield = float(val) / 100.0  # FinMind 殖利率為百分比
        if pe is not None or pb is not None:
            data_source = "finmind"
            components["data_source"] = "finmind"

    # ── Fallback: yfinance (DB-first) ──
    info = {}
    if fundamental_data and "info" in fundamental_data:
        info = fundamental_data["info"]
    elif data_source == "none":
        # Try DB cache before yfinance
        _cache_key = f"fundamental:{stock_id}"
        try:
            _cached = get_data_cache(_cache_key, date.today())
        except Exception:
            _cached = None
        if _cached:
            try:
                info = json.loads(_cached)
            except Exception:
                info = {}
        if not info:
            try:
                import yfinance as yf

                ticker = yf.Ticker(f"{stock_id}.TW")
                info = ticker.info or {}
                # Save to DB cache
                if info:
                    try:
                        safe_info = {
                            k: v
                            for k, v in info.items()
                            if isinstance(v, (str, int, float, bool, type(None)))
                        }
                        upsert_data_cache(
                            _cache_key, date.today(), json.dumps(safe_info)
                        )
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(
                    "yfinance %s.TW info fetch failed (fundamental_value): %s",
                    stock_id,
                    e,
                )

    if pe is None and info.get("trailingPE") is not None:
        pe = info["trailingPE"]
        data_source = data_source if data_source != "none" else "yfinance"
    if div_yield is None and info.get("dividendYield") is not None:
        div_yield = info["dividendYield"]
    roe = info.get("returnOnEquity")
    if "data_source" not in components and data_source != "none":
        components["data_source"] = data_source

    # 1. P/E 風險過濾 (30%)
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

    # 2. P/B mean-reversion (20%)
    if pb is not None:
        components["pb_ratio"] = round(pb, 2)
        if pb < 1.0:
            pb_score = 0.75
        elif pb < 1.5:
            pb_score = 0.60
        elif pb < 2.5:
            pb_score = 0.50
        elif pb < 3.0:
            pb_score = 0.40
        else:
            pb_score = 0.25
    else:
        pb_score = 0.50

    # 3. ROE 穩健度 (25%)
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

    # 4. 殖利率 (25%)
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

    total = pe_score * 0.30 + pb_score * 0.20 + roe_score * 0.25 + div_score * 0.25

    available = bool(
        pe is not None or pb is not None or roe is not None or div_yield is not None
    )
    freshness = 0.9 if data_source == "finmind" else (0.7 if available else 0.0)
    return FactorResult(
        "fundamental_value",
        round(total, 4),
        available,
        freshness,
        components,
        raw_value=total,
    )


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
    return FactorResult(
        "liquidity_quality", round(total, 4), True, 1.0, components, raw_value=total
    )


def _compute_macro_risk(macro_data: dict | None) -> FactorResult:
    """宏觀風險環境 (4%) — VIX + 殖利率曲線 + USD/TWD + 美10Y + 銅價"""
    if macro_data is None:
        return FactorResult("macro_risk", 0.5, False, 0.0)

    components = {}

    # 1. VIX 恐慌指標 (30%)
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

    # 2. 殖利率曲線 (20%) — 10Y-5Y spread 變化 (inversion = bearish)
    yield_spread = macro_data.get("yield_curve_spread", 0)
    yield_change = macro_data.get("yield_curve_change", 0)
    if yield_spread < -0.3:
        yc_score = 0.20  # Deep inversion
    elif yield_spread < 0:
        yc_score = 0.35  # Mild inversion
    elif yield_spread < 0.5:
        yc_score = 0.55  # Normal flat
    else:
        yc_score = 0.70  # Normal steep
    # Adjust for direction of change
    yc_score = max(0.1, min(0.9, yc_score + yield_change * 2.0))
    components["yield_curve_spread"] = round(yield_spread, 4)
    components["yield_curve_score"] = round(yc_score, 4)

    # 3. USD/TWD 趨勢 (20%)
    fx_trend = macro_data.get("usdtwd_trend", 0)
    fx_score = max(0.1, min(0.9, 0.5 - fx_trend * 15.0))
    components["usdtwd_trend"] = round(fx_trend, 4)
    components["fx_score"] = round(fx_score, 4)

    # 4. 美國10Y殖利率變化 (15%)
    tnx_chg = macro_data.get("tnx_change", 0)
    tnx_score = max(0.1, min(0.9, 0.5 - tnx_chg * 1.5))
    components["tnx_change"] = round(tnx_chg, 4)
    components["tnx_score"] = round(tnx_score, 4)

    # 5. 銅價動能 (15%) — 景氣領先指標 (上漲 = risk-on)
    copper_ret = macro_data.get("copper_return_20d", 0)
    copper_score = max(0.1, min(0.9, 0.5 + copper_ret * 4.0))
    components["copper_return_20d"] = round(copper_ret, 4)
    components["copper_score"] = round(copper_score, 4)

    total = (
        vix_score * 0.30
        + yc_score * 0.20
        + fx_score * 0.20
        + tnx_score * 0.15
        + copper_score * 0.15
    )

    available = vix != 20.0 or fx_trend != 0 or tnx_chg != 0
    return FactorResult(
        "macro_risk", round(total, 4), available, 0.9, components, raw_value=total
    )


def _compute_margin_quality(
    stock_id: str, fundamental_data: dict | None = None
) -> FactorResult:
    """季報毛利率/營益率趨勢 (4%) — yfinance 季報數據

    Args:
        stock_id: 股票代號
        fundamental_data: 預取的 yfinance 數據 {"info": {...}, "quarterly_income_stmt": DataFrame}
                          若為 None 則即時抓取 (backward compat)
    """
    try:
        # Resolve data source
        if fundamental_data and "quarterly_income_stmt" in fundamental_data:
            qis = fundamental_data["quarterly_income_stmt"]
            info = fundamental_data.get("info", {})
        else:
            # margin_quality needs qis (quarterly_income_stmt) — must fetch from yfinance
            # DB cache only stores 'info' dict, not DataFrames
            qis = None
            info = {}
            try:
                import yfinance as yf

                ticker = yf.Ticker(f"{stock_id}.TW")
                try:
                    qis = ticker.quarterly_income_stmt
                except Exception:
                    qis = None
                # Try DB cache for info before yfinance
                _cache_key = f"fundamental:{stock_id}"
                try:
                    _cached = get_data_cache(_cache_key, date.today())
                except Exception:
                    _cached = None
                if _cached:
                    try:
                        info = json.loads(_cached)
                    except Exception:
                        info = {}
                if not info:
                    try:
                        info = ticker.info or {}
                        if info:
                            try:
                                safe_info = {
                                    k: v
                                    for k, v in info.items()
                                    if isinstance(
                                        v, (str, int, float, bool, type(None))
                                    )
                                }
                                upsert_data_cache(
                                    _cache_key, date.today(), json.dumps(safe_info)
                                )
                            except Exception:
                                pass
                    except Exception as e:
                        logger.warning(
                            "yfinance %s.TW info fetch failed (margin_quality): %s",
                            stock_id,
                            e,
                        )
                        info = {}
            except ImportError:
                pass

        # Try quarterly income statement first
        if qis is not None and not qis.empty and qis.shape[1] >= 2:
            components = {}

            # Look-ahead bias fix: filter out quarterly data not yet publicly filed.
            # Taiwan companies must file quarterly reports within 45 days of quarter end.
            today = date.today()
            if hasattr(qis.columns, "to_pydatetime"):
                avail_cols = []
                for col in qis.columns:
                    try:
                        q_end = pd.Timestamp(col).date()
                        filing_deadline = q_end + timedelta(days=45)
                        if filing_deadline <= today:
                            avail_cols.append(col)
                    except Exception:
                        avail_cols.append(col)
                if avail_cols:
                    qis = qis[avail_cols]
                if qis.shape[1] < 2:
                    components["note"] = "insufficient_after_filing_date_filter"
                    # Fall through to yfinance fallback below

            # Parse gross margin from quarterly data
            gross_profit = None
            total_revenue = None
            operating_income = None

            if qis.shape[1] >= 2:
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
                gm_latest = (
                    float(gross_profit.iloc[0]) / float(total_revenue.iloc[0])
                    if float(total_revenue.iloc[0]) != 0
                    else 0
                )
                gm_prev = (
                    float(gross_profit.iloc[1]) / float(total_revenue.iloc[1])
                    if float(total_revenue.iloc[1]) != 0
                    else 0
                )

                gm_qoq_change = gm_latest - gm_prev

                # YoY if 5+ quarters
                gm_yoy_change = 0.0
                if qis.shape[1] >= 5:
                    gm_yoy = (
                        float(gross_profit.iloc[4]) / float(total_revenue.iloc[4])
                        if float(total_revenue.iloc[4]) != 0
                        else 0
                    )
                    gm_yoy_change = gm_latest - gm_yoy

                # Scoring: 毛利率趨勢 (60%)
                trend_score = max(
                    0.0, min(1.0, 0.5 + gm_qoq_change * 8.0 + gm_yoy_change * 4.0)
                )
                components["gm_latest"] = round(gm_latest, 4)
                components["gm_qoq_change"] = round(gm_qoq_change, 4)
                components["gm_yoy_change"] = round(gm_yoy_change, 4)
                components["trend_score"] = round(trend_score, 4)

                # 營益率水準 (40%)
                op_margin = 0.0
                if operating_income is not None and float(total_revenue.iloc[0]) != 0:
                    op_margin = float(operating_income.iloc[0]) / float(
                        total_revenue.iloc[0]
                    )
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
                return FactorResult(
                    "margin_quality",
                    round(total, 4),
                    True,
                    0.6,
                    components,
                    raw_value=total,
                )

        # Fallback: ticker.info margins
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
            return FactorResult(
                "margin_quality",
                round(total, 4),
                True,
                0.4,
                components,
                raw_value=total,
            )

    except Exception as e:
        logger.warning("margin_quality computation failed for %s: %s", stock_id, e)

    return FactorResult("margin_quality", 0.5, False, 0.0)


def _compute_sector_aggregates(
    stock_dfs: dict[str, pd.DataFrame], trust_lookup: dict
) -> dict[str, dict]:
    """預計算各產業聚合數據 (sector_rotation 因子用)

    Returns: {sector_name: {net_flow, avg_return_20d, breadth, index_return}}
    """
    from collections import defaultdict

    sector_stocks = defaultdict(list)

    for sid in stock_dfs:
        sector = STOCK_SECTOR.get(sid, DEFAULT_SECTOR)
        sector_stocks[sector].append(sid)

    # Fetch TWSE industry indices — DB-first (daily cache)
    industry_indices: dict[str, float] = {}
    _today = date.today()
    try:
        _cached = get_data_cache("industry_indices", _today)
    except Exception:
        _cached = None
    if _cached:
        try:
            industry_indices = json.loads(_cached)
        except Exception:
            industry_indices = {}
    if not industry_indices:
        try:
            scanner = TWSEScanner()
            industry_indices = scanner.fetch_industry_indices()
            if industry_indices:
                try:
                    upsert_data_cache(
                        "industry_indices", _today, json.dumps(industry_indices)
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.warning(
                "TWSE industry indices fetch failed in sector_aggregates: %s", e
            )

    sector_data = {}
    all_flows = []
    all_returns = []
    per_stock_normalized_flows: list[float] = []  # for cross-sectional stats

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

            # Per-stock volume-normalized foreign flow for cross-sectional stats
            if not df.empty and "volume" in df.columns:
                vol = df["volume"].dropna()
                avg_vol = float(vol.tail(20).mean()) if len(vol) >= 20 else 0
                if avg_vol > 0:
                    per_stock_normalized_flows.append(foreign_cum / avg_vol)

        avg_return = float(np.mean(returns_20d)) if returns_20d else 0.0
        breadth = positive_flow_count / total_count if total_count > 0 else 0.5

        sector_data[sector] = {
            "net_flow": net_flow,
            "avg_return_20d": avg_return,
            "breadth": breadth,
            "stock_count": total_count,
            "index_return": industry_indices.get(sector),  # TWSE 產業指數報酬
        }
        all_flows.append(net_flow)
        all_returns.append(avg_return)

    # Market average
    market_avg_flow = float(np.mean(all_flows)) if all_flows else 0.0
    market_avg_return = float(np.mean(all_returns)) if all_returns else 0.0
    market_avg_index = (
        float(np.mean([v for v in industry_indices.values() if v is not None]))
        if industry_indices
        else 0.0
    )

    # Cross-sectional flow stats for foreign_flow factor normalization
    flow_mean = (
        float(np.mean(per_stock_normalized_flows))
        if per_stock_normalized_flows
        else 0.0
    )
    flow_std = (
        float(np.std(per_stock_normalized_flows))
        if len(per_stock_normalized_flows) >= 5
        else 0.0
    )

    sector_data["_market_avg"] = {
        "net_flow": market_avg_flow,
        "avg_return_20d": market_avg_return,
        "index_return": market_avg_index,
        "flow_mean_normalized": flow_mean,
        "flow_std_normalized": flow_std,
    }

    return sector_data


def _compute_sector_rotation(stock_id: str, sector_data: dict | None) -> FactorResult:
    """產業資金輪動 (3%) — 法人流向 + TWSE產業指數動能 + 相對動能 + 廣度"""
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

    # 1. 產業法人流向 vs 市場平均 (35%)
    flow_diff = s_data["net_flow"] - mkt_flow
    flow_denom = max(abs(mkt_flow), 1000.0)
    flow_score = max(0.0, min(1.0, 0.5 + (flow_diff / flow_denom) * 0.3))
    components["flow_vs_market"] = round(flow_score, 4)
    components["sector"] = sector

    # 2. TWSE 產業指數動能 (30%) — 新增
    index_return = s_data.get("index_return")
    mkt_index_return = market_avg.get("index_return", 0)
    has_index = index_return is not None
    if has_index:
        # 產業指數 vs 市場平均指數的相對強弱
        idx_diff = index_return - mkt_index_return
        index_score = max(0.0, min(1.0, 0.5 + idx_diff * 12.0))
        components["index_return"] = round(index_return, 4)
        components["index_momentum"] = round(index_score, 4)
    else:
        index_score = 0.5
        components["index_momentum"] = 0.5

    # 3. 產業相對動能 vs 市場平均 (20%)
    ret_diff = s_data["avg_return_20d"] - mkt_return
    ret_score = max(0.0, min(1.0, 0.5 + ret_diff * 5.0))
    components["return_vs_market"] = round(ret_score, 4)

    # 4. 產業廣度 (15%)
    breadth = s_data["breadth"]
    breadth_score = max(0.0, min(1.0, breadth))
    components["breadth"] = round(breadth_score, 4)

    total = (
        flow_score * 0.35 + index_score * 0.30 + ret_score * 0.20 + breadth_score * 0.15
    )
    return FactorResult(
        "sector_rotation", round(total, 4), True, 1.0, components, raw_value=total
    )


def _compute_taiwan_etf_momentum(global_data: dict | None) -> FactorResult:
    """台灣ETF動能 (4%) — EWT/0050 20d/60d 報酬 + 相對強度"""
    if global_data is None:
        return FactorResult("taiwan_etf_momentum", 0.5, False, 0.0)

    ewt_1d = global_data.get("ewt_return_1d")
    ewt_20d = global_data.get("ewt_return_20d")
    ewt_60d = global_data.get("ewt_return_60d")
    tw50_20d = global_data.get("tw50_return_20d")
    tw50_60d = global_data.get("tw50_return_60d")
    sox_1d = global_data.get("sox_return", 0)

    # Use EWT primarily, fallback to 0050.TW
    ret_20d = ewt_20d if (ewt_20d is not None and ewt_20d != 0) else tw50_20d
    ret_60d = ewt_60d if (ewt_60d is not None and ewt_60d != 0) else tw50_60d

    if ret_20d is None:
        return FactorResult("taiwan_etf_momentum", 0.5, False, 0.0)

    components = {}

    # 1. 20d 報酬 (40%)
    s20 = max(0.0, min(1.0, 0.5 + ret_20d * 4.0))
    components["return_20d"] = round(ret_20d, 4)
    components["return_20d_score"] = round(s20, 4)
    # Backward compat keys
    components["ewt_return_20d"] = round(ewt_20d, 4) if ewt_20d is not None else 0.0
    components["ewt_20d_score"] = round(s20, 4)

    # 2. 60d 報酬 (25%)
    s60 = 0.5
    if ret_60d is not None:
        s60 = max(0.0, min(1.0, 0.5 + ret_60d * 2.0))
        components["return_60d"] = round(ret_60d, 4)
    if ewt_60d is not None:
        components["ewt_return_60d"] = round(ewt_60d, 4)
    components["ewt_60d_score"] = round(s60, 4)

    # 3. 0050.TW 本地動能 (15%)
    s_tw50 = 0.5
    if tw50_20d is not None and tw50_20d != 0:
        s_tw50 = max(0.0, min(1.0, 0.5 + tw50_20d * 4.0))
        components["tw50_return_20d"] = round(tw50_20d, 4)
    components["tw50_score"] = round(s_tw50, 4)

    # 4. EWT vs SOX 相對強度 (20%)
    ewt_1d_val = ewt_1d if ewt_1d is not None else 0.0
    relative = ewt_1d_val - sox_1d
    s_rel = max(0.0, min(1.0, 0.5 + relative * 10.0))
    components["relative_strength"] = round(s_rel, 4)

    total = s20 * 0.40 + s60 * 0.25 + s_tw50 * 0.15 + s_rel * 0.20
    available = ret_20d != 0 or (ret_60d is not None and ret_60d != 0)
    return FactorResult(
        "taiwan_etf_momentum",
        round(total, 4),
        available,
        0.9,
        components,
        raw_value=total,
    )


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
    return FactorResult(
        "us_manufacturing", round(total, 4), available, 0.9, components, raw_value=total
    )


# ═══════════════════════════════════════════════════════
# Composite (decorrelated) factor computers
# ═══════════════════════════════════════════════════════


def _compute_composite_institutional(
    trust_info: dict,
    df: pd.DataFrame,
    avg_vol_20d: float,
    market_flow_stats: dict | None = None,
) -> FactorResult:
    """Composite institutional flow (15%) — decorrelated merger of foreign+trust+sync.

    Sub-factor weights: foreign 50%, trust 30%, sync 20%.
    Eliminates double-counting from correlated T86 data.
    """
    f_foreign = _compute_foreign_flow(trust_info, df, avg_vol_20d, market_flow_stats)
    f_trust = _compute_trust_flow(trust_info, df, avg_vol_20d)
    f_sync = _compute_institutional_sync(trust_info, df)

    available = f_foreign.available or f_trust.available or f_sync.available
    if not available:
        return FactorResult("composite_institutional", 0.5, False, 0.0)

    # Weighted merge with decorrelation
    sub_scores = []
    sub_weights = []
    if f_foreign.available:
        sub_scores.append(f_foreign.score)
        sub_weights.append(0.50)
    if f_trust.available:
        sub_scores.append(f_trust.score)
        sub_weights.append(0.30)
    if f_sync.available:
        sub_scores.append(f_sync.score)
        sub_weights.append(0.20)

    # Renormalize weights for available sub-factors
    w_total = sum(sub_weights)
    total = sum(s * w / w_total for s, w in zip(sub_scores, sub_weights))

    freshness = max(f_foreign.freshness, f_trust.freshness, f_sync.freshness)
    components = {
        "foreign_flow": {
            "score": f_foreign.score,
            "available": f_foreign.available,
            **f_foreign.components,
        },
        "trust_flow": {
            "score": f_trust.score,
            "available": f_trust.available,
            **f_trust.components,
        },
        "institutional_sync": {
            "score": f_sync.score,
            "available": f_sync.available,
            **f_sync.components,
        },
    }
    return FactorResult(
        "composite_institutional",
        round(total, 4),
        True,
        freshness,
        components,
        raw_value=total,
    )


def _compute_multi_scale_momentum(
    df: pd.DataFrame,
    df_tech: pd.DataFrame,
) -> FactorResult:
    """Multi-scale momentum (10%) — decorrelated merger of short+trend momentum.

    Sub-factor weights: short 50%, trend 50%.
    Uses different timescales (1-5d vs 20-60d) to capture orthogonal momentum.
    """
    f_short = _compute_short_momentum(df)
    f_trend = _compute_trend_momentum(df, df_tech)

    available = f_short.available or f_trend.available
    if not available:
        return FactorResult("multi_scale_momentum", 0.5, False, 0.0)

    sub_scores = []
    sub_weights = []
    if f_short.available:
        sub_scores.append(f_short.score)
        sub_weights.append(0.50)
    if f_trend.available:
        sub_scores.append(f_trend.score)
        sub_weights.append(0.50)

    w_total = sum(sub_weights)
    total = sum(s * w / w_total for s, w in zip(sub_scores, sub_weights))

    freshness = max(f_short.freshness, f_trend.freshness)
    components = {
        "short_momentum": {
            "score": f_short.score,
            "available": f_short.available,
            **f_short.components,
        },
        "trend_momentum": {
            "score": f_trend.score,
            "available": f_trend.available,
            **f_trend.components,
        },
    }
    return FactorResult(
        "multi_scale_momentum",
        round(total, 4),
        True,
        freshness,
        components,
        raw_value=total,
    )


def _compute_global_macro(
    global_data: dict | None,
    macro_data: dict | None,
    sector: str = "",
) -> FactorResult:
    """Global macro composite (8%) — decorrelated merger of global+us_mfg+tw_etf.

    Sub-factor weights: global_context 40%, us_manufacturing 30%, taiwan_etf 30%.
    Eliminates redundant global signals (SOX/TSM/EWT/XLI overlap).
    """
    f_global = _compute_global_context(global_data, sector=sector)
    f_us_mfg = _compute_us_manufacturing(macro_data)
    f_tw_etf = _compute_taiwan_etf_momentum(global_data)

    available = f_global.available or f_us_mfg.available or f_tw_etf.available
    if not available:
        return FactorResult("global_macro", 0.5, False, 0.0)

    sub_scores = []
    sub_weights = []
    if f_global.available:
        sub_scores.append(f_global.score)
        sub_weights.append(0.40)
    if f_us_mfg.available:
        sub_scores.append(f_us_mfg.score)
        sub_weights.append(0.30)
    if f_tw_etf.available:
        sub_scores.append(f_tw_etf.score)
        sub_weights.append(0.30)

    w_total = sum(sub_weights)
    total = sum(s * w / w_total for s, w in zip(sub_scores, sub_weights))

    freshness = max(f_global.freshness, f_us_mfg.freshness, f_tw_etf.freshness)
    components = {
        "global_context": {
            "score": f_global.score,
            "available": f_global.available,
            **f_global.components,
        },
        "us_manufacturing": {
            "score": f_us_mfg.score,
            "available": f_us_mfg.available,
            **f_us_mfg.components,
        },
        "taiwan_etf_momentum": {
            "score": f_tw_etf.score,
            "available": f_tw_etf.available,
            **f_tw_etf.components,
        },
    }
    return FactorResult(
        "global_macro", round(total, 4), True, freshness, components, raw_value=total
    )


# ── Legacy aliases for backward compatibility with tests/imports ──


def _compute_technical_trend(signals: dict, df_tech: pd.DataFrame) -> FactorResult:
    """Legacy alias → technical_signal"""
    r = _compute_technical_signal(signals, df_tech)
    return FactorResult(
        "technical_trend", r.score, r.available, r.freshness, r.components, r.raw_value
    )


def _compute_momentum(df: pd.DataFrame) -> FactorResult:
    """Legacy alias → short_momentum"""
    r = _compute_short_momentum(df)
    return FactorResult(
        "momentum", r.score, r.available, r.freshness, r.components, r.raw_value
    )


def _compute_institutional_flow(trust_info: dict, df: pd.DataFrame) -> FactorResult:
    """Legacy alias → foreign_flow (uses avg_vol_20d=10000 fallback)"""
    avg_vol = 10000.0
    if not df.empty and "volume" in df.columns:
        vol = df["volume"].dropna()
        if len(vol) >= 20:
            avg_vol = float(vol.tail(20).mean())
    r = _compute_foreign_flow(trust_info, df, avg_vol)
    return FactorResult(
        "institutional_flow",
        r.score,
        r.available,
        r.freshness,
        r.components,
        r.raw_value,
    )


def _compute_margin_retail(df: pd.DataFrame) -> FactorResult:
    """Legacy alias → margin_sentiment"""
    r = _compute_margin_sentiment(df)
    return FactorResult(
        "margin_retail", r.score, r.available, r.freshness, r.components, r.raw_value
    )


def _compute_volatility(df: pd.DataFrame, df_tech: pd.DataFrame) -> FactorResult:
    """Legacy alias → volatility_regime"""
    r = _compute_volatility_regime(df, df_tech)
    return FactorResult(
        "volatility", r.score, r.available, r.freshness, r.components, r.raw_value
    )


def _compute_sentiment(
    sentiment_scores: dict, sentiment_df: pd.DataFrame | None, stock_id: str
) -> FactorResult:
    """Legacy alias → news_sentiment"""
    r = _compute_news_sentiment(sentiment_scores, sentiment_df, stock_id)
    return FactorResult(
        "sentiment", r.score, r.available, r.freshness, r.components, r.raw_value
    )


def _compute_export_momentum(global_data: dict | None) -> FactorResult:
    """Legacy alias → taiwan_etf_momentum"""
    return _compute_taiwan_etf_momentum(global_data)


def _compute_liquidity(df: pd.DataFrame) -> FactorResult:
    """Legacy alias → liquidity_quality"""
    r = _compute_liquidity_quality(df)
    return FactorResult(
        "liquidity", r.score, r.available, r.freshness, r.components, r.raw_value
    )


def _compute_value_quality(stock_id: str) -> FactorResult:
    """Legacy alias → fundamental_value"""
    r = _compute_fundamental_value(stock_id)
    return FactorResult(
        "value_quality", r.score, r.available, r.freshness, r.components, r.raw_value
    )


# ═══════════════════════════════════════════════════════
# Phase 4: Confidence Calculator + Risk Discount
# ═══════════════════════════════════════════════════════

# Risk discount thresholds
RISK_DISCOUNTS = [
    # (condition_fn, discount, description)
    # These are evaluated per-stock using factor results and raw data
]


def _compute_confidence(
    factors: list[FactorResult],
    weights: dict[str, float],
    total_score: float,
    df: pd.DataFrame,
) -> dict:
    """多維度信心計算 + 風險折扣

    confidence = (agreement*0.30 + strength*0.30 + coverage*0.25 + freshness*0.15) × risk_discount
    """
    available_factors = [f for f in factors if f.available]

    # 1. Factor agreement 30% — weighted by factor importance
    if available_factors:
        weighted_bull = sum(
            weights.get(f.name, 0) for f in available_factors if f.score > 0.55
        )
        weighted_bear = sum(
            weights.get(f.name, 0) for f in available_factors if f.score < 0.45
        )
        agreement = max(weighted_bull, weighted_bear)
        agreement = min(agreement, 1.0)
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
            freshness = (
                sum(f.freshness * weights.get(f.name, 0) for f in available_factors)
                / total_w
            )
        else:
            freshness = 0.5
    else:
        freshness = 0.0

    raw_confidence = (
        agreement * 0.30 + strength * 0.30 + coverage * 0.25 + freshness * 0.15
    )

    # Risk discount
    risk_discount = 1.0

    if not df.empty and len(df) >= 20:
        close = df["close"].dropna()
        vol_col = df["volume"].dropna()

        # High volatility discount
        if len(close) >= 20:
            returns = close.pct_change().dropna()
            vol_annual = float(returns.tail(20).std()) * (252**0.5)
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
    fundamental_data: dict | None = None,
    per_pbr_df: pd.DataFrame | None = None,
    ex_dividend_window: bool = False,
) -> dict:
    """20 因子多維度評分

    Returns dict with all scores, confidence breakdown, regime, factor_details.
    Backward-compatible with old fields (technical_score, fundamental_score, etc.)
    """
    stock_id = stock_data["stock_id"]
    sector = STOCK_SECTOR.get(stock_id, DEFAULT_SECTOR)

    # Compute avg_vol_20d for institutional flow normalization
    avg_vol_20d = 10000.0
    if not df.empty and "volume" in df.columns:
        vol = df["volume"].dropna()
        if len(vol) >= 20:
            avg_vol_20d = float(vol.tail(20).mean())

    # Extract market flow stats for cross-sectional normalization
    market_flow_stats = None
    if sector_data and "_market_avg" in sector_data:
        mkt = sector_data["_market_avg"]
        if "flow_mean_normalized" in mkt and "flow_std_normalized" in mkt:
            market_flow_stats = {
                "mean_normalized": mkt["flow_mean_normalized"],
                "std_normalized": mkt["flow_std_normalized"],
            }

    # Compute 15 decorrelated factors (composite factors replace correlated groups)
    factors = [
        _compute_composite_institutional(
            trust_info, df, avg_vol_20d, market_flow_stats
        ),
        _compute_technical_signal(signals, df_tech),
        _compute_multi_scale_momentum(df, df_tech),
        _compute_volume_anomaly(df, df_tech),
        _compute_margin_sentiment(df),
        _compute_revenue_momentum(revenue_df),
        _compute_volatility_regime(df, df_tech),
        _compute_news_sentiment(sentiment_scores, sentiment_df, stock_id),
        _compute_global_macro(global_data, macro_data, sector=sector),
        _compute_margin_quality(stock_id, fundamental_data),
        _compute_sector_rotation(stock_id, sector_data),
        _compute_ml_ensemble(ml_scores, stock_id),
        _compute_fundamental_value(stock_id, fundamental_data, per_pbr_df),
        _compute_liquidity_quality(df),
        _compute_macro_risk(macro_data),
    ]

    # Ex-dividend filter: neutralize price-sensitive factors during ex-dividend window
    if ex_dividend_window:
        _ex_div_factors = {"multi_scale_momentum", "volume_anomaly", "technical_signal"}
        for i, f in enumerate(factors):
            if f.name in _ex_div_factors:
                factors[i] = FactorResult(
                    f.name,
                    0.5,
                    available=False,
                    freshness=0.3,
                    components={"reason": "ex_dividend_filter"},
                )

    # Compute weights with regime adjustment and missing-data redistribution
    weights = _compute_weights(factors, regime)

    # Weighted total score (only available factors contribute)
    total_score = 0.0
    for f in factors:
        w = weights.get(f.name, 0)
        if f.available:
            total_score += f.score * w

    # Confidence (compute before signal so we can use it for gating)
    conf = _compute_confidence(factors, weights, total_score, df)
    confidence = conf.get("confidence", 0.5)

    # Determine signal — raise buy/sell threshold when confidence is low
    # to filter out marginal signals that lose money after 0.585% round-trip cost
    buy_threshold = 0.60 if confidence >= 0.45 else 0.65
    sell_threshold = 0.40 if confidence >= 0.45 else 0.35

    if total_score > 0.7:
        signal = "strong_buy"
    elif total_score > buy_threshold:
        signal = "buy"
    elif total_score < 0.3:
        signal = "strong_sell"
    elif total_score < sell_threshold:
        signal = "sell"
    else:
        signal = "hold"

    # Score coverage
    score_coverage = {f.name: f.available for f in factors}
    effective_coverage = round(
        sum(BASE_WEIGHTS.get(f.name, 0) for f in factors if f.available), 2
    )

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

    # Helper to get factor score by name (supports composite sub-factor lookup)
    def _fs(name: str) -> float:
        # Direct factor lookup
        f = next((f for f in factors if f.name == name), None)
        if f:
            return round(f.score, 4)
        # Composite sub-factor lookup
        composite_name = _LEGACY_FACTOR_NAMES.get(name)
        if composite_name:
            cf = next((f for f in factors if f.name == composite_name), None)
            if cf and name in cf.components:
                sub = cf.components[name]
                return round(sub.get("score", 0.5), 4) if isinstance(sub, dict) else 0.5
        return 0.5

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
        "reasoning": "；".join(reasons)
        if reasons
        else stock_data.get("reasoning", "資料不足"),
        # Internal factors for IC tracking
        "_factors": factors,
    }


def _build_reasoning(
    factors: list[FactorResult],
    trust_info: dict,
    signals: dict,
    ml_scores: dict,
    stock_id: str,
) -> list[str]:
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
        ml_label = (
            "ML看多" if ml_val > 0.6 else ("ML看空" if ml_val < 0.4 else "ML中性")
        )
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
    em_factor = next((f for f in factors if f.name == "taiwan_etf_momentum"), None)
    if em_factor and em_factor.available:
        if em_factor.score > 0.65:
            reasons.append("台股ETF動能強勁")
        elif em_factor.score < 0.35:
            reasons.append("台股ETF動能疲弱")

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


def _event(
    step: str, status: str, progress: int, message: str = "", data: dict | None = None
) -> str:
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

        yield _event(
            "universe",
            "done",
            15,
            f"投信買超 {len(trust_data)} 支 + 基本清單 = 共 {len(universe_map)} 支",
            {"count": len(universe_map), "trust_count": len(trust_data)},
        )
    except Exception as e:
        logger.error("TWSE scanner failed: %s", e)
        universe_map = dict(STOCK_LIST)
        trust_lookup = {}
        yield _event(
            "universe",
            "error",
            15,
            f"TWSE 掃描失敗，使用基本清單 ({len(universe_map)} 支): {e}",
        )

    stock_ids = list(universe_map.keys())

    # ── Step 2: FETCH_DATA ────────────────────────────────
    yield _event(
        "fetch_data", "running", 20, f"批次抓取 {len(stock_ids)} 支股票資料..."
    )

    stock_dfs = {}
    fetch_start = (today - timedelta(days=120)).isoformat()
    end_str = today.isoformat()

    async def _fetch_one(sid):
        try:
            df = await asyncio.to_thread(get_stock_prices, sid)
            if df.empty or len(df) < 20:
                new_df = await asyncio.to_thread(
                    fetcher.fetch_all, sid, fetch_start, end_str
                )
                if not new_df.empty:
                    await asyncio.to_thread(upsert_stock_prices, new_df, sid)
                    return sid, new_df
            elif not df.empty:
                latest = df["date"].max()
                if isinstance(latest, date) and (today - latest).days >= 1:
                    new_df = await asyncio.to_thread(
                        fetcher.fetch_all, sid, fetch_start, end_str
                    )
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
        batch = stock_ids[i : i + batch_size]
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
        yield _event(
            "fetch_data",
            "running",
            min(progress, 40),
            f"已抓取 {fetched}/{len(stock_ids)} 支...",
        )

    yield _event(
        "fetch_data", "done", 40, f"資料抓取完成: {fetched}/{len(stock_ids)} 支有資料"
    )

    # ── Step 3: TECHNICAL ─────────────────────────────────
    yield _event("technical", "running", 45, "批次計算技術指標...")

    tech_results = {}  # stock_id -> signals dict
    tech_dfs = {}  # stock_id -> df_tech DataFrame
    for sid, df in stock_dfs.items():
        try:
            df_tech = analyzer.compute_all(df)
            signals = analyzer.get_signals(df_tech)
            tech_results[sid] = signals
            tech_dfs[sid] = df_tech
        except Exception as e:
            logger.warning("Technical analysis failed for %s: %s", sid, e)

    yield _event("technical", "done", 52, f"技術分析完成: {len(tech_results)} 支")

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
                # MA trend override: prevent bear when price above major MAs
                if regime == "bear":
                    current_price = close.iloc[-1]
                    ma20 = close.rolling(20).mean().iloc[-1]
                    ma60 = close.rolling(60).mean().iloc[-1]
                    if current_price > ma20 and current_price > ma60:
                        logger.info(
                            "  ├─ HMM regime override: bear → sideways "
                            "(price %.1f > MA20 %.1f & MA60 %.1f)",
                            current_price,
                            ma20,
                            ma60,
                        )
                        regime = "sideways"
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
                get_sentiment, sid, today - timedelta(days=14), today
            )
            if not sent_df.empty:
                sentiment_dfs[sid] = sent_df
                scores = sent_df["sentiment_score"].dropna()
                if not scores.empty:
                    avg = float(scores.mean())
                    sentiment_scores[sid] = (avg + 1) / 2
        except Exception:
            pass

    yield _event(
        "sentiment", "done", 62, f"情緒資料: {len(sentiment_scores)} 支有情緒分數"
    )

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
                    trainer.predict, start_date=pred_start, end_date=end_str
                )
                if result is not None:
                    sig_map = {
                        "strong_buy": 0.9,
                        "buy": 0.75,
                        "hold": 0.5,
                        "sell": 0.25,
                        "strong_sell": 0.1,
                    }
                    ml_scores[sid] = sig_map.get(result.signal, 0.5)
            except Exception as e:
                logger.warning("ML predict failed for %s: %s", sid, e)

    yield _event("ml_predict", "done", 75, f"ML 預測: {len(ml_scores)} 支有模型")

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

    yield _event(
        "global_data", "done", 78, f"全球/宏觀數據完成, 月營收 {len(revenue_lookup)} 支"
    )

    # ── Step 5.7: SECTOR AGGREGATION ──────────────────────
    sector_data = None
    try:
        sector_data = _compute_sector_aggregates(stock_dfs, trust_lookup)
    except Exception as e:
        logger.warning("Sector aggregation failed: %s", e)

    # ── Step 5.8: EX-DIVIDEND DETECTION (DB-first) ──────
    ex_div_set: set[str] = set()
    try:
        start_div = (today - timedelta(days=5)).isoformat()
        end_div = (today + timedelta(days=2)).isoformat()
        for sid in list(stock_dfs.keys())[:50]:
            cache_key = f"dividend:{sid}"
            div_df = None
            # DB-first
            cached_json = get_data_cache(cache_key, today)
            if cached_json:
                try:
                    div_df = pd.read_json(StringIO(cached_json), orient="records")
                    if "date" in div_df.columns:
                        div_df["date"] = pd.to_datetime(div_df["date"]).dt.date
                except Exception:
                    div_df = None
            if div_df is None:
                div_df = fetcher.fetch_dividend_history(
                    sid, start=start_div, end=end_div
                )
                if not div_df.empty:
                    try:
                        upsert_data_cache(
                            cache_key,
                            today,
                            div_df.to_json(orient="records", date_format="iso"),
                        )
                    except Exception:
                        pass
            if div_df is not None and not div_df.empty and "date" in div_df.columns:
                if any(0 <= (today - d).days <= 2 for d in div_df["date"]):
                    ex_div_set.add(sid)
    except Exception as e:
        logger.warning("Ex-dividend batch check failed: %s", e)

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
            pct = (
                (current_price - float(prev["close"])) / float(prev["close"]) * 100
                if prev["close"] and current_price
                else 0
            )
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
            ex_dividend_window=(sid in ex_div_set),
        )
        scored_results.append(scored)

    # Sort by total_score descending and assign ranking
    scored_results.sort(key=lambda x: x["total_score"], reverse=True)
    for i, r in enumerate(scored_results):
        r["ranking"] = i + 1

    recommendations = rank_recommendations(scored_results)

    yield _event(
        "score_rank",
        "done",
        90,
        f"評分完成: {len(scored_results)} 支 "
        f"(BUY {len(recommendations['buy_recommendations'])}, "
        f"SELL {len(recommendations['sell_recommendations'])})",
    )

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
        await asyncio.to_thread(save_market_scan, scan_records, full_replace=True)
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
                    ic_records.append(
                        {
                            "record_date": today,
                            "stock_id": r["stock_id"],
                            "factor_name": f.name,
                            "factor_score": f.score,
                        }
                    )
        if ic_records:
            await asyncio.to_thread(save_factor_ic_records, ic_records)
    except Exception as e:
        logger.error("IC tracking save failed: %s", e)

    # Generate alerts
    alert_count = 0
    try:
        from api.services.alert_service import generate_alerts_from_scan

        new_alerts = await asyncio.to_thread(
            generate_alerts_from_scan, scored_results, today
        )
        alert_count = len(new_alerts)
    except Exception as e:
        logger.error("Alert generation failed: %s", e)

    # Clean internal fields before sending to client
    for r in scored_results:
        r.pop("_factors", None)
        r.pop("_has_sentiment", None)
        r.pop("_has_ml", None)

    yield _event(
        "done",
        "done",
        100,
        "掃描完成",
        {
            "scan_date": str(today),
            "total_stocks": len(scored_results),
            "alert_count": alert_count,
            "market_regime": regime,
            "stocks": scored_results,
            "buy_recommendations": recommendations["buy_recommendations"][:5],
            "sell_recommendations": recommendations["sell_recommendations"][:5],
        },
    )


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
