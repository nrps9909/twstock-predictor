"""技術指標計算模組 — 使用 ta library

對應課本 CH9 技術分析：
- 移動平均線 (SMA/EMA)
- KD 隨機指標
- RSI 相對強弱指標
- MACD 指數平滑異同移動平均線
- 乖離率 (BIAS)
- 布林通道 (Bollinger Bands)
- OBV 能量潮
- DMI/ADX 趨向指標

Note: 相容 ta 0.5.x (n=) 和 ta 0.7+ (window=)
"""

import inspect as _inspect

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import StochasticOscillator, RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

from src.utils.constants import TECHNICAL_PARAMS

# ta API 版本偵測：0.5.x 用 n=, 0.7+ 用 window=
_USE_WINDOW = "window" in _inspect.signature(SMAIndicator.__init__).parameters


class TechnicalAnalyzer:
    """技術指標計算器"""

    def __init__(self, params: dict | None = None):
        self.params = params or TECHNICAL_PARAMS

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算所有技術指標，回傳加入指標欄位的 DataFrame

        Args:
            df: 必須包含 date, open, high, low, close, volume 欄位

        Returns:
            加入所有技術指標欄位的 DataFrame（原始資料不變）
        """
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ── 移動平均線 SMA ───────────────────────────────
        for w in self.params["sma_windows"]:
            sma = SMAIndicator(close=close, **{"window" if _USE_WINDOW else "n": w})
            df[f"sma_{w}"] = sma.sma_indicator()

        # ── 指數移動平均線 EMA ───────────────────────────
        for w in self.params["ema_windows"]:
            ema = EMAIndicator(close=close, **{"window" if _USE_WINDOW else "n": w})
            df[f"ema_{w}"] = ema.ema_indicator()

        # ── KD 隨機指標 ─────────────────────────────────
        kd_kw = (
            {"window": self.params["kd_window"], "smooth_window": 3}
            if _USE_WINDOW else
            {"n": self.params["kd_window"], "d_n": 3}
        )
        kd = StochasticOscillator(high=high, low=low, close=close, **kd_kw)
        df["kd_k"] = kd.stoch()
        df["kd_d"] = kd.stoch_signal()

        # ── RSI 相對強弱指標 ─────────────────────────────
        rsi = RSIIndicator(close=close, **{"window" if _USE_WINDOW else "n": self.params["rsi_window"]})
        df["rsi_14"] = rsi.rsi()

        # ── MACD ─────────────────────────────────────────
        macd_kw = (
            {"window_fast": self.params["macd_fast"],
             "window_slow": self.params["macd_slow"],
             "window_sign": self.params["macd_signal"]}
            if _USE_WINDOW else
            {"n_fast": self.params["macd_fast"],
             "n_slow": self.params["macd_slow"],
             "n_sign": self.params["macd_signal"]}
        )
        macd = MACD(close=close, **macd_kw)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # ── 乖離率 BIAS ─────────────────────────────────
        for w in self.params["bias_windows"]:
            sma_w = SMAIndicator(close=close, **{"window" if _USE_WINDOW else "n": w}).sma_indicator()
            df[f"bias_{w}"] = ((close - sma_w) / sma_w) * 100

        # ── 布林通道 Bollinger Bands ────────────────────
        bb = BollingerBands(close=close, **{"window" if _USE_WINDOW else "n": self.params["bb_window"]})
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pband"] = bb.bollinger_pband()  # %B

        # ── OBV 能量潮 ──────────────────────────────────
        obv = OnBalanceVolumeIndicator(close=close, volume=volume)
        df["obv"] = obv.on_balance_volume()

        # ── DMI / ADX 趨向指標 ──────────────────────────
        adx = ADXIndicator(
            high=high, low=low, close=close,
            **{"window" if _USE_WINDOW else "n": self.params["adx_window"]},
        )
        df["adx"] = adx.adx()
        df["di_plus"] = adx.adx_pos()
        df["di_minus"] = adx.adx_neg()

        # ── 報酬率 ──────────────────────────────────────
        df["return_1d"] = close.pct_change(1)
        df["return_5d"] = close.pct_change(5)
        df["return_20d"] = close.pct_change(20)

        return df

    def get_signals(self, df: pd.DataFrame) -> dict:
        """根據技術指標產生買賣訊號

        Returns:
            {
                "kd": {"signal": "buy"|"sell"|"neutral", "reason": str},
                "rsi": {...},
                "macd": {...},
                "bias": {...},
                "bb": {...},
                "summary": {"signal": str, "score": float}
            }
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]
        signals = {}
        score = 0  # 正=偏多, 負=偏空

        # KD 訊號
        k, d = latest.get("kd_k", 50), latest.get("kd_d", 50)
        if k < 20 and d < 20:
            signals["kd"] = {"signal": "buy", "reason": f"KD 超賣區 (K={k:.1f}, D={d:.1f})"}
            score += 1
        elif k > 80 and d > 80:
            signals["kd"] = {"signal": "sell", "reason": f"KD 超買區 (K={k:.1f}, D={d:.1f})"}
            score -= 1
        elif len(df) >= 2 and df.iloc[-2]["kd_k"] < df.iloc[-2]["kd_d"] and k > d:
            signals["kd"] = {"signal": "buy", "reason": "KD 黃金交叉"}
            score += 1
        elif len(df) >= 2 and df.iloc[-2]["kd_k"] > df.iloc[-2]["kd_d"] and k < d:
            signals["kd"] = {"signal": "sell", "reason": "KD 死亡交叉"}
            score -= 1
        else:
            signals["kd"] = {"signal": "neutral", "reason": f"KD 中性 (K={k:.1f}, D={d:.1f})"}

        # RSI 訊號
        rsi_val = latest.get("rsi_14", 50)
        if rsi_val < 30:
            signals["rsi"] = {"signal": "buy", "reason": f"RSI 超賣 ({rsi_val:.1f})"}
            score += 1
        elif rsi_val > 70:
            signals["rsi"] = {"signal": "sell", "reason": f"RSI 超買 ({rsi_val:.1f})"}
            score -= 1
        else:
            signals["rsi"] = {"signal": "neutral", "reason": f"RSI 中性 ({rsi_val:.1f})"}

        # MACD 訊號
        macd_val = latest.get("macd", 0)
        macd_sig = latest.get("macd_signal", 0)
        macd_hist = latest.get("macd_hist", 0)
        if len(df) >= 2:
            prev_hist = df.iloc[-2].get("macd_hist", 0)
            if prev_hist < 0 and macd_hist > 0:
                signals["macd"] = {"signal": "buy", "reason": "MACD 柱狀體翻正（多方動能增強）"}
                score += 1
            elif prev_hist > 0 and macd_hist < 0:
                signals["macd"] = {"signal": "sell", "reason": "MACD 柱狀體翻負（空方動能增強）"}
                score -= 1
            else:
                direction = "多方" if macd_hist > 0 else "空方"
                signals["macd"] = {"signal": "neutral", "reason": f"MACD {direction}持續"}
        else:
            signals["macd"] = {"signal": "neutral", "reason": "MACD 資料不足"}

        # 乖離率 BIAS 訊號（以 10日 為主）
        bias_10 = latest.get("bias_10", 0)
        if bias_10 < -3:
            signals["bias"] = {"signal": "buy", "reason": f"10日乖離率過低 ({bias_10:.2f}%)"}
            score += 1
        elif bias_10 > 3:
            signals["bias"] = {"signal": "sell", "reason": f"10日乖離率過高 ({bias_10:.2f}%)"}
            score -= 1
        else:
            signals["bias"] = {"signal": "neutral", "reason": f"乖離率正常 ({bias_10:.2f}%)"}

        # 布林通道 BB 訊號
        bb_pband = latest.get("bb_pband", 0.5)
        if bb_pband < 0:
            signals["bb"] = {"signal": "buy", "reason": "價格跌破布林下軌"}
            score += 1
        elif bb_pband > 1:
            signals["bb"] = {"signal": "sell", "reason": "價格突破布林上軌"}
            score -= 1
        else:
            signals["bb"] = {"signal": "neutral", "reason": f"布林通道內 (%B={bb_pband:.2f})"}

        # 綜合訊號
        total_indicators = 5
        if score >= 2:
            summary_signal = "buy"
        elif score <= -2:
            summary_signal = "sell"
        else:
            summary_signal = "hold"

        # 正規化分數到 0~1
        normalized = (score + total_indicators) / (2 * total_indicators)
        signals["summary"] = {
            "signal": summary_signal,
            "score": round(normalized, 2),
            "raw_score": score,
            "max_score": total_indicators,
        }

        return signals

    def generate_chart_data(self, df: pd.DataFrame) -> dict:
        """為 Plotly 圖表準備資料結構

        Returns:
            {
                "ohlcv": DataFrame,
                "ma_lines": {name: Series},
                "kd": DataFrame(kd_k, kd_d),
                "rsi": Series,
                "macd": DataFrame(macd, macd_signal, macd_hist),
                "bb": DataFrame(bb_upper, bb_middle, bb_lower),
            }
        """
        return {
            "ohlcv": df[["date", "open", "high", "low", "close", "volume"]],
            "ma_lines": {
                f"SMA{w}": df[f"sma_{w}"]
                for w in self.params["sma_windows"]
                if f"sma_{w}" in df.columns
            },
            "kd": df[["date", "kd_k", "kd_d"]],
            "rsi": df[["date", "rsi_14"]],
            "macd": df[["date", "macd", "macd_signal", "macd_hist"]],
            "bb": df[["date", "bb_upper", "bb_middle", "bb_lower"]],
        }
