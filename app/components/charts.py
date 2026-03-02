"""Plotly 圖表元件 — K 線、技術指標、情緒趨勢、預測走勢"""

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ── 統一圖表主題 ──────────────────────────────────────────
# 與 theme.py 的 CSS variables 對齊
_BG = "#0B0F19"
_BG_PAPER = "#131825"
_GRID = "rgba(255,255,255,0.04)"
_TEXT = "#E2E4E9"
_TEXT_DIM = "#8B90A0"
_GOLD = "#D4A017"
_RED = "#EF5350"   # 台股紅漲
_GREEN = "#26A69A"  # 台股綠跌

_CHART_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=_BG_PAPER,
        plot_bgcolor=_BG,
        font=dict(family="Noto Sans TC, JetBrains Mono, sans-serif", color=_TEXT, size=12),
        title=dict(font=dict(size=14, color=_TEXT_DIM), x=0.01, xanchor="left"),
        xaxis=dict(
            gridcolor=_GRID, zeroline=False,
            tickfont=dict(size=10, color=_TEXT_DIM),
        ),
        yaxis=dict(
            gridcolor=_GRID, zeroline=False,
            tickfont=dict(size=10, color=_TEXT_DIM),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color=_TEXT_DIM),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=55, r=15, t=50, b=25),
        colorway=[_GOLD, "#42A5F5", "#AB47BC", "#FF7043", "#66BB6A", "#29B6F6"],
    )
)
pio.templates["twstock"] = _CHART_TEMPLATE
pio.templates.default = "twstock"


def create_candlestick_chart(
    df: pd.DataFrame,
    ma_lines: dict[str, pd.Series] | None = None,
    title: str = "",
    height: int = 600,
) -> go.Figure:
    """K 線圖 + 移動平均線 + 成交量

    Args:
        df: 必須含 date, open, high, low, close, volume
        ma_lines: {名稱: Series} 均線資料
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # K 線
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K線",
            increasing_line_color=_RED,
            decreasing_line_color=_GREEN,
            increasing_fillcolor=_RED,
            decreasing_fillcolor=_GREEN,
        ),
        row=1, col=1,
    )

    # 移動平均線
    ma_colors = [_GOLD, "#42A5F5", "#AB47BC", "#FF7043"]
    if ma_lines:
        for i, (name, series) in enumerate(ma_lines.items()):
            fig.add_trace(
                go.Scatter(
                    x=df["date"], y=series,
                    mode="lines",
                    name=name,
                    line=dict(width=1.5, color=ma_colors[i % len(ma_colors)]),
                ),
                row=1, col=1,
            )

    # 成交量
    colors = [
        _RED if c >= o else _GREEN
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df["date"], y=df["volume"],
            name="成交量",
            marker_color=colors,
            opacity=0.7,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_rangeslider_visible=False,
        template="twstock",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=30),
    )
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="成交量(張)", row=2, col=1)

    return fig


def create_kd_chart(df: pd.DataFrame, height: int = 250) -> go.Figure:
    """KD 指標圖"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["kd_k"],
        mode="lines", name="K",
        line=dict(color=_GOLD, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["kd_d"],
        mode="lines", name="D",
        line=dict(color="#42A5F5", width=1.5),
    ))

    # 超買/超賣區域
    fig.add_hline(y=80, line_dash="dash", line_color=_RED, opacity=0.4,
                  annotation_text="超買 80")
    fig.add_hline(y=20, line_dash="dash", line_color=_GREEN, opacity=0.4,
                  annotation_text="超賣 20")

    fig.update_layout(
        title="KD 隨機指標",
        height=height,
        template="twstock",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=60, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_rsi_chart(df: pd.DataFrame, height: int = 250) -> go.Figure:
    """RSI 指標圖"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["rsi_14"],
        mode="lines", name="RSI(14)",
        line=dict(color="#AB47BC", width=1.5),
    ))

    fig.add_hline(y=70, line_dash="dash", line_color=_RED, opacity=0.4,
                  annotation_text="超買 70")
    fig.add_hline(y=30, line_dash="dash", line_color=_GREEN, opacity=0.4,
                  annotation_text="超賣 30")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3)

    fig.update_layout(
        title="RSI 相對強弱指標",
        height=height,
        template="twstock",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=60, r=20, t=40, b=30),
    )
    return fig


def create_macd_chart(df: pd.DataFrame, height: int = 300) -> go.Figure:
    """MACD 圖（DIF + MACD Signal + 柱狀體）"""
    fig = go.Figure()

    # 柱狀體（OSC）
    colors = [_RED if v >= 0 else _GREEN for v in df["macd_hist"]]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["macd_hist"],
        name="柱狀體",
        marker_color=colors,
        opacity=0.6,
    ))

    # DIF (MACD line)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["macd"],
        mode="lines", name="DIF",
        line=dict(color=_GOLD, width=1.5),
    ))

    # MACD Signal
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["macd_signal"],
        mode="lines", name="MACD 訊號線",
        line=dict(color="#42A5F5", width=1.5),
    ))

    fig.add_hline(y=0, line_color="gray", opacity=0.3)

    fig.update_layout(
        title="MACD 指標",
        height=height,
        template="twstock",
        margin=dict(l=60, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_bollinger_chart(df: pd.DataFrame, height: int = 400) -> go.Figure:
    """布林通道圖"""
    fig = go.Figure()

    # 上軌（先畫，作為 fill 的上界）
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["bb_upper"],
        mode="lines", name="上軌",
        line=dict(color=_RED, width=1, dash="dash"),
    ))
    # 下軌（fill="tonexty" 填滿上軌→下軌整個帶）
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["bb_lower"],
        mode="lines", name="下軌",
        line=dict(color=_GREEN, width=1, dash="dash"),
        fill="tonexty",
        fillcolor="rgba(66, 165, 245, 0.1)",
    ))
    # 中軌
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["bb_middle"],
        mode="lines", name="中軌 (MA20)",
        line=dict(color=_GOLD, width=1),
    ))
    # 收盤價
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"],
        mode="lines", name="收盤價",
        line=dict(color="white", width=1.5),
    ))

    fig.update_layout(
        title="布林通道",
        height=height,
        template="twstock",
        margin=dict(l=60, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_sentiment_trend_chart(
    df: pd.DataFrame,
    price_df: pd.DataFrame | None = None,
    height: int = 400,
) -> go.Figure:
    """情緒趨勢圖（含股價對比）

    Args:
        df: 每日情緒聚合資料，含 date, sentiment_score, sentiment_ma5
        price_df: 股價資料，含 date, close（可選，用於雙軸對比）
    """
    if price_df is not None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    # 股價放主軸（左），符合交易者閱讀習慣
    if price_df is not None:
        fig.add_trace(go.Scatter(
            x=price_df["date"], y=price_df["close"],
            mode="lines", name="收盤價",
            line=dict(color="#42A5F5", width=2),
        ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["sentiment_score"],
        mode="lines+markers", name="情緒分數",
        line=dict(color=_GOLD, width=1),
        marker=dict(size=4),
    ), secondary_y=True if price_df is not None else None)

    if "sentiment_ma5" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["sentiment_ma5"],
            mode="lines", name="5日情緒均線",
            line=dict(color="#FF7043", width=2),
        ), secondary_y=True if price_df is not None else None)

    if price_df is not None:
        fig.update_yaxes(title_text="收盤價", secondary_y=False)
        fig.update_yaxes(title_text="情緒分數 (-1 ~ 1)", secondary_y=True)
    else:
        fig.update_yaxes(title_text="情緒分數 (-1 ~ 1)")

    fig.add_hline(y=0, line_color="gray", opacity=0.3, secondary_y=True if price_df is not None else None)

    fig.update_layout(
        title="市場情緒趨勢",
        height=height,
        template="twstock",
        margin=dict(l=60, r=60, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_prediction_chart(
    history_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    height: int = 500,
) -> go.Figure:
    """預測走勢圖（歷史 + 預測 + 信心區間）

    Args:
        history_df: 含 date, close 的歷史資料
        prediction_df: 含 target_date, predicted_price, confidence_lower, confidence_upper
    """
    fig = go.Figure()

    # 歷史走勢
    fig.add_trace(go.Scatter(
        x=history_df["date"], y=history_df["close"],
        mode="lines", name="歷史收盤價",
        line=dict(color="white", width=2),
    ))

    # 信心區間（填充）
    fig.add_trace(go.Scatter(
        x=prediction_df["target_date"],
        y=prediction_df["confidence_upper"],
        mode="lines", name="預測區間上界",
        line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=prediction_df["target_date"],
        y=prediction_df["confidence_lower"],
        mode="lines", name="預測區間",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(212, 160, 23, 0.15)",
    ))

    # 預測走勢
    fig.add_trace(go.Scatter(
        x=prediction_df["target_date"],
        y=prediction_df["predicted_price"],
        mode="lines+markers", name="預測價格",
        line=dict(color=_GOLD, width=2, dash="dash"),
        marker=dict(size=6),
    ))

    # 連接線（歷史最後一點到預測第一點）
    if not history_df.empty and not prediction_df.empty:
        fig.add_trace(go.Scatter(
            x=[history_df["date"].iloc[-1], prediction_df["target_date"].iloc[0]],
            y=[history_df["close"].iloc[-1], prediction_df["predicted_price"].iloc[0]],
            mode="lines",
            line=dict(color=_GOLD, width=1, dash="dot"),
            showlegend=False,
        ))

    fig.update_layout(
        title="走勢預測",
        height=height,
        template="twstock",
        margin=dict(l=60, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="價格 (TWD)")

    return fig


def create_radar_chart(scores: dict[str, float], height: int = 400) -> go.Figure:
    """三維雷達圖（技術面/情緒面/基本面評分）

    Args:
        scores: {"技術面": 0.7, "情緒面": 0.6, "籌碼面": 0.5}
    """
    categories = list(scores.keys())
    values = list(scores.values())
    # 封閉雷達圖
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(212, 160, 23, 0.25)",
        line=dict(color=_GOLD, width=2),
        name="綜合評分",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=height,
        template="twstock",
        margin=dict(l=60, r=60, t=40, b=40),
    )
    return fig
