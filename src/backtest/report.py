"""績效報告模組

年化報酬、Sharpe、最大回撤、勝率、月報酬熱圖
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def generate_report(backtest_result: dict) -> str:
    """生成文字績效報告"""
    r = backtest_result
    report_lines = [
        "=" * 60,
        "回測績效報告",
        "=" * 60,
        f"累積報酬率:   {r.get('total_return', 0):.2%}",
        f"年化報酬率:   {r.get('annualized_return', 0):.2%}",
        f"Sharpe Ratio: {r.get('sharpe_ratio', 0):.2f}",
        f"最大回撤:     {r.get('max_drawdown', 0):.2%}",
        f"勝率:         {r.get('win_rate', 0):.1%}",
        f"總交易次數:   {r.get('total_trades', 0)}",
        f"平均獲利:     {r.get('avg_win', 0):.2%}",
        f"平均虧損:     {r.get('avg_loss', 0):.2%}",
        f"獲利因子:     {r.get('profit_factor', 0):.2f}",
        f"最終權益:     ${r.get('final_equity', 0):,.0f}",
        "=" * 60,
    ]

    # 月報酬
    monthly = r.get("monthly_returns", [])
    if monthly:
        report_lines.append("\n月報酬率:")
        for i, ret in enumerate(monthly):
            bar = "+" * int(abs(ret) * 200) if ret > 0 else "-" * int(abs(ret) * 200)
            report_lines.append(f"  月{i+1:2d}: {ret:+.2%} {bar}")

    return "\n".join(report_lines)


def generate_comparison_report(result_a: dict, result_b: dict) -> str:
    """生成 A/B 回測比較報告

    Args:
        result_a: Approach A 回測結果
        result_b: Approach B 回測結果

    Returns:
        格式化的比較文字報告
    """
    def _fmt(val, fmt=".2%"):
        if val is None:
            return "N/A"
        return f"{val:{fmt}}"

    lines = [
        "=" * 70,
        "A/B 回測比較報告",
        "=" * 70,
        f"{'指標':<20} {'Approach A':>20} {'Approach B':>20}",
        "-" * 70,
        f"{'累積報酬率':<20} {_fmt(result_a.get('total_return')):>20} {_fmt(result_b.get('total_return')):>20}",
        f"{'年化報酬率':<20} {_fmt(result_a.get('annualized_return')):>20} {_fmt(result_b.get('annualized_return')):>20}",
        f"{'Sharpe Ratio':<20} {_fmt(result_a.get('sharpe_ratio'), '.2f'):>20} {_fmt(result_b.get('sharpe_ratio'), '.2f'):>20}",
        f"{'最大回撤':<20} {_fmt(result_a.get('max_drawdown')):>20} {_fmt(result_b.get('max_drawdown')):>20}",
        f"{'勝率':<20} {_fmt(result_a.get('win_rate'), '.1%'):>20} {_fmt(result_b.get('win_rate'), '.1%'):>20}",
        f"{'總交易次數':<20} {str(result_a.get('total_trades', 0)):>20} {str(result_b.get('total_trades', 0)):>20}",
        f"{'獲利因子':<20} {_fmt(result_a.get('profit_factor'), '.2f'):>20} {_fmt(result_b.get('profit_factor'), '.2f'):>20}",
        f"{'最終權益':<20} {'${:,.0f}'.format(result_a.get('final_equity', 0)):>20} {'${:,.0f}'.format(result_b.get('final_equity', 0)):>20}",
        "=" * 70,
    ]

    # Sharpe improvement
    sa = result_a.get("sharpe_ratio", 0) or 0
    sb = result_b.get("sharpe_ratio", 0) or 0
    diff = sb - sa
    lines.append(f"Sharpe 改善: {diff:+.2f} ({'B 優' if diff > 0 else 'A 優' if diff < 0 else '持平'})")

    return "\n".join(lines)


def generate_plotly_report(backtest_result: dict) -> dict:
    """生成 Plotly 圖表數據（供 Streamlit 使用）

    Returns:
        dict with chart data ready for plotly
    """
    equity = backtest_result.get("equity_curve", [])
    trades = backtest_result.get("trades", [])

    charts = {}

    # 1. 權益曲線
    if equity:
        charts["equity_curve"] = {
            "x": list(range(len(equity))),
            "y": equity,
            "title": "權益曲線",
        }

    # 2. 回撤曲線
    if equity:
        arr = np.array(equity)
        peak = np.maximum.accumulate(arr)
        dd = (arr - peak) / peak
        charts["drawdown"] = {
            "x": list(range(len(dd))),
            "y": dd.tolist(),
            "title": "回撤曲線",
        }

    # 3. 交易 PnL 分佈
    if trades:
        pnls = [t["pnl_pct"] for t in trades]
        charts["pnl_distribution"] = {
            "values": pnls,
            "title": "交易 PnL 分佈",
        }

    return charts
