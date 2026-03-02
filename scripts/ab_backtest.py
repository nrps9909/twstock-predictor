"""A/B 回測驗證 — 比較新舊方法的績效差異

Approach A: 傳統方法（pct_change 標籤，無 Triple Barrier）
Approach B: 新方法（Triple Barrier 標籤 + Purged CV + 樣本權重）

比較指標：Sharpe Ratio, Max Drawdown, Win Rate, paired t-test on daily returns
"""

import argparse
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.report import generate_report, generate_comparison_report
from src.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class ABBacktester:
    """A/B 回測比較器"""

    def __init__(
        self,
        stock_id: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000,
    ):
        self.stock_id = stock_id
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.result_a: dict | None = None
        self.result_b: dict | None = None

    def run_approach_a(self, epochs: int = 30, seq_len: int = 60) -> dict:
        """Approach A: 傳統方法（無 Triple Barrier）

        - 使用 pct_change(5).shift(-5) 標籤
        - 無樣本唯一性權重
        """
        logger.info("=== Approach A: 傳統方法 ===")
        trainer = ModelTrainer(self.stock_id)
        train_results = trainer.train(
            start_date=self.start_date,
            end_date=self.end_date,
            epochs=epochs,
            seq_len=seq_len,
            use_triple_barrier=False,
        )

        backtest_result = self._run_backtest(trainer, seq_len)
        backtest_result["train_results"] = train_results
        backtest_result["approach"] = "A (Traditional)"
        self.result_a = backtest_result
        return backtest_result

    def run_approach_b(self, epochs: int = 30, seq_len: int = 60) -> dict:
        """Approach B: 新方法（Triple Barrier + Purged CV + 樣本權重）

        - Triple Barrier 標籤（ATR-based）
        - 樣本唯一性權重
        - Purged split
        """
        logger.info("=== Approach B: 新方法 (Triple Barrier) ===")
        trainer = ModelTrainer(self.stock_id)
        train_results = trainer.train(
            start_date=self.start_date,
            end_date=self.end_date,
            epochs=epochs,
            seq_len=seq_len,
            use_triple_barrier=True,
        )

        backtest_result = self._run_backtest(trainer, seq_len)
        backtest_result["train_results"] = train_results
        backtest_result["approach"] = "B (Triple Barrier)"
        self.result_b = backtest_result
        return backtest_result

    def _run_backtest(self, trainer: ModelTrainer, seq_len: int) -> dict:
        """用 trainer 的預測結果進行回測"""
        config = BacktestConfig(initial_capital=self.initial_capital)
        engine = BacktestEngine(config)

        # 取得預測信號
        result = trainer.predict(self.start_date, self.end_date, seq_len)
        if result is None:
            logger.error("無法產生預測")
            return {"error": "no predictions"}

        # 建構 signals DataFrame（簡化版：使用最後一日的信號）
        signals = pd.DataFrame([{
            "date": date.fromisoformat(self.end_date),
            "stock_id": self.stock_id,
            "signal": result.signal,
            "confidence": result.signal_strength,
            "position_size": min(result.signal_strength * 0.2, 0.2),
        }])

        from src.analysis.features import FeatureEngineer
        fe = FeatureEngineer()
        df = fe.build_features(self.stock_id, self.start_date, self.end_date)
        if df.empty:
            return {"error": "no data"}

        prices = df[["date", "close"]].copy()
        prices["stock_id"] = self.stock_id
        prices["date"] = pd.to_datetime(prices["date"]).dt.date

        return engine.run(signals, prices)

    def compare(self) -> dict:
        """比較兩種方法的績效

        Returns:
            比較結果 dict（含 paired t-test）
        """
        if self.result_a is None or self.result_b is None:
            raise RuntimeError("請先執行 run_approach_a() 和 run_approach_b()")

        a = self.result_a
        b = self.result_b

        comparison = {
            "sharpe_a": a.get("sharpe_ratio", 0),
            "sharpe_b": b.get("sharpe_ratio", 0),
            "sharpe_improvement": b.get("sharpe_ratio", 0) - a.get("sharpe_ratio", 0),
            "max_dd_a": a.get("max_drawdown", 0),
            "max_dd_b": b.get("max_drawdown", 0),
            "win_rate_a": a.get("win_rate", 0),
            "win_rate_b": b.get("win_rate", 0),
            "total_return_a": a.get("total_return", 0),
            "total_return_b": b.get("total_return", 0),
        }

        # Paired t-test on daily returns
        returns_a = np.array(a.get("equity_curve", [0, 0]))
        returns_b = np.array(b.get("equity_curve", [0, 0]))

        daily_ret_a = np.diff(returns_a) / returns_a[:-1] if len(returns_a) > 1 else np.array([0])
        daily_ret_b = np.diff(returns_b) / returns_b[:-1] if len(returns_b) > 1 else np.array([0])

        # Align lengths
        min_len = min(len(daily_ret_a), len(daily_ret_b))
        if min_len > 1:
            t_stat, p_value = stats.ttest_rel(
                daily_ret_b[:min_len], daily_ret_a[:min_len]
            )
            comparison["t_statistic"] = float(t_stat)
            comparison["p_value"] = float(p_value)
            comparison["significant"] = p_value < 0.05
        else:
            comparison["t_statistic"] = 0.0
            comparison["p_value"] = 1.0
            comparison["significant"] = False

        # Winner
        if comparison["sharpe_b"] > comparison["sharpe_a"]:
            comparison["winner"] = "B (Triple Barrier)"
        elif comparison["sharpe_a"] > comparison["sharpe_b"]:
            comparison["winner"] = "A (Traditional)"
        else:
            comparison["winner"] = "Tie"

        logger.info("A/B 比較結果: %s", comparison)
        return comparison


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="A/B Backtest: Traditional vs Triple Barrier")
    parser.add_argument("--stock", default="2330", help="Stock ID")
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default="2025-12-31", help="End date")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital")
    args = parser.parse_args()

    ab = ABBacktester(args.stock, args.start, args.end, args.capital)

    print("\n[1/3] Running Approach A (Traditional)...")
    result_a = ab.run_approach_a(epochs=args.epochs)
    print(generate_report(result_a))

    print("\n[2/3] Running Approach B (Triple Barrier)...")
    result_b = ab.run_approach_b(epochs=args.epochs)
    print(generate_report(result_b))

    print("\n[3/3] Comparing...")
    comparison = ab.compare()
    print(generate_comparison_report(result_a, result_b))

    print(f"\nWinner: {comparison['winner']}")
    if comparison.get("significant"):
        print(f"  Statistically significant (p={comparison['p_value']:.4f})")
    else:
        print(f"  Not statistically significant (p={comparison['p_value']:.4f})")


if __name__ == "__main__":
    main()
