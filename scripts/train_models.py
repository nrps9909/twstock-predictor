#!/usr/bin/env python3
"""訓練模型

Usage:
    python scripts/train_models.py 2330            # 訓練台積電模型
    python scripts/train_models.py 2330 --epochs 100
    python scripts/train_models.py all              # 訓練所有股票
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.trainer import ModelTrainer
from src.db.database import init_db
from src.utils.constants import STOCK_LIST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def train_stock(stock_id: str, days: int, epochs: int, test_ratio: float):
    """訓練單支股票的模型"""
    print(f"\n{'='*60}")
    print(f"訓練 {stock_id} ({STOCK_LIST.get(stock_id, stock_id)})")
    print(f"{'='*60}")

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    trainer = ModelTrainer(stock_id)
    try:
        results = trainer.train(
            start_date.isoformat(),
            end_date.isoformat(),
            epochs=epochs,
            test_ratio=test_ratio,
        )
        print(f"\n✅ 訓練結果:")
        for model, metrics in results.items():
            print(f"  {model}: {metrics}")
        return True
    except Exception as e:
        print(f"\n❌ 訓練失敗: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="訓練台股預測模型")
    parser.add_argument("stock_id", help="股票代號或 'all'")
    parser.add_argument("--days", type=int, default=365, help="訓練資料天數")
    parser.add_argument("--epochs", type=int, default=50, help="LSTM 訓練輪數")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="測試集比例")
    args = parser.parse_args()

    init_db()

    if args.stock_id == "all":
        success = 0
        for stock_id in STOCK_LIST:
            if train_stock(stock_id, args.days, args.epochs, args.test_ratio):
                success += 1
        print(f"\n完成: {success}/{len(STOCK_LIST)} 支股票")
    else:
        train_stock(args.stock_id, args.days, args.epochs, args.test_ratio)


if __name__ == "__main__":
    main()
