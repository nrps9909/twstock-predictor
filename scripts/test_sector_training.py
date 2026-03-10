"""Sector training test — validate Chronos + pooled XGBoost pipeline.

Tests:
1. Single-stock Chronos training (use_chronos=True)
2. Sector pooled training (train_sector)
3. Quality gate + Chronos gate
4. Predict with Chronos model
"""

import logging
import sys
import time
from datetime import date, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("sector_test")

for name in ["httpx", "urllib3", "httpcore", "hmmlearn", "chronos", "transformers"]:
    logging.getLogger(name).setLevel(logging.WARNING)


def main():
    from src.models.trainer import ModelTrainer
    from api.services.market_service import STOCK_SECTOR

    stock_id = "2317"
    today = date.today()
    start = today - timedelta(days=3000)
    sector = STOCK_SECTOR.get(stock_id, "unknown")

    print("=" * 70)
    print(f"  SECTOR TRAINING TEST: {stock_id} (sector={sector})")
    print(f"  Date range: {start} ~ {today}")
    print("=" * 70)

    sector_stocks = [sid for sid, sec in STOCK_SECTOR.items() if sec == sector]
    print(f"\n  Sector stocks ({len(sector_stocks)}): {sector_stocks}")

    # ── Test 1: Sector Training ──
    print(f"\n[1/3] Running sector training...")
    t0 = time.time()
    trainer = ModelTrainer(stock_id)
    try:
        results = trainer.train_sector(
            sector=sector,
            start_date=start.isoformat(),
            end_date=today.isoformat(),
            epochs=100,
            max_features=20,
        )
        elapsed = time.time() - t0
        print(f"  Sector training completed in {elapsed:.1f}s")
    except Exception as e:
        print(f"  [FAIL] Sector training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # ── Test 2: Check results ──
    print("\n[2/3] Checking training results...")

    # Chronos results
    chr_res = results.get("chronos", {})
    if chr_res:
        chr_dir = chr_res.get("test_direction_acc", 0)
        chr_naive = chr_res.get("test_beats_naive", False)
        n_samples = chr_res.get("n_train_samples", 0)
        n_stocks = chr_res.get("n_sector_stocks", 0)
        print(f"\n  Chronos:")
        print(f"    train_samples={n_samples}")
        print(f"    sector_stocks={n_stocks}")
        print(f"    direction_acc={chr_dir:.4f}")
        print(f"    beats_naive={chr_naive}")
        if chr_dir > 0.52:
            print(f"    [OK] PASS: direction_acc > 0.52")
        else:
            print(f"    ~ direction_acc <= 0.52")
    else:
        print("\n  Chronos: NOT TRAINED (may be expected if pipeline loaded)")

    # XGBoost results
    xgb_res = results.get("xgboost", {})
    if xgb_res:
        xgb_dir = xgb_res.get("test_direction_acc", 0)
        print(f"\n  XGBoost Regressor:")
        print(f"    direction_acc={xgb_dir:.4f}")
        print(f"    no_learning={xgb_res.get('no_learning', None)}")
        print(f"    beats_naive={xgb_res.get('test_beats_naive', None)}")

    # XGBClassifier results
    cls_res = results.get("xgb_classifier", {})
    if cls_res:
        cls_dir = cls_res.get("test_direction_acc", 0)
        cls_score_dir = cls_res.get("test_score_dir_acc", 0)
        print(f"\n  XGBClassifier:")
        print(f"    direction_acc={cls_dir:.4f}")
        print(f"    score_dir_acc={cls_score_dir:.4f}")

    # Quality gate
    gate = results.get("quality_gate", {})
    print(f"\n  Quality Gate:")
    print(f"    chronos_passed={gate.get('chronos_passed')}")
    print(f"    chronos_dir_acc={gate.get('chronos_direction_acc', 0):.4f}")
    print(f"    lstm_passed={gate.get('lstm_passed')}")
    print(f"    xgb_passed={gate.get('xgb_passed')}")
    print(f"    xgb_cls_passed={gate.get('xgb_cls_passed')}")
    print(f"    overall_passed={gate.get('overall_passed')}")

    # Stacking
    stk_res = results.get("stacking", {})
    if stk_res:
        print(f"\n  StackingEnsemble:")
        print(f"    fitted={stk_res.get('fitted')}")
        print(f"    models={stk_res.get('models')}")

    # Config
    config = results.get("config", {})
    if config:
        print(f"\n  Config:")
        print(f"    sector={config.get('sector')}")
        print(f"    n_stocks={config.get('n_stocks')}")
        print(f"    n_total_rows={config.get('n_total_rows')}")
        print(f"    n_features={config.get('n_features')}")
        print(f"    training_type={config.get('training_type')}")

    # ── Test 3: Predict ──
    print(f"\n[3/3] Running prediction with trained models...")
    try:
        pred_start = (today - timedelta(days=200)).isoformat()
        pred = trainer.predict(pred_start, today.isoformat())
        if pred:
            print(f"  signal={pred.signal}")
            print(f"  signal_strength={pred.signal_strength:.4f}")
            print(f"  predicted_returns={pred.predicted_returns}")
            print(
                f"  model_used={'chronos' if trainer.chronos else 'lstm' if trainer.lstm else 'xgb_only'}"
            )
            print(f"  [OK] Prediction succeeded")
        else:
            print("  Prediction returned None (all models may have failed gate)")
    except Exception as e:
        print(f"  Prediction error: {e}")
        import traceback

        traceback.print_exc()

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    checks = []

    # Sector training completed
    checks.append(
        (
            "Sector training completed",
            config.get("training_type") == "sector_pooled",
            config.get("training_type", "N/A"),
        )
    )

    # Data pooling
    n_total = config.get("n_total_rows", 0)
    checks.append((f"Pooled data > 2000 rows", n_total > 2000, str(n_total)))

    # At least one model passed
    checks.append(("Quality gate passed", gate.get("overall_passed", False), ""))

    # Chronos trained
    checks.append(("Chronos trained", bool(chr_res), str(bool(chr_res))))

    for name, passed, val in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}: {val}")

    all_passed = all(p for _, p, _ in checks)
    print(f"\n  {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
