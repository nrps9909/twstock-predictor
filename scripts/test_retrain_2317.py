"""Retrain test for 2317 (Hon Hai) — validate ML pipeline fixes.

Checks:
1. Sample weight mean ~= 1.0 (was 0.68)
2. XGBoost regressor no_learning = False
3. XGBClassifier train/val gap < 0.15 (was 0.24)
4. LSTM direction_acc > 0.52
5. Quality gate results
"""

import logging
import sys
import time
from datetime import date, timedelta

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("retrain_test")

# Suppress noisy loggers
for name in ["httpx", "urllib3", "httpcore", "hmmlearn"]:
    logging.getLogger(name).setLevel(logging.WARNING)


def main():
    from src.models.trainer import ModelTrainer
    from src.analysis.features import (
        FEATURE_COLUMNS,
        FeatureEngineer,
        TB_TARGET_COLUMN,
        TB_WEIGHT_COLUMN,
    )

    stock_id = "2317"
    today = date(2026, 3, 6)  # latest data date
    start = today - timedelta(days=3000)

    print("=" * 70)
    print(f"  RETRAIN TEST: {stock_id} Hon Hai")
    print(f"  Date range: {start} ~ {today}")
    print("=" * 70)

    # ── Step 1: Build features and check weights ──
    print("\n[1/4] Building features...")
    fe = FeatureEngineer()
    df = fe.build_features(stock_id, start.isoformat(), today.isoformat())
    print(f"  Total samples: {len(df)}")

    if TB_WEIGHT_COLUMN in df.columns:
        raw_weights = df[TB_WEIGHT_COLUMN].values
        non_zero = raw_weights[raw_weights > 0]
        print(f"  Raw weight stats (before prepare_tabular):")
        print(f"    mean={non_zero.mean():.4f}, std={non_zero.std():.4f}")
        print(f"    non-zero count: {len(non_zero)}/{len(raw_weights)}")

    # Check weight re-normalization in prepare_tabular
    feature_cols = [c for c in df.columns if c in FEATURE_COLUMNS][:15]
    X, y, w = fe.prepare_tabular(
        df, feature_cols, TB_TARGET_COLUMN, weight_col=TB_WEIGHT_COLUMN
    )
    if w is not None:
        w_pos = w[w > 0]
        w_mean = w_pos.mean() if len(w_pos) > 0 else 0
        print(f"  After prepare_tabular re-normalization:")
        print(f"    weight mean={w_mean:.4f} (should be ~=1.0)")
        print(f"    samples with weight>0: {len(w_pos)}/{len(w)}")
        if abs(w_mean - 1.0) < 0.02:
            print("    [OK] PASS: weight mean ~= 1.0")
        else:
            print(f"    [FAIL] FAIL: weight mean = {w_mean:.4f}")

    # ── Step 2: Train model ──
    print(f"\n[2/4] Training model (epochs=100)...")
    t0 = time.time()
    trainer = ModelTrainer(stock_id)
    results = trainer.train(
        start_date=start.isoformat(),
        end_date=today.isoformat(),
        epochs=100,
        seq_len=40,
        max_features=20,
    )
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.1f}s")

    # ── Step 3: Check results ──
    print("\n[3/4] Checking training results...")

    # LSTM results
    lstm_res = results.get("lstm", {})
    if lstm_res:
        lstm_dir = lstm_res.get("test_direction_acc", 0)
        lstm_train = lstm_res.get("final_train_loss", 0)
        lstm_val = lstm_res.get("final_val_loss", 0)
        lstm_test = lstm_res.get("test_loss", 0)
        print(f"\n  LSTM:")
        print(f"    train_loss={lstm_train:.6f}")
        print(f"    val_loss={lstm_val:.6f}")
        print(f"    test_loss={lstm_test:.6f}")
        print(f"    direction_acc={lstm_dir:.4f}")
        if lstm_dir > 0.55:
            print(f"    [OK] PASS: direction_acc > 0.55")
        elif lstm_dir > 0.52:
            print(f"    ~ MARGINAL: direction_acc > 0.52 but < 0.55")
        else:
            print(f"    [FAIL] FAIL: direction_acc <= 0.52")

    # XGBoost regressor results
    xgb_res = results.get("xgboost", {})
    if xgb_res:
        xgb_dir = xgb_res.get("test_direction_acc", 0)
        no_learning = xgb_res.get("no_learning", None)
        print(f"\n  XGBoost Regressor:")
        print(f"    direction_acc={xgb_dir:.4f}")
        print(f"    no_learning={no_learning}")
        print(f"    train_mse={xgb_res.get('train_mse', 'N/A')}")
        print(f"    val_mse={xgb_res.get('val_mse', 'N/A')}")
        if no_learning is False:
            print(f"    [OK] PASS: model learned features")
        elif no_learning is True:
            print(f"    [FAIL] FAIL: no learning detected")

    # XGBClassifier results
    cls_res = results.get("xgb_classifier", {})
    if cls_res:
        cls_train = cls_res.get("train_accuracy", 0)
        cls_val = cls_res.get("val_accuracy", 0)
        cls_dir = cls_res.get("test_direction_acc", 0)
        cls_score_dir = cls_res.get("test_score_dir_acc", 0)
        gap = abs(cls_train - cls_val) if cls_train and cls_val else 0
        print(f"\n  XGBClassifier:")
        print(f"    train_acc={cls_train:.4f}")
        print(f"    val_acc={cls_val:.4f}")
        print(f"    train/val gap={gap:.4f}")
        print(f"    test_direction_acc={cls_dir:.4f}")
        print(f"    test_score_dir_acc={cls_score_dir:.4f}")
        if gap < 0.15:
            print(f"    [OK] PASS: overfitting gap < 0.15")
        else:
            print(f"    [FAIL] FAIL: overfitting gap = {gap:.4f} (was 0.24)")

    # Quality gate
    gate = results.get("quality_gate", {})
    print(f"\n  Quality Gate:")
    print(f"    lstm_passed={gate.get('lstm_passed')}")
    print(f"    xgb_passed={gate.get('xgb_passed')}")
    print(f"    xgb_cls_passed={gate.get('xgb_cls_passed')}")
    print(f"    overall_passed={gate.get('overall_passed')}")
    if gate.get("pbo"):
        print(f"    pbo={gate['pbo']:.4f}")
    if gate.get("mean_oos_acc"):
        print(f"    mean_oos_acc={gate['mean_oos_acc']:.4f}")

    # Ensemble weights
    ens = results.get("ensemble", {})
    if ens:
        print(f"\n  Ensemble:")
        print(f"    lstm_weight={ens.get('lstm_weight', 'N/A')}")
        print(f"    xgb_weight={ens.get('xgb_weight', 'N/A')}")

    # Config
    config = results.get("config", {})
    if config:
        print(f"\n  Config:")
        print(f"    n_features={config.get('n_features')}")
        print(f"    target={config.get('target')}")

    # ── Step 4: Predict ──
    print(f"\n[4/4] Running prediction...")
    try:
        pred_start = (today - timedelta(days=200)).isoformat()
        pred = trainer.predict(pred_start, today.isoformat())
        if pred:
            print(f"  signal={pred.signal}")
            print(f"  signal_strength={pred.signal_strength:.4f}")
            print(f"  predicted_returns={pred.predicted_returns:.6f}")
        else:
            print("  Prediction returned None")
    except Exception as e:
        print(f"  Prediction error: {e}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    checks = []

    # Weight check
    if w is not None:
        w_pos = w[w > 0]
        w_ok = abs(w_pos.mean() - 1.0) < 0.02
        checks.append(("Weight mean ~= 1.0", w_ok, f"{w_pos.mean():.4f}"))

    # No-learning check
    if xgb_res:
        nl = xgb_res.get("no_learning")
        checks.append(("XGB no_learning=False", nl is False, str(nl)))

    # Overfitting gap
    if cls_res and cls_train and cls_val:
        gap = abs(cls_train - cls_val)
        checks.append(("Classifier gap < 0.15", gap < 0.15, f"{gap:.4f}"))

    # LSTM direction
    if lstm_res:
        d = lstm_res.get("test_direction_acc", 0)
        checks.append(("LSTM dir_acc > 0.52", d > 0.52, f"{d:.4f}"))

    # Overall
    checks.append(("Quality gate passed", gate.get("overall_passed", False), ""))

    for name, passed, val in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}: {val}")

    all_passed = all(p for _, p, _ in checks)
    print(f"\n  {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
