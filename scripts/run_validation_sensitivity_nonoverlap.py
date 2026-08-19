# -*- coding: utf-8 -*-
"""Validation-bearing sensitivity for the stride=10 non-overlapping protocol."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPRO = Path(__file__).resolve().parents[1] / "results_v26"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_converged_nonoverlap as rcn  # noqa: E402
import run_converged_rul_study as rc  # noqa: E402
import run_converged_star as rcs  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "processed" / "PHM2012")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    data_dir = args.data_dir
    epochs = args.epochs
    datasets = rcn.create_nonoverlap_datasets(data_dir, history_length=10, rul_key="rul_linear")
    out = {"model": "TCN", "dataset": "PHM2012", "label": "linear", "protocol": "stride10", "validation_offsets": {}}
    for offset in [1, 2, 3, 4, 5]:
        folds = []
        t0 = time.time()
        for test_idx in range(len(datasets)):
            val_idx = (test_idx + offset) % len(datasets)
            train_loader, val_loader, test_loader = rcs.build_loocv_loaders(datasets, test_idx, val_idx)
            rc.reset_seed()
            model = rc.train_model(rc.make_model("TCN"), train_loader, val_loader, epochs)
            val_pred, val_true = rc.predict_loader(model, val_loader)
            test_pred, test_true = rc.predict_loader(model, test_loader)
            metrics = rc.evaluate_predictions(test_true, test_pred, np.abs(val_true - val_pred))
            metrics["best_epoch"] = int(getattr(model, "_best_epoch", 0))
            folds.append(metrics)
            print(f"offset {offset} fold {test_idx}: R2={folds[-1]['r2']:.4f}", flush=True)
        r2 = [f["r2"] for f in folds]
        out["validation_offsets"][str(offset)] = {
            "r2_mean": float(np.mean(r2)),
            "r2_std": float(np.std(r2, ddof=1)) if len(r2) > 1 else 0.0,
            "r2_min": float(np.min(r2)),
            "r2_max": float(np.max(r2)),
            "rmse_mean": float(np.mean([f["rmse"] for f in folds])),
            "mae_mean": float(np.mean([f["mae"] for f in folds])),
            "folds": folds,
        }
        print(f"offset {offset}: mean {np.mean(r2):.4f} ({time.time()-t0:.0f}s)", flush=True)
    REPRO.mkdir(parents=True, exist_ok=True)
    path = REPRO / "validation_sensitivity_phm_tcn_nonoverlap.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
