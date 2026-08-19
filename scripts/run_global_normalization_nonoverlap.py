# -*- coding: utf-8 -*-
"""Global-normalization sensitivity for the stride=10 non-overlapping protocol."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPRO = Path(__file__).resolve().parents[1] / "results_v26"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_converged_nonoverlap as rcn  # noqa: E402
import run_converged_rul_study as rc  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "processed" / "PHM2012")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    data_dir = args.data_dir
    epochs = args.epochs
    out = {"model": "TCN", "dataset": "PHM2012", "protocol": "stride10", "labels": {}}
    for key, display in [("rul_linear_global", "linear global"), ("rul_piecewise_global", "piecewise global")]:
        datasets = rcn.create_nonoverlap_datasets(data_dir, history_length=10, rul_key=key)
        loocv = rc.run_loocv(datasets, "TCN", epochs)
        reps = []
        for seed in [42, 43, 44, 45, 46, 47, 48, 49]:
            rep = rc.run_random_seed(datasets, "TCN", seed, epochs)
            rep["seed"] = seed
            reps.append(rep)
            print(display, "seed", seed, "R2", round(rep["r2"], 4), flush=True)
        r2 = [r["r2"] for r in reps]
        out["labels"][display] = {
            "loocv": loocv,
            "random_mean": float(np.mean(r2)),
            "random_min": float(np.min(r2)),
            "random_max": float(np.max(r2)),
            "random_reps": reps,
        }
        print(display, "LOOCV", loocv["r2"], "random", [round(x, 4) for x in r2], flush=True)
    REPRO.mkdir(parents=True, exist_ok=True)
    path = REPRO / "global_normalization_phm_tcn_nonoverlap.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
