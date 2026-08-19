# -*- coding: utf-8 -*-
"""Split sensitivity with non-overlapping history sequences (stride=10)."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_converged_rul_study as rc  # noqa: E402
import run_converged_nonoverlap as rcn  # noqa: E402
from run_converged_nonoverlap import NonOverlapFlatSequenceDataset, nonoverlap_starts  # noqa: E402


def build_loaders(datasets, mode, seed, train_ratio=0.7, val_ratio=0.15):
    features, targets = rcn.rcs.raw_arrays(datasets)
    starts = nonoverlap_starts(datasets)
    n_seq = len(starts)
    if mode.startswith("random"):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n_seq)
        n_train = int(n_seq * train_ratio)
        n_val = int(n_seq * val_ratio)
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
    elif mode.startswith("chronological"):
        train_idx, val_idx, test_idx = [], [], []
        start = 0
        for ds in datasets:
            n_bear = len(ds.features)
            n_seq_bear = max(0, (n_bear - rcn.HISTORY) // rcn.STRIDE + 1)
            n_tr = int(n_seq_bear * train_ratio)
            n_va = int(n_seq_bear * val_ratio)
            train_idx.extend(range(start, start + n_tr))
            val_idx.extend(range(start + n_tr, start + n_tr + n_va))
            test_idx.extend(range(start + n_tr + n_va, start + n_seq_bear))
            start += n_seq_bear
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
    else:
        raise ValueError(mode)

    last_train = starts[train_idx] + rcn.HISTORY - 1
    mean = features[last_train].mean(axis=0).astype(np.float32)
    std = features[last_train].std(axis=0).astype(np.float32)
    std[std < 1e-12] = 1.0
    norm = (features - mean) / (std + 1e-12)
    ds = NonOverlapFlatSequenceDataset(norm, targets, starts)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=rcn.BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=rcn.BATCH, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=rcn.BATCH, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def run_one(datasets, mode, seed, train_ratio, val_ratio, epochs):
    train_loader, val_loader, test_loader = build_loaders(
        datasets, mode, seed, train_ratio, val_ratio
    )
    rc.reset_seed()
    model = rc.train_model(rc.make_model("TCN"), train_loader, val_loader, epochs)
    val_pred, val_true = rc.predict_loader(model, val_loader)
    test_pred, test_true = rc.predict_loader(model, test_loader)
    metrics = rc.evaluate_predictions(test_true, test_pred, np.abs(val_true - val_pred))
    metrics["best_epoch"] = int(getattr(model, "_best_epoch", 0))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    datasets = rcn.create_nonoverlap_datasets(args.data_dir, rul_key="rul_piecewise")
    ds_name = Path(args.data_dir).name
    results = {"config": vars(args), "dataset": ds_name, "label": "piecewise", "model": "TCN", "splits": {}}

    for mode, train_ratio, val_ratio in [
        ("random_60", 0.6, 0.2),
        ("random_80", 0.8, 0.1),
        ("chronological_70", 0.7, 0.15),
    ]:
        seeds = [42, 43, 44] if mode.startswith("random") else [42]
        reps = []
        for seed in seeds:
            t0 = time.time()
            rep = run_one(datasets, mode, seed, train_ratio, val_ratio, args.epochs)
            rep["seed"] = seed
            reps.append(rep)
            print(f"{ds_name}/{mode}/seed{seed}: R2={rep['r2']:.4f} best={rep['best_epoch']} ({time.time()-t0:.0f}s)", flush=True)
        r2 = [r["r2"] for r in reps]
        results["splits"][mode] = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": round(1.0 - train_ratio - val_ratio, 4),
            "r2_mean": float(np.mean(r2)),
            "r2_min": float(np.min(r2)),
            "r2_max": float(np.max(r2)),
            "best_epoch_mean": float(np.mean([r["best_epoch"] for r in reps])),
            "reps": reps,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
