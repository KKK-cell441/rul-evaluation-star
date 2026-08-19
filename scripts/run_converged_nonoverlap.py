# -*- coding: utf-8 -*-
"""Converged STAR runner with non-overlapping history sequences.

This runner preserves the previously defined protocol settings:
- same features, labels, models, hyperparameters, and seeds
- LOOCV still leaves the test bearing out of training and validation
- random split still uses 70/15/15 with seeds 42-49

The only protocol change is history sequence construction:
- old: stride=1 sliding sequences
- new: stride=10 non-overlapping sequences
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_converged_rul_study as rc  # noqa: E402
import run_converged_star as rcs  # noqa: E402

HISTORY = 10
STRIDE = 10
BATCH = 1024


def create_nonoverlap_datasets(data_dir, history_length=HISTORY, rul_key="rul_linear"):
    """Create per-bearing datasets whose history sequences do not overlap."""
    data_dir = Path(data_dir)
    features = np.load(data_dir / "features.npy")
    rul = np.load(data_dir / f"{rul_key}.npy")
    n_per = np.load(data_dir / "n_per_bearing.npy")
    datasets = []
    start = 0
    for n in n_per:
        dummy = np.zeros((n, 1, 1, 1), dtype=np.float16)
        ds = rcs.rc.BearingRULDataset(
            features=features[start : start + n],
            cwt_scalograms=dummy,
            rul_targets=rul[start : start + n],
            history_length=history_length,
            stride=STRIDE,
        )
        datasets.append(ds)
        start += n
    return datasets


def nonoverlap_starts(datasets):
    """Return global feature-window starts for non-overlapping sequences."""
    starts = []
    offset = 0
    for ds in datasets:
        n = len(ds.features)
        starts.extend(offset + s for s in range(0, n - HISTORY + 1, STRIDE))
        offset += n
    return np.asarray(starts, dtype=np.int64)


class NonOverlapFlatSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, features, rul, starts):
        self.features = np.asarray(features, dtype=np.float32)
        self.rul = np.asarray(rul, dtype=np.float32)
        self.starts = np.asarray(starts, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        end = start + HISTORY
        dummy = np.zeros((HISTORY, 1, 1, 1), dtype=np.float32)
        return {
            "stat_features": torch.from_numpy(self.features[start:end]).float(),
            "cwt_scalograms": torch.from_numpy(dummy).float(),
            "rul": torch.tensor(self.rul[end - 1], dtype=torch.float32),
        }


def build_random_loaders(datasets, idx, batch_size=BATCH):
    features, targets = rcs.raw_arrays(datasets)
    starts = nonoverlap_starts(datasets)
    n_train, n_val = int(len(idx) * 0.7), int(len(idx) * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    last_train = starts[train_idx] + HISTORY - 1
    mean = features[last_train].mean(axis=0).astype(np.float32)
    std = features[last_train].std(axis=0).astype(np.float32)
    std[std < 1e-12] = 1.0
    norm = (features - mean) / (std + 1e-12)
    ds = NonOverlapFlatSequenceDataset(norm, targets, starts)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def run_random_seed(datasets, model_name, seed, max_epochs):
    rng = np.random.default_rng(seed)
    if model_name in ("LinearRegression", "Constant"):
        _, y = rcs.sequence_arrays(datasets)
        idx = rng.permutation(len(y))
        if model_name == "LinearRegression":
            return rcs.lr_random_seq(datasets, idx)
        return rcs.constant_random_seq(datasets, idx)
    starts = nonoverlap_starts(datasets)
    idx = rng.permutation(len(starts))
    rc.reset_seed()
    train_loader, val_loader, test_loader = build_random_loaders(datasets, idx)
    model = rc.train_model(rc.make_model(model_name), train_loader, val_loader, max_epochs)
    val_pred, val_true = rc.predict_loader(model, val_loader)
    test_pred, test_true = rc.predict_loader(model, test_loader)
    return rc.evaluate_predictions(test_true, test_pred, np.abs(val_true - val_pred))


rc.create_light_datasets = create_nonoverlap_datasets
rc.build_random_loaders = build_random_loaders
rc.run_random_seed = run_random_seed

if __name__ == "__main__":
    rc.main()
