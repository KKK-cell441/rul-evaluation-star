# -*- coding: utf-8 -*-
"""Converged STAR runner for RUL labeling critique.

This wrapper avoids a PyTorch Subset-of-ConcatDataset compatibility issue in
the base runner by using a flat sequence dataset for random time-step splits.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_converged_rul_study as rc  # noqa: E402

HISTORY = 10
BATCH = 1024


class FlatSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, features, rul, history_length=HISTORY):
        self.features = np.asarray(features, dtype=np.float32)
        self.rul = np.asarray(rul, dtype=np.float32)
        self.history_length = history_length
        self.n_sequences = max(0, len(self.rul) - history_length + 1)

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx
        end = idx + self.history_length
        dummy = np.zeros((self.history_length, 1, 1, 1), dtype=np.float32)
        return {
            "stat_features": torch.from_numpy(self.features[start:end]).float(),
            "cwt_scalograms": torch.from_numpy(dummy).float(),
            "rul": torch.tensor(self.rul[end - 1], dtype=torch.float32),
        }


def raw_arrays(datasets):
    features = np.concatenate([ds.features for ds in datasets], axis=0)
    targets = np.concatenate([ds.rul for ds in datasets], axis=0)
    return features, targets


def sequence_arrays(datasets):
    features_all = []
    targets_all = []
    for ds in datasets:
        n = len(ds.features)
        features_all.append(ds.features[HISTORY - 1 : n])
        targets_all.append(ds.rul[HISTORY - 1 : n])
    return np.concatenate(features_all), np.concatenate(targets_all)


def build_loocv_loaders(datasets, test_idx, val_idx, batch_size=BATCH):
    train_indices = [i for i in range(len(datasets)) if i not in (test_idx, val_idx)]
    train_ds = [datasets[i] for i in train_indices]
    val_ds = [datasets[val_idx]]
    test_ds = [datasets[test_idx]]
    feat_mean, feat_std = rc.compute_normalisation_params(train_ds)
    for ds in train_ds + val_ds + test_ds:
        ds.set_normalisation(feat_mean, feat_std)
    train_loader = DataLoader(ConcatDataset(train_ds), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ConcatDataset(val_ds), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ConcatDataset(test_ds), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def build_random_loaders(datasets, idx, batch_size=BATCH):
    features, targets = raw_arrays(datasets)
    n_train, n_val = int(len(idx) * 0.7), int(len(idx) * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    last_train_idx = train_idx + HISTORY - 1
    mean = features[last_train_idx].mean(axis=0).astype(np.float32)
    std = features[last_train_idx].std(axis=0).astype(np.float32)
    std[std < 1e-12] = 1.0
    norm = (features - mean) / (std + 1e-12)
    ds = FlatSequenceDataset(norm, targets, history_length=HISTORY)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def lr_random_seq(datasets, idx):
    X, y = sequence_arrays(datasets)
    n_train, n_val = int(len(y) * 0.7), int(len(y) * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    mean = X[train_idx].mean(axis=0).astype(np.float32)
    std = X[train_idx].std(axis=0).astype(np.float32)
    std[std < 1e-12] = 1.0
    Xn = (X - mean) / (std + 1e-12)
    model = rc.LinearRegression().fit(Xn[train_idx], y[train_idx])
    val_pred = model.predict(Xn[val_idx])
    test_pred = model.predict(Xn[test_idx])
    cal_res = np.abs(y[val_idx] - np.clip(val_pred, 0.0, 1.0))
    return rc.evaluate_predictions(y[test_idx], test_pred, cal_res)


def constant_random_seq(datasets, idx):
    _, y = sequence_arrays(datasets)
    n_train, n_val = int(len(y) * 0.7), int(len(y) * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    c = np.mean(y[train_idx])
    cal_res = np.abs(y[val_idx] - c)
    return rc.evaluate_predictions(y[test_idx], np.full_like(y[test_idx], c), cal_res)


def run_random_seed(datasets, model_name, seed, max_epochs):
    _, y = sequence_arrays(datasets)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    if model_name == "LinearRegression":
        return lr_random_seq(datasets, idx)
    if model_name == "Constant":
        return constant_random_seq(datasets, idx)
    train_loader, val_loader, test_loader = build_random_loaders(datasets, idx)
    model = rc.train_model(rc.make_model(model_name), train_loader, val_loader, max_epochs)
    val_pred, val_true = rc.predict_loader(model, val_loader)
    test_pred, test_true = rc.predict_loader(model, test_loader)
    return rc.evaluate_predictions(test_true, test_pred, np.abs(val_true - val_pred))


rc.build_loocv_loaders = build_loocv_loaders
rc.build_random_loaders = build_random_loaders
rc.run_random_seed = run_random_seed

if __name__ == "__main__":
    rc.main()
