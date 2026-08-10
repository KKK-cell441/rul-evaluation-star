# -*- coding: utf-8 -*-
"""Converged STAR protocol for the RUL labeling critique.

Protocol:
- datasets: XJTU-SY (8 bearings), PHM2012, IMS
- labels: linear and piecewise (primary comparison)
- models: LinearRegression, StatLSTM, TCN, PatchTST, Constant
- LOOCV per bearing and random time-step split
- max epochs 100, cosine annealing, early stopping patience 30
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from torch.utils.data import ConcatDataset, DataLoader, Subset

ROOT_FW = Path.home() / "Documents" / "\u673a\u5668\u5b66\u4e60" / "RUL-Prediction-Framework"
ROOT_1111 = Path.home() / "Documents" / "1111" / "dbca-net"
sys.path.insert(0, str(ROOT_1111))
sys.path.insert(0, str(ROOT_FW))

from data.dataset import BearingRULDataset  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402
from importlib import util as _ilu  # noqa: E402


def _load_module(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tcn_mod = _load_module("tcn_model", ROOT_FW / "src" / "models" / "tcn_model.py")
_patch_mod = _load_module("patchtst_model", ROOT_FW / "src" / "models" / "patchtst_model.py")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
ALPHA = 0.10
BATCH_SIZE = 512


class StatLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.head(out).squeeze(-1)


def create_light_datasets(data_dir, history_length=10, rul_key="rul_linear"):
    data_dir = Path(data_dir)
    features = np.load(data_dir / "features.npy")
    rul = np.load(data_dir / f"{rul_key}.npy")
    n_per = np.load(data_dir / "n_per_bearing.npy")
    datasets = []
    start = 0
    for n in n_per:
        dummy = np.zeros((n, 1, 1, 1), dtype=np.float16)
        ds = BearingRULDataset(
            features=features[start : start + n],
            cwt_scalograms=dummy,
            rul_targets=rul[start : start + n],
            history_length=history_length,
            stride=1,
        )
        datasets.append(ds)
        start += n
    return datasets


def arrays_from_datasets(datasets):
    features = np.concatenate([ds.features for ds in datasets], axis=0)
    targets = np.concatenate([ds.rul for ds in datasets], axis=0)
    sizes = [ds.features.shape[0] for ds in datasets]
    starts = np.cumsum([0] + sizes)
    return features, targets, starts, sizes


def build_loocv_loaders(datasets, test_idx, val_idx, batch_size=BATCH_SIZE):
    train_indices = [i for i in range(len(datasets)) if i not in (test_idx, val_idx)]
    train_ds = [datasets[i] for i in train_indices]
    val_ds = [datasets[val_idx]]
    test_ds = [datasets[test_idx]]
    feat_mean, feat_std = compute_normalisation_params(train_ds)
    for ds in train_ds + val_ds + test_ds:
        ds.set_normalisation(feat_mean, feat_std)
    train_loader = DataLoader(ConcatDataset(train_ds), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ConcatDataset(val_ds), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ConcatDataset(test_ds), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def build_random_loaders(datasets, idx, batch_size=BATCH_SIZE):
    all_ds = ConcatDataset(datasets)
    n_train, n_val = int(len(all_ds) * 0.7), int(len(all_ds) * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    valid_features = np.concatenate([ds.features[ds.valid_starts] for ds in datasets])
    feat_mean = valid_features[train_idx].mean(axis=0).astype(np.float32)
    feat_std = valid_features[train_idx].std(axis=0).astype(np.float32)
    feat_std[feat_std < 1e-12] = 1.0
    for ds in datasets:
        ds.set_normalisation(feat_mean, feat_std)
    train_loader = DataLoader(Subset(all_ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(all_ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(all_ds, test_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def compute_normalisation_params(train_datasets):
    all_feats = np.concatenate([ds.features for ds in train_datasets], axis=0)
    mean = np.mean(all_feats, axis=0).astype(np.float32)
    std = np.std(all_feats, axis=0).astype(np.float32)
    std[std < 1e-12] = 1.0
    return mean, std


def make_model(model_name):
    if model_name == "StatLSTM":
        return StatLSTM(input_size=10, hidden_size=128, num_layers=2)
    if model_name == "TCN":
        return _tcn_mod.TCNRULModel(
            input_dim=10, seq_len=10, num_channels=[32, 64, 128],
            kernel_size=3, dropout=0.2, hidden_dim=64,
        )
    if model_name == "PatchTST":
        return _patch_mod.PatchTSTRULModel(
            input_dim=10, seq_len=10, patch_len=4, stride=2,
            d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1,
        )
    raise ValueError(model_name)


def train_model(model, train_loader, val_loader, max_epochs=100, patience=30):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history = {"train": [], "val": []}
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch in train_loader:
            stat = batch["stat_features"].to(DEVICE)
            rul = batch["rul"].to(DEVICE)
            optimizer.zero_grad()
            out = model(stat)
            loss = criterion(out, rul)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(rul)
            train_n += len(rul)
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                stat = batch["stat_features"].to(DEVICE)
                rul = batch["rul"].to(DEVICE)
                out = model(stat)
                val_loss_sum += criterion(out, rul).item() * len(rul)
                val_n += len(rul)
        train_loss = train_loss_sum / max(train_n, 1)
        val_loss = val_loss_sum / max(val_n, 1)
        history["train"].append(float(train_loss))
        history["val"].append(float(val_loss))
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
        scheduler.step()
    if best_state is not None:
        model.load_state_dict(best_state)
    model._best_epoch = best_epoch
    model._history = history
    return model


def predict_loader(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            stat = batch["stat_features"].to(DEVICE)
            rul = batch["rul"].to(DEVICE)
            out = model(stat)
            preds.append(out.cpu().numpy())
            targets.append(rul.cpu().numpy())
    return np.clip(np.concatenate(preds), 0.0, 1.0), np.concatenate(targets)


def evaluate_predictions(y_true, y_pred, cal_residuals=None, alpha=ALPHA):
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)
    m = compute_metrics(y_true, y_pred)
    out = {"rmse": float(m.rmse), "mae": float(m.mae), "r2": float(m.r2)}
    if cal_residuals is not None and len(cal_residuals):
        q = np.quantile(cal_residuals, 1 - alpha) * ((len(cal_residuals) + 1) / len(cal_residuals))
        lower = y_pred - q
        upper = y_pred + q
        out["picp"] = float(np.mean((y_true >= lower) & (y_true <= upper)))
        out["mpiw"] = float(np.mean(upper - lower) / max(float(np.ptp(y_true)), 1e-9))
        out["interval_width"] = float(np.mean(upper - lower))
    else:
        out["picp"] = None
        out["mpiw"] = None
        out["interval_width"] = None
    return out


def summarize_folds(folds):
    keys = ["r2", "rmse", "mae", "picp", "mpiw"]
    out = {}
    for k in keys:
        vals = [f[k] for f in folds if f[k] is not None]
        if not vals:
            out[k] = None
            continue
        arr = np.asarray(vals, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        n = len(arr)
        try:
            from scipy.stats import t
            ci = t.interval(0.95, max(n - 1, 1), loc=mean, scale=std / np.sqrt(n))
        except Exception:
            ci = (mean - 1.96 * std / np.sqrt(n), mean + 1.96 * std / np.sqrt(n))
        out[k] = {
            "mean": mean,
            "std": std,
            "ci95_low": float(ci[0]),
            "ci95_high": float(ci[1]),
        }
    out["folds"] = folds
    return out


def normalize_rows(features, train_idx):
    mean = features[train_idx].mean(axis=0).astype(np.float32)
    std = features[train_idx].std(axis=0).astype(np.float32)
    std[std < 1e-12] = 1.0
    return (features - mean) / (std + 1e-12)


def lr_loocv(datasets, test_idx, val_idx):
    X, y, starts, _ = arrays_from_datasets(datasets)
    train_idx = [i for i in range(len(datasets)) if i not in (test_idx, val_idx)]
    tr_i = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in train_idx])
    va_i = np.arange(starts[val_idx], starts[val_idx + 1])
    te_i = np.arange(starts[test_idx], starts[test_idx + 1])
    Xn = normalize_rows(X, tr_i)
    model = LinearRegression().fit(Xn[tr_i], y[tr_i])
    val_pred = model.predict(Xn[va_i])
    test_pred = model.predict(Xn[te_i])
    cal_res = np.abs(y[va_i] - np.clip(val_pred, 0.0, 1.0))
    return evaluate_predictions(y[te_i], test_pred, cal_res)


def constant_loocv(datasets, test_idx, val_idx):
    _, y, starts, _ = arrays_from_datasets(datasets)
    train_idx = [i for i in range(len(datasets)) if i not in (test_idx, val_idx)]
    tr_i = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in train_idx])
    va_i = np.arange(starts[val_idx], starts[val_idx + 1])
    te_i = np.arange(starts[test_idx], starts[test_idx + 1])
    c = np.mean(y[tr_i])
    cal_res = np.abs(y[va_i] - c)
    return evaluate_predictions(y[te_i], np.full_like(y[te_i], c), cal_res)


def lr_random(datasets, idx):
    X, y, _, _ = arrays_from_datasets(datasets)
    n_train, n_val = int(len(y) * 0.7), int(len(y) * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    Xn = normalize_rows(X, train_idx)
    model = LinearRegression().fit(Xn[train_idx], y[train_idx])
    val_pred = model.predict(Xn[val_idx])
    test_pred = model.predict(Xn[test_idx])
    cal_res = np.abs(y[val_idx] - np.clip(val_pred, 0.0, 1.0))
    return evaluate_predictions(y[test_idx], test_pred, cal_res)


def constant_random(datasets, idx):
    _, y, _, _ = arrays_from_datasets(datasets)
    n_train, n_val = int(len(y) * 0.7), int(len(y) * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    c = np.mean(y[train_idx])
    cal_res = np.abs(y[val_idx] - c)
    return evaluate_predictions(y[test_idx], np.full_like(y[test_idx], c), cal_res)


def run_loocv(datasets, model_name, max_epochs):
    folds = []
    best_epochs = []
    for test_idx in range(len(datasets)):
        val_idx = (test_idx + 1) % len(datasets)
        if model_name == "LinearRegression":
            folds.append(lr_loocv(datasets, test_idx, val_idx))
            continue
        if model_name == "Constant":
            folds.append(constant_loocv(datasets, test_idx, val_idx))
            continue
        train_loader, val_loader, test_loader = build_loocv_loaders(datasets, test_idx, val_idx)
        model = train_model(make_model(model_name), train_loader, val_loader, max_epochs)
        best_epochs.append(int(getattr(model, "_best_epoch", 0)))
        val_pred, val_true = predict_loader(model, val_loader)
        test_pred, test_true = predict_loader(model, test_loader)
        folds.append(evaluate_predictions(test_true, test_pred, np.abs(val_true - val_pred)))
    out = summarize_folds(folds)
    if best_epochs:
        out["best_epoch"] = {"mean": float(np.mean(best_epochs)), "std": float(np.std(best_epochs))}
    return out


def run_random_seed(datasets, model_name, seed, max_epochs):
    rng = np.random.default_rng(seed)
    if model_name in ("LinearRegression", "Constant"):
        _, y, _, _ = arrays_from_datasets(datasets)
        idx = rng.permutation(len(y))
        if model_name == "LinearRegression":
            return lr_random(datasets, idx)
        return constant_random(datasets, idx)
    total = sum(len(ds) for ds in datasets)
    idx = rng.permutation(total)
    train_loader, val_loader, test_loader = build_random_loaders(datasets, idx)
    model = train_model(make_model(model_name), train_loader, val_loader, max_epochs)
    val_pred, val_true = predict_loader(model, val_loader)
    test_pred, test_true = predict_loader(model, test_loader)
    return evaluate_predictions(test_true, test_pred, np.abs(val_true - val_pred))


def run_random(datasets, model_name, max_epochs, random_seeds=(42, 43, 44)):
    reps = []
    best_epochs = []
    for seed in random_seeds:
        t0 = time.time()
        rep = run_random_seed(datasets, model_name, seed, max_epochs)
        reps.append(rep)
        print(f"  random seed {seed}: R2={rep['r2']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    keys = ["r2", "rmse", "mae", "picp", "mpiw"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in reps if r[k] is not None]
        if not vals:
            summary[k] = None
            continue
        arr = np.asarray(vals, dtype=np.float64)
        summary[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    summary["reps"] = reps
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--models", nargs="+", default=["Constant", "LinearRegression", "StatLSTM", "TCN", "PatchTST"])
    parser.add_argument("--labels", nargs="+", default=["linear", "piecewise"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        results = {"config": vars(args), "datasets": {}}

    ds_name = Path(args.data_dir).name
    ds_results = results["datasets"].setdefault(ds_name, {})
    for label in args.labels:
        label_results = ds_results.setdefault(label, {})
        datasets = create_light_datasets(args.data_dir, history_length=10, rul_key=f"rul_{label}")
        for model_name in args.models:
            existing = label_results.get(model_name)
            if existing is not None and existing.get("random") is not None:
                print(f"skip {label}/{model_name}", flush=True)
                continue
            t0 = time.time()
            loocv = existing.get("loocv") if existing is not None else None
            if loocv is None:
                loocv = run_loocv(datasets, model_name, args.epochs)
                label_results[model_name] = {"loocv": loocv, "random": None}
                out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
                print(f"{ds_name}/{label}/{model_name}: LOOCV saved R2={loocv['r2']['mean']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            random = run_random(datasets, model_name, args.epochs, tuple(args.random_seeds))
            label_results[model_name] = {"loocv": loocv, "random": random}
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"{ds_name}/{label}/{model_name}: Random R2={random['r2']['mean']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved", out_path)


if __name__ == "__main__":
    main()
