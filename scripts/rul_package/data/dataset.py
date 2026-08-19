"""PyTorch Dataset for preprocessed XJTU-SY / IMS bearing data.

Loads precomputed features.npy, cwt_scalograms.npy, and RUL labels.
Uses per-bearing splits (not random) to avoid data leakage.
"""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class BearingRULDataset(Dataset):
    """Dataset for a single bearing's run-to-failure trajectory.

    Loads pre-extracted features + CWT scalograms + RUL labels from disk.
    Each sample is a sequence of L consecutive windows (history_length).
    """

    def __init__(
        self,
        features: np.ndarray,
        cwt_scalograms: np.ndarray,
        rul_targets: np.ndarray,
        history_length: int = 10,
        stride: int = 1,
        stat_mean: np.ndarray | None = None,
        stat_std: np.ndarray | None = None,
    ):
        """
        Parameters
        ----------
        features : (N, 10) pre-extracted statistical features
        cwt_scalograms : (N, 3, 64, 64) pre-computed CWT images
        rul_targets : (N,) RUL labels
        history_length : int, number of windows stacked per sample
        stride : int, step between samples
        stat_mean, stat_std : normalisation params (set after fitting on training set)
        """
        assert len(features) == len(cwt_scalograms) == len(rul_targets)
        self.features = features.astype(np.float32)
        self.cwt_scalograms = cwt_scalograms.astype(np.float32)
        self.rul = rul_targets.astype(np.float32)
        self.history_length = history_length
        self.stride = stride
        self.n_total = len(features)

        # Default: no normalisation (fitted on training set)
        if stat_mean is not None and stat_std is not None:
            self.stat_mean = stat_mean.astype(np.float32)
            self.stat_std = stat_std.astype(np.float32)
        else:
            self.stat_mean = np.zeros(features.shape[1], dtype=np.float32)
            self.stat_std = np.ones(features.shape[1], dtype=np.float32)

        # Valid sequence indices
        self.valid_starts = list(range(0, self.n_total - history_length + 1, stride))

    def set_normalisation(self, mean: np.ndarray, std: np.ndarray):
        self.stat_mean = mean.astype(np.float32)
        self.stat_std = std.astype(np.float32)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> dict:
        start = self.valid_starts[idx]
        end = start + self.history_length

        # Statistical features (L, 10) — normalise
        stat_seq = self.features[start:end].copy()
        stat_seq = (stat_seq - self.stat_mean) / (self.stat_std + 1e-12)

        # CWT scalograms (L, 3, 64, 64) — already in (L, C, H, W) format
        cwt_seq = self.cwt_scalograms[start:end].copy()

        # RUL label (last window in the sequence)
        rul = self.rul[end - 1]

        return {
            "stat_features": torch.from_numpy(stat_seq).float(),
            "cwt_scalograms": torch.from_numpy(cwt_seq).float(),
            "rul": torch.tensor(rul, dtype=torch.float32),
        }


def load_preprocessed_data(data_dir: str) -> dict:
    """Load preprocessed data from a processed dataset directory."""
    data_dir = Path(data_dir)
    data = {
        "features": np.load(data_dir / "features.npy"),
        "cwt": np.load(data_dir / "cwt_scalograms.npy"),
        "rul": np.load(data_dir / "rul_linear.npy"),  # NEW: linear RUL labels
        "n_per_bearing": np.load(data_dir / "n_per_bearing.npy"),
    }
    with open(data_dir / "normalization_stats.json") as f:
        data["norm_stats"] = json.load(f)
    return data


def create_per_bearing_datasets(
    data_dir: str,
    history_length: int = 10,
    stride: int = 1,
    rul_key: str = "rul_linear",
) -> list[BearingRULDataset]:
    """Load data and create one BearingRULDataset per bearing.

    Returns a list of 8 datasets (one per bearing).
    """
    data_dir = Path(data_dir)
    features = np.load(data_dir / "features.npy")
    cwt = np.load(data_dir / "cwt_scalograms.npy")
    rul = np.load(data_dir / f"{rul_key}.npy")
    n_per_bearing = np.load(data_dir / "n_per_bearing.npy")

    datasets = []
    start = 0
    for i, n in enumerate(n_per_bearing):
        ds = BearingRULDataset(
            features=features[start:start+n],
            cwt_scalograms=cwt[start:start+n],
            rul_targets=rul[start:start+n],
            history_length=history_length,
            stride=stride,
        )
        datasets.append(ds)
        start += n

    return datasets


def train_val_test_split_by_bearing(
    datasets: list[BearingRULDataset],
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
) -> tuple[list[BearingRULDataset], list[BearingRULDataset], list[BearingRULDataset]]:
    """Split by bearing index (not random) to prevent data leakage.

    XJTU-SY has 8 bearings. Recommended split:
      Train: bearings 1-5 (indices 0-4)
      Val:   bearing  6   (index 5)
      Test:  bearings 7-8 (indices 6-7)
    """
    train_ds = [datasets[i] for i in train_indices]
    val_ds = [datasets[i] for i in val_indices]
    test_ds = [datasets[i] for i in test_indices]
    return train_ds, val_ds, test_ds


def compute_normalisation_params(
    train_datasets: list[BearingRULDataset],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute dataset-wide Z-score mean/std from training Datasets only."""
    all_feats = np.concatenate([ds.features for ds in train_datasets], axis=0)
    mean = np.mean(all_feats, axis=0).astype(np.float32)
    std = np.std(all_feats, axis=0).astype(np.float32)
    std[std < 1e-12] = 1.0
    return mean, std


def build_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    history_length: int = 10,
    stride: int = 1,
    train_indices: list[int] | None = None,
    val_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
    rul_key: str = "rul_linear",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders with per-bearing split.

    Default split (XJTU-SY, 8 bearings):
      Train: 0-4 (5 bearings, 62.5%)
      Val:   5   (1 bearing,  12.5%)
      Test:  6-7 (2 bearings, 25.0%)
    """
    if train_indices is None:
        train_indices = [0, 1, 2, 3, 4]
    if val_indices is None:
        val_indices = [5]
    if test_indices is None:
        test_indices = [6, 7]

    datasets = create_per_bearing_datasets(data_dir, history_length, stride, rul_key)
    train_ds, val_ds, test_ds = train_val_test_split_by_bearing(
        datasets, train_indices, val_indices, test_indices
    )

    # Fit normalisation on training bearings only (Eq. 4 — prevent data leakage)
    feat_mean, feat_std = compute_normalisation_params(train_ds)
    for ds in train_ds:
        ds.set_normalisation(feat_mean, feat_std)
    for ds in val_ds:
        ds.set_normalisation(feat_mean, feat_std)
    for ds in test_ds:
        ds.set_normalisation(feat_mean, feat_std)

    train_loader = DataLoader(
        ConcatDataset(train_ds), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        ConcatDataset(val_ds), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        ConcatDataset(test_ds), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    n_train = sum(len(ds) for ds in train_ds)
    n_val = sum(len(ds) for ds in val_ds)
    n_test = sum(len(ds) for ds in test_ds)
    print(f"[DataLoader] Per-bearing split: train={train_indices} val={val_indices} test={test_indices}")
    print(f"  Train={n_train} samples, Val={n_val}, Test={n_test}")

    return train_loader, val_loader, test_loader


def build_ims_loader(
    data_dir: str,
    batch_size: int = 64,
    history_length: int = 10,
    rul_key: str = "rul_linear",
    feat_mean: np.ndarray | None = None,
    feat_std: np.ndarray | None = None,
) -> DataLoader:
    """Build a DataLoader for IMS cross-dataset evaluation."""
    datasets = create_per_bearing_datasets(data_dir, history_length, rul_key=rul_key)
    for ds in datasets:
        if feat_mean is not None and feat_std is not None:
            ds.set_normalisation(feat_mean, feat_std)
    loader = DataLoader(
        ConcatDataset(datasets), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    print(f"[DataLoader] IMS: {sum(len(ds) for ds in datasets)} samples")
    return loader
