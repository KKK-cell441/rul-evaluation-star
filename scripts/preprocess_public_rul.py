# -*- coding: utf-8 -*-
"""Full preprocessing for PHM2012 and IMS into the package processed format."""
import argparse
from pathlib import Path
import json
import sys
import time

import numpy as np
import scipy.io

ROOT_DATA = Path("data") / "raw"
PROCESSED = Path("data") / "processed"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rul_package.data.preprocessing import extract_features_from_window  # noqa: E402
from rul_package.utils.rul_labels import make_rul_labels  # noqa: E402

FS = 25_600
WINDOW = 2_560
CWT_SIZE = 64


def windows_from_signal(signal):
    n = len(signal)
    starts = list(range(0, n - WINDOW + 1, WINDOW))
    return [signal[s : s + WINDOW] for s in starts if len(signal[s : s + WINDOW]) == WINDOW]


def process_signal(signal):
    wins = windows_from_signal(signal)
    features, cwts = [], []
    for w in wins:
        f, c = extract_features_from_window(w, fs=FS, cwt_size=CWT_SIZE)
        features.append(f)
        cwts.append(c)
    features = np.asarray(features, dtype=np.float32)
    cwts = np.asarray(cwts, dtype=np.float32)
    labels = {name: make_rul_labels(len(features), strategy=name).astype(np.float32)
              for name in ["linear", "piecewise", "exponential"]}
    return features, cwts, labels


def save_processed(dataset_name, all_features, all_cwts, all_labels, n_per_bearing):
    out = PROCESSED / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "features.npy", all_features)
    np.save(out / "cwt_scalograms.npy", all_cwts)
    np.save(out / "n_per_bearing.npy", np.array(n_per_bearing, dtype=np.int32))
    for name, arr in all_labels.items():
        np.save(out / f"rul_{name}.npy", arr)
    norm = {
        "feature_mean": all_features.mean(axis=0).tolist(),
        "feature_std": all_features.std(axis=0).tolist(),
        "history_length": 10,
        "cwt_width": CWT_SIZE,
        "cwt_height": CWT_SIZE,
        "cwt_wavelet": "morl",
        "window_size": WINDOW,
        "stride": WINDOW,
        "sampling_rate": FS,
    }
    (out / "normalization_stats.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")
    print(f"Saved {dataset_name}: windows={len(all_features)}, n_per_bearing={n_per_bearing}")


def load_phm_bearing(bearing_dir):
    files = sorted(bearing_dir.glob("acc_*.csv"))
    rows = []
    for f in files:
        arr = np.loadtxt(f, delimiter=",")
        rows.append(arr[:, [4, 5]].mean(axis=1))
    return np.concatenate(rows)


def load_ims_test(test_dir, channel=0):
    files = sorted(p for p in test_dir.rglob("*") if p.is_file())
    rows = []
    for f in files:
        arr = np.loadtxt(f, delimiter="\t")
        rows.append(arr[:, channel])
    return np.concatenate(rows)


def preprocess_phm():
    root = ROOT_DATA / "PHM2012" / "Learning_set"
    bearings = sorted([p for p in root.iterdir() if p.is_dir()])
    feats, cwts, labels = [], [], {k: [] for k in ["linear", "piecewise", "exponential"]}
    n_per = []
    for b in bearings:
        t0 = time.time()
        signal = load_phm_bearing(b)
        f, c, lab = process_signal(signal)
        feats.append(f)
        cwts.append(c)
        n_per.append(len(f))
        for k in labels:
            labels[k].append(lab[k])
        print(b.name, "windows", len(f), "time", round(time.time() - t0, 1))
    save_processed("PHM2012", np.concatenate(feats), np.concatenate(cwts),
                   {k: np.concatenate(v) for k, v in labels.items()}, n_per)


def preprocess_ims():
    root = ROOT_DATA / "IMS"
    test_dirs = ["1st_test", "2nd_test", "4th_test"]
    feats, cwts, labels = [], [], {k: [] for k in ["linear", "piecewise", "exponential"]}
    n_per = []
    for name in test_dirs:
        t0 = time.time()
        signal = load_ims_test(root / name, channel=0)
        f, c, lab = process_signal(signal)
        feats.append(f)
        cwts.append(c)
        n_per.append(len(f))
        for k in labels:
            labels[k].append(lab[k])
        print(name, "windows", len(f), "time", round(time.time() - t0, 1))
    save_processed("IMS", np.concatenate(feats), np.concatenate(cwts),
                   {k: np.concatenate(v) for k, v in labels.items()}, n_per)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT_DATA, help="raw data root containing PHM2012/ and IMS/")
    parser.add_argument("--output", type=Path, default=PROCESSED, help="processed data output root")
    args = parser.parse_args()
    ROOT_DATA = args.input
    PROCESSED = args.output
    preprocess_phm()
    preprocess_ims()
