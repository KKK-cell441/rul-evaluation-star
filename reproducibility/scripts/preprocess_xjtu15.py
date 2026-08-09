# -*- coding: utf-8 -*-
"""Preprocess full XJTU-SY 15 bearings into dbca-net processed format."""
from pathlib import Path
import sys
import time

import numpy as np

ROOT_FW = Path.home() / "Documents" / "机器学习" / "RUL-Prediction-Framework"
ROOT_DATA = ROOT_FW / "data" / "raw" / "XJTU-SY"
sys.path.insert(0, str(ROOT_FW / "scripts"))

from preprocess_public_rul import process_signal, save_processed  # noqa: E402


def load_bearing(bearing_dir):
    files = sorted(bearing_dir.glob("*.csv"))
    rows = []
    for f in files:
        arr = np.loadtxt(f, delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        rows.append(arr[:, :2].mean(axis=1))
    return np.concatenate(rows)


def main():
    feats, cwts, labels = [], [], {k: [] for k in ["linear", "piecewise", "exponential"]}
    n_per = []
    for cond in sorted(ROOT_DATA.iterdir()):
        if not cond.is_dir():
            continue
        for bearing in sorted(cond.iterdir()):
            if not bearing.is_dir():
                continue
            t0 = time.time()
            signal = load_bearing(bearing)
            f, c, lab = process_signal(signal)
            feats.append(f)
            cwts.append(c.astype(np.float16))
            n_per.append(len(f))
            for k in labels:
                labels[k].append(lab[k])
            print(bearing.name, "windows", len(f), "time", round(time.time() - t0, 1))
    save_processed("XJTU-SY15", np.concatenate(feats), np.concatenate(cwts),
                   {k: np.concatenate(v) for k, v in labels.items()}, n_per)


if __name__ == "__main__":
    main()
