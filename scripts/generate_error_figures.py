# -*- coding: utf-8 -*-
"""Regenerate the RMSE/MAE comparison figures from the v26 non-overlap results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1] / "results_v26"
OUT = Path(__file__).resolve().parents[1] / "figures"
FILES = {
    "XJTU-SY": BASE / "converged_xjtu8_nonoverlap_v26.json",
    "PHM2012": BASE / "converged_phm_nonoverlap_v26.json",
    "IMS": BASE / "converged_ims_nonoverlap_v26.json",
}
MODELS = ["LinearRegression", "StatLSTM", "TCN", "PatchTST"]
MODEL_LABELS = ["LR", "LSTM", "TCN", "PatchTST"]


def load():
    out = {}
    for ds, path in FILES.items():
        root = json.loads(path.read_text(encoding="utf-8"))
        out[ds] = root["datasets"][ds]
    return out


def values(data, metric):
    result = {}
    for ds, labels in data.items():
        rows = []
        for model in MODELS:
            rows.append({
                "lin_loo": labels["linear"][model]["loocv"][metric]["mean"],
                "lin_rnd": labels["linear"][model]["random"][metric]["mean"],
                "pw_loo": labels["piecewise"][model]["loocv"][metric]["mean"],
                "pw_rnd": labels["piecewise"][model]["random"][metric]["mean"],
            })
        result[ds] = rows
    return result


def make_figure(metric, filename, title):
    data = values(load(), metric)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), sharey=True)
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    labels = ["Linear LOOCV", "Linear random", "Piecewise LOOCV", "Piecewise random"]
    for ax, ds in zip(axes, FILES):
        rows = data[ds]
        x = np.arange(len(MODELS))
        width = 0.2
        for j, key in enumerate(["lin_loo", "lin_rnd", "pw_loo", "pw_rnd"]):
            vals = [r[key] for r in rows]
            ax.bar(x + (j - 1.5) * width, vals, width, label=labels[j], color=colors[j])
        ax.set_title(ds, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(title, fontsize=12)
    axes[0].legend(frameon=False, fontsize=11, loc="upper right")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    make_figure("rmse", "comparison_rmse.png", "RMSE")
    make_figure("mae", "comparison_mae.png", "MAE")
    print("saved", OUT)


if __name__ == "__main__":
    main()
