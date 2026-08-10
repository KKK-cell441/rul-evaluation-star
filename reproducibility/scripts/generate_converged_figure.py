from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parents[2] / "figures"
FILES = {
    "XJTU-SY": "converged_xjtu8_v1.json",
    "PHM2012": "converged_phm_v1.json",
    "IMS": "converged_ims_v1.json",
}
MODELS = ["LinearRegression", "StatLSTM", "TCN", "PatchTST"]
COLORS = ["#4c72b0", "#dd8452", "#55a868", "#8172b3"]


def main():
    data = {}
    for ds, name in FILES.items():
        p = BASE / name
        if p.exists():
            data[ds] = json.loads(p.read_text(encoding="utf-8")).get("datasets", {}).get(ds, {})

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    for ax, ds in zip(axes, FILES):
        labels = data.get(ds, {})
        x = np.arange(len(MODELS))
        width = 0.25
        lin_means = []
        lin_errs = []
        pw_means = []
        pw_errs = []
        rnd_means = []
        rnd_errs = []
        for model in MODELS:
            lin_r2 = labels.get("linear", {}).get(model, {}).get("loocv", {}).get("r2", {})
            pw_r2 = labels.get("piecewise", {}).get(model, {}).get("loocv", {}).get("r2", {})
            rnd_r2 = labels.get("piecewise", {}).get(model, {}).get("random", {}).get("r2", {})
            lin_means.append(lin_r2.get("mean", np.nan))
            lin_errs.append([max(0.0, lin_r2.get("mean", 0) - lin_r2.get("ci95_low", 0)), max(0.0, lin_r2.get("ci95_high", 0) - lin_r2.get("mean", 0))])
            pw_means.append(pw_r2.get("mean", np.nan))
            pw_errs.append([max(0.0, pw_r2.get("mean", 0) - pw_r2.get("ci95_low", 0)), max(0.0, pw_r2.get("ci95_high", 0) - pw_r2.get("mean", 0))])
            rnd_means.append(rnd_r2.get("mean", np.nan))
            rnd_errs.append([max(0.0, rnd_r2.get("mean", 0) - rnd_r2.get("min", 0)), max(0.0, rnd_r2.get("max", 0) - rnd_r2.get("mean", 0))])
        lin_errs = np.asarray(lin_errs).T
        pw_errs = np.asarray(pw_errs).T
        rnd_errs = np.asarray(rnd_errs).T
        ax.bar(x - width, lin_means, width, yerr=lin_errs, label="Linear LOOCV", color=COLORS[0], capsize=3)
        ax.bar(x, pw_means, width, yerr=pw_errs, label="Piecewise LOOCV", color=COLORS[1], capsize=3)
        ax.bar(x + width, rnd_means, width, yerr=rnd_errs, label="Piecewise random", color=COLORS[2], capsize=3)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels(["LR", "LSTM", "TCN", "PatchTST"])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("$R^2$")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "converged_r2_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "converged_r2_comparison.pdf", bbox_inches="tight")
    print("saved", OUT / "converged_r2_comparison.png")


if __name__ == "__main__":
    main()
