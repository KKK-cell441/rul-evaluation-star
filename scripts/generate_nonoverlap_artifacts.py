# -*- coding: utf-8 -*-
"""Generate non-overlap tables, appendix, stats, and figure from JSON results."""

import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1] / "results_v26"
OUT_TABLES = Path(__file__).resolve().parents[1] / "tables_nonoverlap"
OUT_STATS = BASE / "nonoverlap_stats"
OUT_FIGURES = Path(__file__).resolve().parents[1] / "figures"

MAIN_FILES = {
    "XJTU-SY": "converged_xjtu8_nonoverlap_v26.json",
    "PHM2012": "converged_phm_nonoverlap_v26.json",
    "IMS": "converged_ims_nonoverlap_v26.json",
}
SENS_FILES = {
    "XJTU-SY": "split_sensitivity_xjtu8_nonoverlap_v26.json",
    "PHM2012": "split_sensitivity_phm_nonoverlap_v26.json",
    "IMS": "split_sensitivity_ims_nonoverlap_v26.json",
}
MODELS = ["LinearRegression", "StatLSTM", "TCN", "PatchTST"]
DEEP_MODELS = ["StatLSTM", "TCN", "PatchTST"]
MODEL_LABELS = {"LinearRegression": "Linear Reg.", "StatLSTM": "LSTM", "TCN": "TCN", "PatchTST": "PatchTST"}


def load_main():
    out = {}
    for ds, name in MAIN_FILES.items():
        root = json.loads((BASE / name).read_text(encoding="utf-8"))
        out[ds] = root["datasets"][ds]
    return out


def load_sens():
    out = {}
    for ds, name in SENS_FILES.items():
        out[ds] = json.loads((BASE / name).read_text(encoding="utf-8"))
    return out


def fmt(x, digits=3):
    if x is None:
        return "--"
    return f"{x:.{digits}f}"


def ci_str(v):
    return f"{fmt(v['mean'])} [{fmt(v['ci95_low'])}, {fmt(v['ci95_high'])}]"


def random_str(v):
    return f"{fmt(v['mean'])} [{fmt(v['min'])}, {fmt(v['max'])}]"


def pairs(a, b):
    above = sum(1 for x in a for y in b if x > y)
    below = sum(1 for x in a for y in b if x < y)
    total = len(a) * len(b)
    return above, below, total


def cliff(a, b):
    above, below, total = pairs(a, b)
    return (above - below) / total if total else None


def pairwise_above(a, b):
    above, _, total = pairs(a, b)
    return above / total if total else None


def permutation_p(a, b):
    """One-sided exact permutation p for random-split values exceeding LOOCV folds."""
    all_vals = list(a) + list(b)
    n_a, n_b = len(a), len(b)
    total_subsets = 0
    count_ge = 0
    observed = pairwise_above(a, b)
    for combo in itertools.combinations(range(len(all_vals)), n_b):
        bmask = set(combo)
        aa = [all_vals[i] for i in range(len(all_vals)) if i not in bmask]
        bb = [all_vals[i] for i in bmask]
        pv = pairwise_above(aa, bb)
        total_subsets += 1
        if pv >= observed:
            count_ge += 1
    return count_ge / total_subsets


def write_text(path, text):
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    (OUT_TABLES / path).write_text(text, encoding="utf-8")


def build_table1(data):
    lines = []
    lines.append(r"\begin{table*}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Non-overlap protocol comparison (stride=10). LOOCV $R^2$ values are mean [95\% CI]; random values are mean [min--max] across eight seeds (42--49). $\delta_{\mathrm{label}}$ compares piecewise LOOCV with linear LOOCV folds; $\delta_{\mathrm{protocol}}$ compares piecewise random runs with linear LOOCV folds.}")
    lines.append(r"\label{tab:main_nonoverlap}")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Model & Linear LOOCV & Piecewise LOOCV & $\Delta$ label & Piecewise random & $\Delta$ protocol & $\delta_{\mathrm{label}}$ & $\delta_{\mathrm{protocol}}$ \\")
    lines.append(r"\midrule")
    for ds in MAIN_FILES:
        labels = data[ds]
        lines.append(r"\midrule")
        lines.append(rf"{ds} & \multicolumn{{8}}{{c}}{{}} \\")
        lines.append(r"\midrule")
        for model in MODELS:
            lin = labels["linear"][model]
            pw = labels["piecewise"][model]
            lin_loo = lin["loocv"]["r2"]
            pw_loo = pw["loocv"]["r2"]
            pw_rnd = pw["random"]["r2"]
            lin_folds = [f["r2"] for f in lin["loocv"]["folds"]]
            pw_folds = [f["r2"] for f in pw["loocv"]["folds"]]
            rnd_reps = [r["r2"] for r in pw["random"]["reps"]]
            delta_label = pw_loo["mean"] - lin_loo["mean"]
            delta_protocol = pw_rnd["mean"] - lin_loo["mean"]
            d_label = cliff(pw_folds, lin_folds)
            d_protocol = cliff(rnd_reps, lin_folds)
            lines.append(rf"{MODEL_LABELS[model]} & {ci_str(lin_loo)} & {ci_str(pw_loo)} & {fmt(delta_label)} & {random_str(pw_rnd)} & {fmt(delta_protocol)} & {fmt(d_label)} & {fmt(d_protocol)} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def build_table2(data):
    lines = []
    lines.append(r"\begin{table*}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Complete $2\times2$ non-overlap protocol comparison. Values are mean $R^2$; LOOCV values are means across held-out bearings and random values are means across eight seeds (42--49). Interaction is the descriptive difference-in-differences term defined in Section 4.4.}")
    lines.append(r"\label{tab:factorial_nonoverlap}")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Model & Linear LOOCV & Linear random & Piecewise LOOCV & Piecewise random & Interaction \\")
    lines.append(r"\midrule")
    for ds in MAIN_FILES:
        labels = data[ds]
        lines.append(r"\midrule")
        for model in MODELS:
            lin_loo = labels["linear"][model]["loocv"]["r2"]["mean"]
            lin_rnd = labels["linear"][model]["random"]["r2"]
            pw_loo = labels["piecewise"][model]["loocv"]["r2"]["mean"]
            lin_rnd_mean = labels["linear"][model]["random"]["r2"]["mean"]
            pw_loo_mean = labels["piecewise"][model]["loocv"]["r2"]["mean"]
            pw_rnd = labels["piecewise"][model]["random"]["r2"]
            interaction = (pw_rnd["mean"] - pw_loo_mean) - (lin_rnd_mean - lin_loo)
            lines.append(rf"{ds} & {MODEL_LABELS[model]} & {fmt(lin_loo)} & {random_str(lin_rnd)} & {fmt(pw_loo)} & {random_str(pw_rnd)} & {fmt(interaction)} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def build_table3(data):
    lines = []
    lines.append(r"\begin{table*}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Absolute error metrics for the primary non-overlap protocol comparison. LOOCV values are means across held-out bearings; random-split values are means across eight seeds (42--49).}")
    lines.append(r"\label{tab:errors_nonoverlap}")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Model & $R^2_{\mathrm{lin,LOOCV}}$ & $\mathrm{RMSE}_{\mathrm{lin,LOOCV}}$ & $\mathrm{MAE}_{\mathrm{lin,LOOCV}}$ & $R^2_{\mathrm{pw,random}}$ & $\mathrm{RMSE}_{\mathrm{pw,random}}$ & $\mathrm{MAE}_{\mathrm{pw,random}}$ \\")
    lines.append(r"\midrule")
    for ds in MAIN_FILES:
        labels = data[ds]
        lines.append(r"\midrule")
        for model in MODELS:
            lin = labels["linear"][model]
            pw = labels["piecewise"][model]
            lines.append(rf"{ds} & {MODEL_LABELS[model]} & {fmt(lin['loocv']['r2']['mean'])} & {fmt(lin['loocv']['rmse']['mean'])} & {fmt(lin['loocv']['mae']['mean'])} & {fmt(pw['random']['r2']['mean'])} & {fmt(pw['random']['rmse']['mean'])} & {fmt(pw['random']['mae']['mean'])} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def build_table4(sens):
    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{TCN split sensitivity with non-overlap and piecewise labels. Random rows are mean [min--max] over seeds 42, 43, 44; time-block uses chronological 70/15/15 per bearing.}")
    lines.append(r"\label{tab:sensitivity_nonoverlap}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Split & $R^2$ \\")
    lines.append(r"\midrule")
    mode_labels = {"random_60": "Random 60/20/20", "random_80": "Random 80/10/10", "chronological_70": "Time-block 70/15/15"}
    for ds in MAIN_FILES:
        root = sens[ds]
        lines.append(r"\midrule")
        for mode in ["random_60", "random_80", "chronological_70"]:
            v = root["splits"][mode]
            lines.append(rf"{ds} & {mode_labels[mode]} & {fmt(v['r2_mean'])} [{fmt(v['r2_min'])}, {fmt(v['r2_max'])}] \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_table5(data):
    rows = [
        ("PHM2012", "TCN"),
        ("PHM2012", "PatchTST"),
        ("IMS", "TCN"),
        ("IMS", "PatchTST"),
    ]
    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Illustrative protocol-dependent model-selection results under non-overlap. Random split uses piecewise labels; LOOCV uses linear labels.}")
    lines.append(r"\label{tab:deployment_selection_nonoverlap}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Model & Piecewise random $R^2$ & Linear LOOCV $R^2$ \\")
    lines.append(r"\midrule")
    for ds, model in rows:
        pw_rnd = data[ds]["piecewise"][model]["random"]["r2"]["mean"]
        lin_loo = data[ds]["linear"][model]["loocv"]["r2"]["mean"]
        lines.append(rf"{ds} & {MODEL_LABELS[model]} & {fmt(pw_rnd)} & {fmt(lin_loo)} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_appendix_b(data):
    lines = []
    lines.append(r"\begin{table*}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-fold and per-seed evidence for the non-overlap primary protocol comparison. LOOCV range is the min--max of linear-label fold $R^2$; seed columns are piecewise-label random-split $R^2$ for seeds 42--49.}")
    lines.append(r"\label{tab:perfold_nonoverlap}")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lllccccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Model & Linear LOOCV range & S42 & S43 & S44 & S45 & S46 & S47 & S48 & S49 & Protocol gap \\")
    lines.append(r"\midrule")
    for ds in MAIN_FILES:
        labels = data[ds]
        for model in MODELS:
            lin_folds = [f["r2"] for f in labels["linear"][model]["loocv"]["folds"]]
            pw_reps = [r["r2"] for r in labels["piecewise"][model]["random"]["reps"]]
            gap = labels["piecewise"][model]["random"]["r2"]["mean"] - labels["linear"][model]["loocv"]["r2"]["mean"]
            lo = min(lin_folds)
            hi = max(lin_folds)
            lines.append(rf"{ds} & {MODEL_LABELS[model]} & ${fmt(lo)}$ to ${fmt(hi)}$ & " + " & ".join(fmt(x) for x in pw_reps) + rf" & {fmt(gap)} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def write_stats(data):
    OUT_STATS.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in MAIN_FILES:
        labels = data[ds]
        for model in DEEP_MODELS:
            lin_folds = [f["r2"] for f in labels["linear"][model]["loocv"]["folds"]]
            pw_reps = [r["r2"] for r in labels["piecewise"][model]["random"]["reps"]]
            rows.append((ds, model, pairwise_above(pw_reps, lin_folds), cliff(pw_reps, lin_folds)))
    with open(OUT_STATS / "cliff_and_permutation_nonoverlap.csv", "w", encoding="utf-8") as f:
        f.write("dataset,model,pairwise_random_gt_loo,cliff_delta\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f}\n")


def make_figure(data):
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), sharey=True)
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    for ax, ds in zip(axes, MAIN_FILES):
        labels = data[ds]
        x = np.arange(len(MODELS))
        width = 0.25
        lin_means, lin_errs = [], []
        pw_means, pw_errs = [], []
        rnd_means, rnd_errs = [], []
        for model in MODELS:
            lin = labels["linear"][model]["loocv"]["r2"]
            pw = labels["piecewise"][model]["loocv"]["r2"]
            rnd = labels["piecewise"][model]["random"]["r2"]
            lin_means.append(lin["mean"])
            lin_errs.append([max(0.0, lin["mean"] - lin["ci95_low"]), max(0.0, lin["ci95_high"] - lin["mean"])])
            pw_means.append(pw["mean"])
            pw_errs.append([max(0.0, pw["mean"] - pw["ci95_low"]), max(0.0, pw["ci95_high"] - pw["mean"])])
            rnd_means.append(rnd["mean"])
            rnd_errs.append([max(0.0, rnd["mean"] - rnd["min"]), max(0.0, rnd["max"] - rnd["mean"])])
        ax.bar(x - width, lin_means, width, yerr=np.asarray(lin_errs).T, label="Linear LOOCV", color=colors[0], capsize=3)
        ax.bar(x, pw_means, width, yerr=np.asarray(pw_errs).T, label="Piecewise LOOCV", color=colors[1], capsize=3)
        ax.bar(x + width, rnd_means, width, yerr=np.asarray(rnd_errs).T, label="Piecewise random", color=colors[2], capsize=3)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_title(ds, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(["LR", "LSTM", "TCN", "PatchTST"])
        ax.tick_params(axis='both', labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("$R^2$", fontsize=12)
    axes[0].legend(frameon=False, loc="lower right", fontsize=11)
    fig.tight_layout()
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURES / "converged_r2_comparison_nonoverlap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIGURES / "converged_r2_comparison_nonoverlap.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    data = load_main()
    sens = load_sens()
    write_text("table1_main_nonoverlap.tex", build_table1(data))
    write_text("table2_factorial_nonoverlap.tex", build_table2(data))
    write_text("table3_errors_nonoverlap.tex", build_table3(data))
    write_text("table4_sensitivity_nonoverlap.tex", build_table4(sens))
    write_text("table5_deployment_nonoverlap.tex", build_table5(data))
    write_text("appendix_b_nonoverlap.tex", build_appendix_b(data))
    write_stats(data)
    make_figure(data)
    print("saved non-overlap tables, stats, and figure")


if __name__ == "__main__":
    main()
