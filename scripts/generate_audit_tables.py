# -*- coding: utf-8 -*-
"""Generate the v26 audit tables that use the existing non-overlap result JSON."""

import json
import math
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results_v26"
OUT = Path(__file__).resolve().parents[1] / "tables_nonoverlap"

MAIN_FILES = {
    "XJTU-SY": RESULTS / "converged_xjtu8_nonoverlap_v26.json",
    "PHM2012": RESULTS / "converged_phm_nonoverlap_v26.json",
    "IMS": RESULTS / "converged_ims_nonoverlap_v26.json",
}

MODELS = ["LinearRegression", "StatLSTM", "TCN", "PatchTST"]
DEEP_MODELS = ["StatLSTM", "TCN", "PatchTST"]
MODEL_LABELS = {
    "LinearRegression": "Linear Reg.",
    "StatLSTM": "LSTM",
    "TCN": "TCN",
    "PatchTST": "PatchTST",
}
STRIDE1_FILES = {
    "XJTU-SY": RESULTS / "converged_xjtu8_stride1_v1.json",
    "PHM2012": RESULTS / "converged_phm_stride1_v1.json",
    "IMS": RESULTS / "converged_ims_stride1_v1.json",
}


def build_stride_table(data):
    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        r"\caption{Random-split history-construction comparison (stride=1 versus stride=10). Values are mean $R^2$, RMSE, and MAE across eight seeds (42--49); $\Delta R^2$ is the stride=10 value minus the stride=1 value. RMSE and MAE are computed on trajectory-normalized RUL.}",
        r"\label{tab:stride_random}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllccccccc}",
        r"\toprule",
        r"Dataset & Label & Model & $R^2_{\mathrm{stride}=1}$ & $R^2_{\mathrm{stride}=10}$ & $\Delta R^2$ & $\mathrm{RMSE}_{\mathrm{stride}=1}$ & $\mathrm{RMSE}_{\mathrm{stride}=10}$ & $\mathrm{MAE}_{\mathrm{stride}=1}$ & $\mathrm{MAE}_{\mathrm{stride}=10}$ \\",
        r"\midrule",
    ]
    for ds in STRIDE1_FILES:
        stride1 = json.loads(STRIDE1_FILES[ds].read_text(encoding="utf-8"))["datasets"][ds]
        stride10 = data[ds]
        lines.append(r"\midrule")
        for label in ["linear", "piecewise"]:
            for model in DEEP_MODELS:
                s1 = stride1[label][model]["random"]
                s10 = stride10[label][model]["random"]
                delta = s10["r2"]["mean"] - s1["r2"]["mean"]
                label_name = "Linear" if label == "linear" else "Piecewise"
                lines.append(
                    rf"{ds} & {label_name} & {MODEL_LABELS[model]} "
                    rf"& {fmt(s1['r2']['mean'])} & {fmt(s10['r2']['mean'])} & {fmt(delta)} "
                    rf"& {fmt(s1['rmse']['mean'])} & {fmt(s10['rmse']['mean'])} "
                    rf"& {fmt(s1['mae']['mean'])} & {fmt(s10['mae']['mean'])} \\"
                )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    return "\n".join(lines)


def fmt(x, digits=3):
    if x is None:
        return "--"
    if abs(x) < 5e-4:
        return "0.000"
    return f"{x:.{digits}f}"


def loo(v):
    lo = v.get("ci95_low")
    hi = v.get("ci95_high")
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or not math.isfinite(lo) or not math.isfinite(hi):
        return fmt(v["mean"])
    return f"{fmt(v['mean'])} [{fmt(lo)}, {fmt(hi)}]"


def rnd(v):
    return f"{fmt(v['mean'])} [{fmt(v['min'])}, {fmt(v['max'])}]"


def load_main():
    out = {}
    for ds, path in MAIN_FILES.items():
        root = json.loads(path.read_text(encoding="utf-8"))
        out[ds] = root["datasets"][ds]
    return out


def build_rmse_table(data):
    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        r"\caption{Complete $2\times2$ non-overlap protocol comparison for RMSE. LOOCV values are mean [95\% CI] across held-out bearings; random values are mean [min--max] across eight seeds (42--49). RMSE is computed on trajectory-normalized RUL.}",
        r"\label{tab:rmse_2x2_nonoverlap}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Model & Linear LOOCV & Linear random & Piecewise LOOCV & Piecewise random \\",
        r"\midrule",
    ]
    for ds in MAIN_FILES:
        labels = data[ds]
        lines.append(r"\midrule")
        for model in MODELS:
            lin_loo = labels["linear"][model]["loocv"]["rmse"]
            lin_rnd = labels["linear"][model]["random"]["rmse"]
            pw_loo = labels["piecewise"][model]["loocv"]["rmse"]
            pw_rnd = labels["piecewise"][model]["random"]["rmse"]
            lines.append(rf"{ds} & {MODEL_LABELS[model]} & {loo(lin_loo)} & {rnd(lin_rnd)} & {loo(pw_loo)} & {rnd(pw_rnd)} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    return "\n".join(lines)


def build_mae_table(data):
    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        r"\caption{Complete $2\times2$ non-overlap protocol comparison for MAE. LOOCV values are mean [95\% CI] across held-out bearings; random values are mean [min--max] across eight seeds (42--49). MAE is computed on trajectory-normalized RUL.}",
        r"\label{tab:mae_2x2_nonoverlap}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Model & Linear LOOCV & Linear random & Piecewise LOOCV & Piecewise random \\",
        r"\midrule",
    ]
    for ds in MAIN_FILES:
        labels = data[ds]
        lines.append(r"\midrule")
        for model in MODELS:
            lin_loo = labels["linear"][model]["loocv"]["mae"]
            lin_rnd = labels["linear"][model]["random"]["mae"]
            pw_loo = labels["piecewise"][model]["loocv"]["mae"]
            pw_rnd = labels["piecewise"][model]["random"]["mae"]
            lines.append(rf"{ds} & {MODEL_LABELS[model]} & {loo(lin_loo)} & {rnd(lin_rnd)} & {loo(pw_loo)} & {rnd(pw_rnd)} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    return "\n".join(lines)


def build_constant_baseline_table(data):
    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        r"\caption{Constant predictor baseline under the non-overlapping stride-10 protocol. The constant predictor always outputs the mean training RUL. LOOCV values are mean [95\% CI]; random values are mean [min--max] across eight seeds.}",
        r"\label{tab:constant_baseline_nonoverlap}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllccc}",
        r"\toprule",
        r"Dataset & Label & Protocol & $R^2$ & RMSE & MAE \\",
        r"\midrule",
    ]
    for ds in MAIN_FILES:
        labels = data[ds]
        for label in ["linear", "piecewise"]:
            constant = labels[label]["Constant"]
            lines.append(rf"\midrule")
            lines.append(rf"{ds} & {label} & LOOCV & {loo(constant['loocv']['r2'])} & {loo(constant['loocv']['rmse'])} & {loo(constant['loocv']['mae'])} \\")
            lines.append(rf"{ds} & {label} & Random & {rnd(constant['random']['r2'])} & {rnd(constant['random']['rmse'])} & {rnd(constant['random']['mae'])} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    return "\n".join(lines)


def build_validation_table():
    path = RESULTS / "validation_sensitivity_phm_tcn_nonoverlap.json"
    root = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Validation-bearing sensitivity for PHM2012 TCN under the non-overlapping stride-10 protocol and linear labels. The main LOOCV protocol uses offset 1. Values are means across the six held-out bearings.}",
        r"\label{tab:validation_sensitivity_nonoverlap}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Validation offset & $R^2$ mean & $R^2$ min--max & RMSE mean & MAE mean \\",
        r"\midrule",
    ]
    for offset, v in root["validation_offsets"].items():
        lines.append(
            rf"{offset} & {fmt(v['r2_mean'])} & {fmt(v['r2_min'])}--{fmt(v['r2_max'])} "
            rf"& {fmt(v['rmse_mean'])} & {fmt(v['mae_mean'])} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_global_normalization_table():
    path = RESULTS / "global_normalization_phm_tcn_nonoverlap.json"
    root = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Global-normalization sensitivity for PHM2012 TCN under the non-overlapping stride-10 protocol. Random values are mean [min--max] across eight seeds (42--49).}",
        r"\label{tab:global_normalization_nonoverlap}",
        r"\small",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Label & LOOCV $R^2$ & Random $R^2$ & Random RMSE \\",
        r"\midrule",
    ]
    for label, v in root["labels"].items():
        loo_metrics = v["loocv"]["r2"]
        random_r2 = v["random_reps"]
        r2_mean = v["random_mean"]
        r2_min = v["random_min"]
        r2_max = v["random_max"]
        rmse_mean = float(np.mean([r["rmse"] for r in random_r2]))
        lines.append(
            rf"{label} & {loo(loo_metrics)} & {fmt(r2_mean)} [{fmt(r2_min)}, {fmt(r2_max)}] & {fmt(rmse_mean)} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def bootstrap_ci(values, n_boot=10000, seed=123):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return arr[0], arr[0]
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def build_random_bootstrap_table(data):
    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        r"\caption{Seed-level bootstrap summaries for random sequence-level splitting under the non-overlapping stride-10 protocol. Bootstrap intervals resample the eight fixed seed runs and quantify computational split sensitivity, not independent bearing-level uncertainty.}",
        r"\label{tab:random_bootstrap_nonoverlap}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllccccc}",
        r"\toprule",
        r"Dataset & Model & Label & $R^2$ mean [95\% CI] & RMSE mean [95\% CI] & MAE mean [95\% CI] \\",
        r"\midrule",
    ]
    for ds in MAIN_FILES:
        labels = data[ds]
        for label in ["linear", "piecewise"]:
            for model in MODELS:
                random = labels[label][model]["random"]
                r2_rep = [r["r2"] for r in random["reps"]]
                rmse_rep = [r["rmse"] for r in random["reps"]]
                mae_rep = [r["mae"] for r in random["reps"]]
                r2_lo, r2_hi = bootstrap_ci(r2_rep)
                rmse_lo, rmse_hi = bootstrap_ci(rmse_rep)
                mae_lo, mae_hi = bootstrap_ci(mae_rep)
                lines.append(
                    rf"{ds} & {MODEL_LABELS[model]} & {label} "
                    rf"& {fmt(random['r2']['mean'])} [{fmt(r2_lo)}, {fmt(r2_hi)}] "
                    rf"& {fmt(random['rmse']['mean'])} [{fmt(rmse_lo)}, {fmt(rmse_hi)}] "
                    rf"& {fmt(random['mae']['mean'])} [{fmt(mae_lo)}, {fmt(mae_hi)}] \\"
                )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    return "\n".join(lines)


def main():
    data = load_main()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table_stride_random.tex").write_text(build_stride_table(data), encoding="utf-8")
    (OUT / "table_rmse_2x2_nonoverlap.tex").write_text(build_rmse_table(data), encoding="utf-8")
    (OUT / "table_mae_2x2_nonoverlap.tex").write_text(build_mae_table(data), encoding="utf-8")
    (OUT / "table_constant_baseline_nonoverlap.tex").write_text(build_constant_baseline_table(data), encoding="utf-8")
    (OUT / "table_random_bootstrap_nonoverlap.tex").write_text(build_random_bootstrap_table(data), encoding="utf-8")
    validation_path = RESULTS / "validation_sensitivity_phm_tcn_nonoverlap.json"
    global_path = RESULTS / "global_normalization_phm_tcn_nonoverlap.json"
    if validation_path.exists():
        (OUT / "table_validation_sensitivity_nonoverlap.tex").write_text(build_validation_table(), encoding="utf-8")
    if global_path.exists():
        (OUT / "table_global_normalization_nonoverlap.tex").write_text(build_global_normalization_table(), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
