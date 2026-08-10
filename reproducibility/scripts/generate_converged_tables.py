from pathlib import Path
import json
import itertools

import numpy as np

BASE = Path(__file__).resolve().parents[1] / "results"
FILES = {
    "XJTU-SY": "converged_xjtu8_v1.json",
    "PHM2012": "converged_phm_v1.json",
    "IMS": "converged_ims_v1.json",
}
MODELS = ["LinearRegression", "StatLSTM", "TCN", "PatchTST"]


def load():
    out = {}
    for ds, name in FILES.items():
        p = BASE / name
        if p.exists():
            root = json.loads(p.read_text(encoding="utf-8"))
            out[ds] = root.get("datasets", {}).get(ds, {})
    return out


def fmt(x, digits=3):
    if x is None:
        return "--"
    return f"{x:.{digits}f}"


def ci_str(v):
    if not v:
        return "--"
    lo = v.get("ci95_low")
    hi = v.get("ci95_high")
    mean = v.get("mean")
    return f"{fmt(mean)} [{fmt(lo)}; {fmt(hi)}]"


def random_str(v):
    if not v:
        return "--"
    return f"{fmt(v.get('mean'))} [{fmt(v.get('min'))}; {fmt(v.get('max'))}]"


def cliff(a, b):
    a = list(a)
    b = list(b)
    if not a or not b:
        return None
    wins = 0
    for x, y in itertools.product(a, b):
        if x > y:
            wins += 1
        elif x < y:
            wins -= 1
    return wins / (len(a) * len(b))


def main():
    data = load()
    for ds in ["XJTU-SY", "PHM2012", "IMS"]:
        labels = data.get(ds)
        if not labels:
            print("% missing", ds)
            continue
        print("%", ds)
        print(r"\midrule")
        for model in MODELS:
            lin = labels.get("linear", {}).get(model, {})
            pw = labels.get("piecewise", {}).get(model, {})
            lin_loo = lin.get("loocv", {}).get("r2")
            pw_loo = pw.get("loocv", {}).get("r2")
            pw_rnd = pw.get("random", {}).get("r2")
            lin_folds = [f["r2"] for f in lin.get("loocv", {}).get("folds", [])]
            pw_folds = [f["r2"] for f in pw.get("loocv", {}).get("folds", [])]
            rnd_reps = [r["r2"] for r in pw.get("random", {}).get("reps", [])]
            d_label = cliff(pw_folds, lin_folds)
            d_protocol = cliff(rnd_reps, lin_folds)
            delta_label = None
            if pw_loo and lin_loo:
                delta_label = pw_loo["mean"] - lin_loo["mean"]
            delta_protocol = None
            if pw_rnd and lin_loo:
                delta_protocol = pw_rnd["mean"] - lin_loo["mean"]
            print(f"{model} & {ci_str(lin_loo)} & {ci_str(pw_loo)} & {fmt(delta_label)} & {random_str(pw_rnd)} & {fmt(delta_protocol)} & {fmt(d_label)} & {fmt(d_protocol)} \\\\")
    print("% END")


if __name__ == "__main__":
    main()
