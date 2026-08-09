from pathlib import Path
import json

BASE = Path.home() / "Documents" / "\u673a\u5668\u5b66\u4e60" / "RUL-Prediction-Framework" / "results"
FILES = {
    "XJTU-SY": "split_sensitivity_xjtu8_v1.json",
    "PHM2012": "split_sensitivity_phm_v1.json",
    "IMS": "split_sensitivity_ims_v1.json",
}
MODES = ["random_60", "random_80", "chronological_70"]


def fmt(x):
    return f"{x:.3f}"


def main():
    for ds, name in FILES.items():
        p = BASE / name
        if not p.exists():
            print("% missing", ds)
            continue
        root = json.loads(p.read_text(encoding="utf-8"))
        print("%", ds)
        print(r"\midrule")
        for mode in MODES:
            v = root.get("splits", {}).get(mode)
            if not v:
                continue
            label = {"random_60": "Random 60/20/20", "random_80": "Random 80/10/10", "chronological_70": "Time-block 70/15/15"}[mode]
            print(f"{label} & {fmt(v['r2_mean'])} [{fmt(v['r2_min'])}; {fmt(v['r2_max'])}] & {int(round(v['best_epoch_mean']))} \\\\")
    print("% END")


if __name__ == "__main__":
    main()
