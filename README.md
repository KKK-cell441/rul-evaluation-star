# RUL Evaluation Protocol Study

A reproducible framework for evaluating the influence of labeling and protocol design on bearing RUL prediction performance.

Reproducibility package for the manuscript:

**Reconsidering Reliability Assessment of Data-Driven Predictive Maintenance: The Impact of Evaluation Protocols on Bearing Remaining Useful Life Prediction**

## Contents

- `reproducibility/` contains preprocessing scripts, converged training and split-sensitivity runners, result JSON files, and table/figure generators.
- `manuscript_revised_v18_jim.pdf` is the latest Springer/JIM submission candidate.
- `manuscript_revised_v7.pdf` remains the Elsevier source version.
- `figures/` contains the figures used in the manuscript.
- `figures/graphical_abstract.png` and `graphical_abstract.pdf` are provided for the JIM submission package.

## Environment

- Python 3.12
- PyTorch >= 2.0
- NumPy, SciPy, scikit-learn, pandas, PyWavelets, matplotlib, seaborn

Install with:

```bash
pip install -r reproducibility/requirements.txt
```

## Data

Raw datasets are publicly available from their original sources. Download them first, then run:

- `reproducibility/scripts/preprocess_public_rul.py` for PHM2012 and IMS
- `reproducibility/scripts/preprocess_xjtu15.py` for XJTU-SY

Place processed data under `reproducibility/data/processed/` with `XJTU-SY/`, `PHM2012/`, and `IMS/` subdirectories.

## Quick Reproduction

```bash
python reproducibility/scripts/run_converged_rul_study.py \
  --data-dir reproducibility/data/processed/XJTU-SY \
  --models Constant LinearRegression StatLSTM TCN PatchTST \
  --labels linear piecewise \
  --epochs 100 \
  --random-seeds 42 43 44 45 46 47 48 49 \
  --output reproducibility/results/converged_xjtu8_v1.json
```

Repeat for PHM2012 with `--epochs 100` and IMS with `--epochs 60`.

## Split Sensitivity

```bash
python reproducibility/scripts/run_split_sensitivity.py \
  --data-dir reproducibility/data/processed/PHM2012 \
  --epochs 100 \
  --output reproducibility/results/split_sensitivity_phm_v1.json
```

## Additional Sensitivity Analyses

- Global normalization: `reproducibility/results/global_normalization_phm_tcn.json`
- Validation-bearing rotations: `reproducibility/results/validation_sensitivity_phm_tcn.json`
- Exact permutation and Cliff's delta: `reproducibility/results/cliff_delta_exact.csv` and `reproducibility/results/v14_summary.json`

## Table Reproduction

- Main R2 and 2x2 factorial tables: `reproducibility/results/converged_*.json`
- RMSE/MAE table: `reproducibility/results/converged_*.json`
- Split sensitivity table: `reproducibility/results/split_sensitivity_*.json`
- Statistical summary: `reproducibility/results/v14_summary.json`
- STAR checklist table: static in the manuscript

See `reproducibility/README.md` for additional details.
