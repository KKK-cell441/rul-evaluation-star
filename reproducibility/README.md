# RUL Evaluation Protocol Reproducibility Package

This package supports the manuscript:
*Reconsidering Reliability Assessment of Data-Driven Predictive Maintenance: The Impact of Evaluation Protocols on Bearing Remaining Useful Life Prediction*

## Environment

- Python 3.12
- PyTorch >= 2.0
- NumPy, SciPy, scikit-learn, pandas, PyWavelets, matplotlib, seaborn

Install with:

```bash
pip install -r requirements.txt
```

## Data

The scripts expect processed data under `data/processed/` in the following layout:

- `XJTU-SY/`
- `PHM2012/`
- `IMS/`

Use `scripts/preprocess_public_rul.py` for PHM2012 and IMS, and `scripts/preprocess_xjtu15.py` for XJTU-SY. Raw datasets are publicly available from their original sources.

## Converged Protocol

Run:

```bash
python scripts/run_converged_rul_study.py \
  --data-dir data/processed/XJTU-SY \
  --models Constant LinearRegression StatLSTM TCN PatchTST \
  --labels linear piecewise \
  --epochs 100 \
  --random-seeds 42 43 44 45 46 47 48 49 \
  --output results/converged_xjtu8_v1.json
```

Repeat for PHM2012 with `--epochs 100` and IMS with `--epochs 60`.

## Split Sensitivity

```bash
python scripts/run_split_sensitivity.py \
  --data-dir data/processed/PHM2012 \
  --epochs 100 \
  --output results/split_sensitivity_phm_v1.json
```

This runs TCN with piecewise labels under random 60/20/20, random 80/10/10, and chronological time-block 70/15/15.

## Additional Sensitivity Analyses

- `results/global_normalization_phm_tcn.json`: global-normalized RUL targets for PHM2012 TCN.
- `results/validation_sensitivity_phm_tcn.json`: PHM2012 TCN linear-LOOCV with validation-bearing offsets 1, 2, and 3.
- `results/cliff_delta_exact.csv` and `results/v14_summary.json`: 8-seed Cliff's delta and exact permutation summaries.

## Outputs and Table Mapping

- `results/converged_*.json`: main protocol results, including per-fold and per-seed values, RMSE, MAE, and R2.
- `results/split_sensitivity_*.json`: split-ratio and time-block sensitivity.
- Table 1, Table 2, and the RMSE/MAE table are generated from `results/converged_*.json`.
- Table 4 is generated from `results/split_sensitivity_*.json`.
- Table 5 is the STAR checklist in the manuscript.

## Status

The scientific results are final. The primary random-split protocol uses eight fixed seeds (42-49) for all 12 model-dataset cells.

Before journal submission, the following author-controlled items must be completed:

- ORCID: 0009-0003-4491-6619
- Email: 2024212760@nefu.edu.cn
- Funding statement
- Public repository: https://github.com/KKK-cell441/rul-evaluation-star
- DOI: https://doi.org/10.5281/zenodo.21866359
- DOI note: 21866359 is the current archived release; update after the v18 package is released.

Quantitative prevalence survey counts are not part of this version. Protocol-level claims are limited to the reproduced experimental family.

The package reproduces the benchmark experiments reported in the manuscript; it does not reproduce industrial deployment results.
