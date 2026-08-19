# RUL Evaluation Protocol Audit: v26 Reproducibility Package

This package supports the manuscript:

*An Evaluation-Protocol Audit Framework for Data-Driven Bearing Remaining Useful Life Prediction*

The package uses the non-overlapping stride-10 history construction described in the manuscript and reproduces the reported main tables, sensitivity analyses, validation-bearing sensitivity, and global-normalization sensitivity.

## Environment

- Python 3.12
- PyTorch >= 2.0 with CUDA
- NumPy, SciPy, scikit-learn, pandas, matplotlib, seaborn, PyWavelets

Install with:

```bash
pip install -r requirements.txt
```

## Data

Raw XJTU-SY, PHM2012, and IMS datasets are public. Run the preprocessing scripts first:

```bash
python scripts/preprocess_xjtu15.py --input <raw-xjtu> --output data/processed/XJTU-SY
python scripts/preprocess_public_rul.py --input <raw-phm2012> --output data/processed/PHM2012
python scripts/preprocess_public_rul.py --input <raw-ims> --output data/processed/IMS
```

Each processed directory must contain `features.npy`, `rul_linear.npy`, `rul_piecewise.npy`, `n_per_bearing.npy`, and `normalization_stats.json`.

## Main Protocol

```bash
python scripts/run_converged_nonoverlap.py \
  --data-dir data/processed/XJTU-SY \
  --models Constant LinearRegression StatLSTM TCN PatchTST \
  --labels linear piecewise \
  --epochs 100 \
  --random-seeds 42 43 44 45 46 47 48 49 \
  --output results/converged_xjtu8_nonoverlap_v26.json
```

Repeat with PHM2012 using `--epochs 100` and IMS using `--epochs 60`.

## Split Sensitivity

```bash
python scripts/run_split_sensitivity_nonoverlap.py \
  --data-dir data/processed/PHM2012 \
  --epochs 100 \
  --output results/split_sensitivity_phm_nonoverlap_v26.json
```

Repeat for XJTU-SY and IMS.

## Additional Sensitivity Analyses

```bash
python scripts/run_validation_sensitivity_nonoverlap.py \
  --data-dir data/processed/PHM2012 \
  --epochs 100
python scripts/run_global_normalization_nonoverlap.py \
  --data-dir data/processed/PHM2012 \
  --epochs 100
```

These scripts use the processed PHM2012 data and write to `results/`.

## Table and Figure Generation

```bash
python scripts/generate_nonoverlap_artifacts.py
python scripts/generate_audit_tables.py
```

## Result Files

- `results/converged_*_nonoverlap_v26.json`: main R2, RMSE, MAE, per-fold, and per-seed values.
- `results/split_sensitivity_*_nonoverlap_v26.json`: random-ratio and chronological split sensitivity.
- `results/validation_sensitivity_phm_tcn_nonoverlap.json`: validation-bearing rotation results.
- `results/global_normalization_phm_tcn_nonoverlap.json`: global-normalization sensitivity.

## Reproducibility Notes

The package uses fixed split seeds 42-49 and a fixed model-initialization seed. GPU numerical libraries can still produce small run-to-run differences, so reported values should be treated as converged protocol results rather than bit-exact GPU outputs. No local machine-specific absolute paths are required.
