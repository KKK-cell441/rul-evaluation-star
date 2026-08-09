# RUL Evaluation Protocol Reproducibility Package

This package supports the manuscript:
*Rethinking RUL Labeling Strategies in Bearing Prognostics: Converged Multi-Dataset Evidence for Evaluation Inflation*

## Environment
```
pip install -r requirements.txt
```

## Data
The scripts expect processed data under `data/processed/` in the following layout:
- `XJTU-SY/`
- `PHM2012/`
- `IMS/`

Use `scripts/preprocess_public_rul.py` for PHM2012 and IMS, and `scripts/preprocess_xjtu15.py` for XJTU-SY. The raw datasets are publicly available from their original sources.

## Converged Protocol
Run:
```
python scripts/run_converged_star.py \
  --data-dir data/processed/XJTU-SY \
  --models Constant LinearRegression StatLSTM TCN \
  --labels linear piecewise \
  --epochs 100 \
  --random-seeds 42 43 44 \
  --output results/converged_xjtu8_v1.json
```

Repeat for PHM2012 with `--epochs 100` and IMS with `--epochs 60`.

## Split Sensitivity
Run:
```
python scripts/run_split_sensitivity.py \
  --data-dir data/processed/PHM2012 \
  --epochs 100 \
  --output results/split_sensitivity_phm_v1.json
```

This runs TCN with piecewise labels under random 60/20/20, random 80/10/10, and chronological time-block 70/15/15.

## Outputs
- `results/converged_*.json`: main converged protocol, per-fold and per-seed results.
- `results/split_sensitivity_*.json`: split-ratio and time-block sensitivity.
- `supplementary/supplementary_prevalence_survey.pdf`: prevalence survey coding rules and coding sheet draft.

## Status
The scientific results are final. Before journal submission, the following author-controlled items must be completed:
- Corresponding email and ORCID.
- Funding statement.
- Public repository URL and Zenodo DOI.
- Final full-text verification of all 20 prevalence survey rows.
