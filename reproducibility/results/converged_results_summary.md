# Converged RUL Protocol Results (v14)

## Protocol
- Datasets: XJTU-SY (8 bearings), PHM2012 (6 bearings), IMS (3 test sets)
- Models: LinearRegression, StatLSTM, TCN, PatchTST, Constant baseline
- Labels: linear, piecewise (80% plateau)
- Evaluation: LOOCV and random 70/15/15 time-step split
- Training: AdamW, MSE, cosine annealing, batch size 1024, early stopping patience 30
- Max epochs: 100 for XJTU-SY and PHM2012, 60 for IMS
- Random seeds: 42, 43, 44, 45, 46, 47, 48, 49
- Confidence intervals: 95% t-intervals over held-out bearings
- Effect size: Cliff's delta with exact permutation tests

## Main Results
R2 linear LOOCV -> R2 piecewise random

| Dataset | Model | Linear LOOCV R2 [95% CI] | Piecewise Random R2 [min-max] | Protocol gap |
|---|---|---|---|---|
| XJTU-SY | Linear Regression | 0.178 [0.176; 0.180] | 0.405 [0.364; 0.447] | 0.228 |
| XJTU-SY | StatLSTM | 0.223 [0.221; 0.224] | 0.410 [0.385; 0.435] | 0.188 |
| XJTU-SY | TCN | 0.222 [0.220; 0.224] | 0.410 [0.382; 0.440] | 0.188 |
| XJTU-SY | PatchTST | 0.222 [0.220; 0.224] | 0.416 [0.380; 0.453] | 0.194 |
| PHM2012 | Linear Regression | -0.168 [-0.366; 0.029] | 0.416 [0.376; 0.465] | 0.584 |
| PHM2012 | StatLSTM | 0.245 [0.020; 0.469] | 0.785 [0.757; 0.809] | 0.540 |
| PHM2012 | TCN | 0.174 [0.030; 0.318] | 0.843 [0.818; 0.865] | 0.669 |
| PHM2012 | PatchTST | 0.369 [0.159; 0.578] | 0.823 [0.775; 0.857] | 0.454 |
| IMS | Linear Regression | -2.109 [-5.677; 1.459] | 0.320 [0.310; 0.333] | 2.429 |
| IMS | StatLSTM | -1.208 [-3.714; 1.297] | 0.692 [0.681; 0.706] | 1.900 |
| IMS | TCN | -2.331 [-5.216; 0.554] | 0.710 [0.703; 0.717] | 3.041 |
| IMS | PatchTST | -1.058 [-3.125; 1.009] | 0.659 [0.639; 0.689] | 1.718 |

Cliff's delta for the protocol effect (piecewise random vs linear LOOCV) is 1.0 in all twelve model-dataset cells.

## Statistical Evidence
- XJTU-SY exact permutation p = 7.8e-5 (1/12870)
- PHM2012 exact permutation p = 3.3e-4 (1/3003)
- IMS exact permutation p = 0.0061 (1/165)
- Per-fold and per-seed values are listed in `results/v14_summary.json`, `results/cliff_delta_exact.csv`, and the manuscript appendix.

## Result Files
- `results/converged_xjtu8_v1.json`
- `results/converged_phm_v1.json`
- `results/converged_ims_v1.json`
- `results/v14_summary.json`
- `results/cliff_delta_exact.csv`
- `results/global_normalization_phm_tcn.json`
- `results/validation_sensitivity_phm_tcn.json`

## Reproducibility Scripts
- `scripts/run_converged_rul_study.py` (base converged protocol)
- `scripts/run_converged_star.py` (flat random-split wrapper)
- `scripts/run_split_sensitivity.py`
- Table/figure generators are provided in the paper reproducibility package.
