# Converged RUL Protocol Results (v4)

## Protocol
- Datasets: XJTU-SY (8 bearings), PHM2012 (6 bearings), IMS (3 test sets)
- Models: LinearRegression, StatLSTM, TCN, Constant baseline
- Labels: linear, piecewise (80% plateau)
- Evaluation: LOOCV and random 70/15/15 time-step split
- Training: AdamW, MSE, cosine annealing, batch size 1024, early stopping patience 30
- Max epochs: 100 for XJTU-SY and PHM2012, 60 for IMS
- Random seeds: 42, 43, 44
- Confidence intervals: 95% t-intervals over held-out bearings
- Effect size: Cliff's delta

## Main Results
R2 linear LOOCV -> R2 piecewise random

| Dataset | Model | Linear LOOCV R2 [95% CI] | Piecewise Random R2 [min-max] | Protocol gap |
|---|---|---|---|---|
| XJTU-SY | Linear Regression | 0.178 [0.176; 0.180] | 0.430 [0.414; 0.447] | 0.252 |
| XJTU-SY | StatLSTM | 0.223 [0.221; 0.224] | 0.422 [0.401; 0.435] | 0.199 |
| XJTU-SY | TCN | 0.222 [0.220; 0.224] | 0.421 [0.396; 0.433] | 0.199 |
| PHM2012 | Linear Regression | -0.168 [-0.366; 0.029] | 0.397 [0.376; 0.420] | 0.566 |
| PHM2012 | StatLSTM | 0.245 [0.020; 0.469] | 0.791 [0.778; 0.803] | 0.546 |
| PHM2012 | TCN | 0.174 [0.030; 0.318] | 0.862 [0.859; 0.865] | 0.688 |
| IMS | Linear Regression | -2.109 [-5.677; 1.459] | 0.320 [0.313; 0.328] | 2.429 |
| IMS | StatLSTM | -1.208 [-3.714; 1.297] | 0.699 [0.695; 0.706] | 1.907 |
| IMS | TCN | -2.331 [-5.216; 0.554] | 0.715 [0.714; 0.717] | 3.046 |

Cliff's delta for the protocol effect (piecewise random vs linear LOOCV) is 1.0 in all nine model-dataset cells.

## Result Files
- `results/converged_xjtu8_v1.json`
- `results/converged_phm_v1.json`
- `results/converged_ims_v1.json`

## Reproducibility Scripts
- `scripts/run_converged_rul_study.py` (base converged protocol)
- `scripts/run_converged_star.py` (flat random-split wrapper)
- Table/figure generators are provided in the paper reproducibility package.
