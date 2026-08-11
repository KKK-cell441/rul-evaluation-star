# Reproducibility Notes (v18)

## Datasets and preprocessing
- XJTU-SY: `scripts/preprocess_xjtu15.py` (full public dataset; v14 uses 8 bearings from conditions 1-2).
- PHM2012: `scripts/preprocess_public_rul.py`.
- IMS: `scripts/preprocess_public_rul.py`.
- Feature extraction uses the dbca-net preprocessing module with 2,560-sample windows, 10 statistical/spectral features, and history length 10.

## Converged protocol
- Models: LinearRegression, StatLSTM, TCN, PatchTST, Constant.
- Labels: linear, piecewise (80% plateau).
- Evaluation: LOOCV and random 70/15/15 time-step splits.
- Training: AdamW, MSE, cosine annealing, batch size 1024, early stopping patience 30.
- Max epochs: 100 (XJTU-SY, PHM2012), 60 (IMS).
- Random seeds: 42, 43, 44, 45, 46, 47, 48, 49.
- Exact permutation tests: XJTU-SY p=7.8e-5, PHM2012 p=3.3e-4, IMS p=0.0061.

## Split sensitivity
- TCN, piecewise labels.
- Random ratios: 60/20/20, 80/10/10.
- Time-block split: 70/15/15 chronological per bearing.
- Results: `results/split_sensitivity_*.json`.

## Additional sensitivity analyses
- Global normalization: `results/global_normalization_phm_tcn.json`.
- Validation-bearing rotations: `results/validation_sensitivity_phm_tcn.json`.
- 8-seed statistical summary: `results/v14_summary.json`, `results/cliff_delta_exact.csv`.

## Files
- `scripts/run_converged_rul_study.py`
- `scripts/run_converged_star.py`
- `scripts/run_split_sensitivity.py`
- `scripts/generate_converged_figure.py`
- `scripts/generate_converged_tables.py`
- `results/converged_*.json`
- `results/split_sensitivity_*.json`
- `results/converged_results_summary.md`
- `paper/manuscript_revised_v18_jim.pdf/tex`
- Quantitative prevalence survey counts are not used as evidence in the manuscript.

## STAR audit evidence
- `fulltexts/P02.pdf`, `P03.pdf`, `P05.pdf`, `P06.pdf`, `P16.pdf`: publicly available journal full texts used for the verified STAR examples in Appendix A.
- `prevalence_survey_coding_sheet.csv`: coding sheet with candidate studies and three-level audit status (reported / unclear / unavailable).
- Audit evidence recorded in the coding sheet: P02 (70/30 bearing-level split, normalized RUL), P03 (leave-one-bearing-out, reliability-rate labels), P05 (condition-specific leave-one-bearing-out, piece-wise linear RUL with unclear plateau), P06 (pre-IF plateau and after-IF health percentage labels, source/target bearing holdout), P16 (Cincinnati/IMS 1:1 parity split, PCA degradation index).
- `prevalence_survey_candidates.md`, `prevalence_survey.md`, and `supplementary_prevalence_survey.tex/pdf`: candidate list and survey template.
- Appendix A rows are completed only when the protocol can be verified from the audited full text.
