# Reproducibility Notes (v5)

## Datasets and preprocessing
- XJTU-SY: `scripts/preprocess_xjtu15.py` (full public dataset; v5 uses 8 bearings from conditions 1-2).
- PHM2012: `scripts/preprocess_public_rul.py`.
- IMS: `scripts/preprocess_public_rul.py`.
- Feature extraction uses the dbca-net preprocessing module with 2,560-sample windows, 10 statistical/spectral features, and history length 10.

## Converged protocol
- Models: LinearRegression, StatLSTM, TCN, Constant.
- Labels: linear, piecewise (80% plateau).
- Evaluation: LOOCV and random 70/15/15 time-step splits.
- Training: AdamW, MSE, cosine annealing, batch size 1024, early stopping patience 30.
- Max epochs: 100 (XJTU-SY, PHM2012), 60 (IMS).
- Random seeds: 42, 43, 44.

## Split sensitivity
- TCN, piecewise labels.
- Random ratios: 60/20/20, 80/10/10.
- Time-block split: 70/15/15 chronological per bearing.
- Results: `results/split_sensitivity_*.json`.

## Files
- `scripts/run_converged_rul_study.py`
- `scripts/run_converged_star.py`
- `scripts/run_split_sensitivity.py`
- `results/converged_*.json`
- `results/split_sensitivity_*.json`
- `results/converged_results_summary.md`
- `paper/manuscript_revised_v5.pdf/tex`
- Quantitative prevalence survey counts are not used as evidence in the manuscript.

## STAR audit evidence
- `fulltexts/P02.pdf`, `P03.pdf`, `P05.pdf`, `P06.pdf`, `P16.pdf`: publicly available journal full texts used for the verified STAR examples in Appendix A.
- `prevalence_survey_coding_sheet.csv`: coding sheet with candidate studies and three-level audit status (reported / unclear / unavailable).
- `prevalence_survey_candidates.md`, `prevalence_survey.md`, and `supplementary_prevalence_survey.tex/pdf`: candidate list and survey template.
- Appendix A rows are completed only when the protocol can be verified from the audited full text.
