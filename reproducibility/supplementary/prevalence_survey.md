# Prevalence Survey (20 RUL Papers, 2019-2024)

## Scope
We reviewed 20 recent bearing RUL prediction papers published in journals including Mechanical Systems and Signal Processing, IEEE Transactions on Industrial Informatics, IEEE Transactions on Instrumentation and Measurement, and IEEE Access.

## Coding Rules
- Piecewise label: the paper uses an 80% plateau piecewise linear RUL label.
- Random time-step split: sliding-window samples are randomly assigned to train/validation/test without per-bearing holdout.
- Combined: both piecewise labels and random time-step splitting are used in the reported main evaluation.
- Constant baseline: the paper reports a constant predictor or another trivial baseline for calibration.

## Counts
- Piecewise labels: 18/20 (90%)
- Random time-step split: 16/20 (80%)
- Combined piecewise + random split: 14/20 (70%)
- Constant predictor baseline reported: 0/20 (0%)

## Interpretation
The survey supports the claim that the protocol family reproduced in this study is common in the recent bearing RUL literature. It does not imply that every paper with a high R2 value has the same protocol.
