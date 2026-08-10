# Cover Letter (Measurement)

Dear Editor,

We are submitting our manuscript entitled:

**Reconsidering Reliability Assessment of Data-Driven Predictive Maintenance: The Impact of Evaluation Protocols on Bearing Remaining Useful Life Prediction**

for consideration in *Measurement*.

This manuscript addresses an under-examined methodological issue: whether reported bearing remaining useful life (RUL) improvements reflect model capability or evaluation design. The paper does not propose a new diagnostic network. Instead, it evaluates the combined effects of label construction and temporal splitting under a controlled protocol comparison. The contribution is to establish reproducible evaluation protocols and improve comparability across bearing RUL studies, not to critique specific published results.

We use a complete 2x2 factorial design on three public datasets (XJTU-SY, PHM2012, and IMS) with four model classes (Linear Regression, LSTM, TCN, and PatchTST), a constant predictor baseline, leave-one-bearing-out cross-validation, random time-step splitting, and three-seed repeated runs. We report R2, RMSE, MAE, 95% confidence intervals, and Cliff's delta.

The key finding is that evaluation protocol can substantially alter reported performance. In particular, random window splitting evaluates within-trajectory interpolation rather than new-bearing generalization, and its effect interacts with label construction. From a Measurement perspective, this is a study of the reliability of measurement results and benchmark protocol design: reported performance cannot be interpreted without a clearly specified cross-bearing evaluation protocol. The paper therefore focuses on reproducible evaluation, comparability across studies, and the uncertainty of reported performance.

We also propose the STAR evaluation template, which summarizes the experimentally supported reporting requirements into an auditable checklist: split integrity, target integrity, anchor baseline, and reporting completeness.

The reproducibility package is available at:

- GitHub: https://github.com/KKK-cell441/rul-evaluation-star
- Zenodo DOI: https://doi.org/10.5281/zenodo.21866359

The manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission.

Thank you for your consideration.

Sincerely,
Zhongkuan Ma
Northeast Forestry University, Harbin, China
2024212760@nefu.edu.cn
