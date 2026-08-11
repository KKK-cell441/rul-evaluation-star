# Cover Letter (Journal of Intelligent Manufacturing)

Dear Editor,

We are submitting our manuscript entitled:

**Reconsidering Reliability Assessment of Data-Driven Predictive Maintenance: The Impact of Evaluation Protocols on Bearing Remaining Useful Life Prediction**

for consideration in the *Journal of Intelligent Manufacturing*.

This manuscript addresses a reliability-assessment issue in intelligent manufacturing: whether reported bearing remaining useful life (RUL) scores can support deployment-readiness decisions for AI-based predictive maintenance systems. We investigate how evaluation protocols define reliability estimates and whether benchmark scores support deployment-oriented model selection. We provide an empirical framework to improve reliability assessment of data-driven prognostic models in intelligent manufacturing. The paper does not propose a new diagnostic network. Instead, it establishes reproducible evaluation protocols and improves comparability across bearing RUL studies. It is not a critique of specific models or published results; it isolates protocol factors under controlled conditions and focuses on industrial AI reliability and reproducibility. The study is not an architecture ranking exercise; the model classes are controlled to test whether the protocol effect persists across different inductive biases.

We use a complete 2x2 factorial design on three public datasets (XJTU-SY, PHM2012, and IMS) with four model classes, a constant predictor baseline, leave-one-bearing-out cross-validation, random time-step splitting, and eight-seed repeated runs (42-49). We report R2, RMSE, MAE, 95% confidence intervals, exact permutation tests, and Cliff's delta. The results show that evaluation protocol can substantially alter reported performance, and that the effect of random window splitting interacts with label construction.

From a manufacturing intelligence perspective, this is relevant because benchmark scores are used to select models for predictive maintenance deployment. In practical manufacturing systems, RUL models must generalize across different assets, so benchmark choices should be auditable before deployment. A score produced under a protocol that evaluates within-trajectory interpolation may not support a decision about unseen-bearing generalization under the benchmark data. The STAR reporting template summarizes these choices as a recommended reporting aid for authors, reviewers, and industrial reliability teams. We encourage reviewers and authors to request STAR-style reporting during peer review.

All preprocessing, training, split-sensitivity, table, and figure scripts, together with per-fold and per-seed JSON results, are publicly available. Reviewers can reproduce the reported tables without requesting code or data from the authors.

The reproducibility package is available at:

- GitHub: https://github.com/KKK-cell441/rul-evaluation-star
- Zenodo DOI: https://doi.org/10.5281/zenodo.21866359

The manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission.

Thank you for your consideration.

Sincerely,
Zhongkuan Ma
2024212760@nefu.edu.cn
