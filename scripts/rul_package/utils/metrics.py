"""Regression evaluation metrics used in the paper."""

from dataclasses import dataclass
import numpy as np


@dataclass
class RegressionMetrics:
    """Container for all regression metrics reported in the paper.

    Table 2 columns: MAE ↓, RMSE ↓, MAPE ↓, R² ↑, NASA Score ↓
    """
    mae: float
    rmse: float
    mape: float
    r2: float
    nasa_score: float

    def as_dict(self) -> dict:
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MAPE": self.mape,
            "R²": self.r2,
            "NASA_Score": self.nasa_score,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute all 5 metrics used in the paper."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    pct_errors = np.abs(errors) / (np.abs(y_true) + 1e-12) * 100.0

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mape = float(np.mean(pct_errors))

    # R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    # NASA Score  (from the paper)
    #  S = sum_i [ exp(-d_i / 13) - 1 ]  for d_i < 0  (early prediction, penalised)
    #    + sum_i [ exp( d_i / 10) - 1 ]  for d_i >= 0 (late prediction, penalised more)
    d = y_pred - y_true  # positive = late prediction
    nasa_early = np.exp(-d[d < 0] / 13.0) - 1.0
    nasa_late = np.exp(d[d >= 0] / 10.0) - 1.0
    nasa_score = float(np.sum(np.concatenate([nasa_early, nasa_late])))

    return RegressionMetrics(mae=mae, rmse=rmse, mape=mape, r2=r2, nasa_score=nasa_score)
