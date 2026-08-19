"""RUL labeling strategies with configurable piecewise / linear / exponential."""

from typing import Literal
import numpy as np

RULStrategy = Literal["linear", "piecewise", "exponential"]


def make_rul_labels(
    n_windows: int,
    strategy: RULStrategy = "linear",
    piecewise_cutoff: float = 0.8,
    exponential_alpha: float = 3.0,
) -> np.ndarray:
    """Generate RUL label array for a single bearing's full lifetime.

    Parameters
    ----------
    n_windows : int
        Number of time windows in this bearing's run-to-failure trajectory.
    strategy : RULStrategy
        Which labeling scheme to use.
    piecewise_cutoff : float
        Fraction of life where RUL stays at 1.0 before linear decay
        (only used when strategy='piecewise').
    exponential_alpha : float
        Controls curvature of exponential degradation (only when strategy='exponential').

    Returns
    -------
    np.ndarray, shape (n_windows,)
        RUL values normalized to [0, 1].  1.0 = start of life, 0.0 = failure.

    Notes
    -----
    The returned values represent **remaining-life proportion** (1 → 0),
    which is the convention used in the DBCA-Net paper Eq. (3).
    """
    if n_windows <= 0:
        return np.array([], dtype=np.float32)

    t = np.arange(n_windows, dtype=np.float64)

    if strategy == "linear":
        # RUL = 1 - t / (n - 1)   →  1.0 at start, 0.0 at failure
        # Every window contributes real degradation signal — no plateau.
        rul = 1.0 - t / (n_windows - 1)

    elif strategy == "piecewise":
        # Original paper Eq. (3): plateau at 1.0 for first `cutoff` fraction,
        # then linear decay over the remaining life.
        # ⚠ This can cause data leakage / inflated R² — kept here for comparison.
        rul = np.ones(n_windows, dtype=np.float64)
        cutoff_idx = int(n_windows * piecewise_cutoff)
        n_decay = n_windows - cutoff_idx
        if n_decay > 1:
            rul[cutoff_idx:] = np.linspace(1.0, 0.0, n_decay)
        elif n_decay == 1:
            rul[cutoff_idx] = 0.0  # last window

    elif strategy == "exponential":
        # HI(t) = 1 - (exp(alpha * t_norm) - 1) / (exp(alpha) - 1)
        # Early degradation is slow, accelerates near failure.
        # Closer to physical fatigue crack propagation.
        t_norm = t / (n_windows - 1)
        exp_alpha = np.exp(exponential_alpha)
        rul = 1.0 - (np.exp(exponential_alpha * t_norm) - 1.0) / (exp_alpha - 1.0)

    else:
        raise ValueError(f"Unknown RUL strategy: {strategy}")

    # Clip numerical noise
    rul = np.clip(rul, 0.0, 1.0).astype(np.float32)
    return rul
