"""Error metrics used for training and validation."""
from __future__ import annotations

import numpy as np


def _check(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot compute a metric on empty arrays")
    return y_true, y_pred


def mse(y_true, y_pred) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination. Returns 0.0 if y_true is constant."""
    y_true, y_pred = _check(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def nrmse(y_true, y_pred) -> float:
    """RMSE normalised by the standard deviation of y_true (scale-free)."""
    y_true, y_pred = _check(y_true, y_pred)
    sd = np.std(y_true)
    return float(rmse(y_true, y_pred) / sd) if sd > 0 else float(rmse(y_true, y_pred))
