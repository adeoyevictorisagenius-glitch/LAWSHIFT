"""Fitness evaluation: prediction error + complexity penalty (lower is better).

fitness = NRMSE(y, prediction) + parsimony * (number of nodes)

NRMSE is RMSE divided by the standard deviation of y, so it is scale-free: 0 is a
perfect fit and ~1 means "no better than predicting the mean".

Numeric constants inside a candidate are tuned with a few Levenberg-Marquardt
steps (SciPy). The tuned values are written back to the tree
(Lamarckian evolution), so evolution searches *structure* while SciPy fine-tunes
the numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import least_squares

from .expressions import Node, evaluate, get_constants, size

CONST_BOUND = 1e3
MAX_TUNED_CONSTANTS = 8
BAD_FITNESS = 1e9


@dataclass
class Individual:
    tree: Node
    loss: float = float("inf")      # NRMSE on training data
    fitness: float = float("inf")   # loss + complexity penalty
    size: int = 0


def optimize_constants(tree: Node, X: np.ndarray, y: np.ndarray, max_nfev: int = 20) -> None:
    """Tune Const nodes in place. Silently leaves the tree unchanged on failure."""
    consts = get_constants(tree)
    if not consts or len(consts) > MAX_TUNED_CONSTANTS or len(y) < len(consts):
        return
    original = np.array([c.value for c in consts], dtype=float)

    def set_values(v: np.ndarray) -> None:
        for c, val in zip(consts, v):
            c.value = float(val)

    def residual(v: np.ndarray) -> np.ndarray:
        set_values(v)
        return evaluate(tree, X) - y

    try:
        start = np.clip(original, -CONST_BOUND, CONST_BOUND)
        before = float(np.sum(residual(start) ** 2))
        res = least_squares(residual, start, max_nfev=max_nfev, method="lm")
        # LM is unbounded, so keep constants inside the allowed range
        if np.all(np.isfinite(res.x)) and np.all(np.abs(res.x) <= CONST_BOUND) \
                and float(np.sum(res.fun ** 2)) <= before:
            set_values(res.x)
        else:
            set_values(start)
    except Exception:  # numerical failure: keep the original constants
        set_values(original)


def compute_loss(tree: Node, X: np.ndarray, y: np.ndarray) -> float:
    """NRMSE of the tree on (X, y). Returns inf if the prediction is unusable."""
    pred = evaluate(tree, X)
    if not np.all(np.isfinite(pred)):
        return float("inf")
    sd = float(np.std(y))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return rmse / sd if sd > 0 else rmse


def evaluate_individual(
    tree: Node,
    X: np.ndarray,
    y: np.ndarray,
    parsimony: float = 0.001,
    tune_constants: bool = True,
) -> Individual:
    if tune_constants:
        optimize_constants(tree, X, y)
    loss = compute_loss(tree, X, y)
    n = size(tree)
    fitness = loss + parsimony * n if np.isfinite(loss) else BAD_FITNESS
    return Individual(tree=tree, loss=loss, fitness=float(fitness), size=n)
