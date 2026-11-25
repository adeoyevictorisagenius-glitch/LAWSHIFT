"""Residual diagnostics: are the equation's mistakes the harmless kind?

A good model's errors should be small, centred on zero, similar in size everywhere and roughly bell-shaped.
These checks run on HELD-OUT rows only, so they describe how the equation behaves on data it has not seen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import stats

HETERO_RHO = 0.3        # |Spearman rho| between |error| and prediction above this (and significant) = errors change with size
BIAS_SHARE = 0.1        # mean error larger than 10% of the RMSE = systematically high or low


@dataclass
class ResidualDiagnostics:
    n: int
    mean_error: float           # mean of (actual - predicted); positive = the equation tends to under-predict
    rmse: float
    bias_share: float           # |mean_error| / rmse
    biased: bool
    shapiro_p: float | None     # normality of the errors
    errors_look_normal: bool | None
    hetero_rho: float | None    # Spearman correlation of |error| with the prediction
    hetero_p: float | None
    errors_grow: bool           # errors get bigger (or smaller) as the prediction grows
    interval95: float           # +/- 1.96 * std of the errors
    worst_rows: List[int]       # positions (in the supplied arrays) of the three largest absolute errors


def residual_diagnostics(y_true, y_pred) -> ResidualDiagnostics:
    y = np.asarray(y_true, float).ravel()
    p = np.asarray(y_pred, float).ravel()
    if y.shape != p.shape or y.size < 5:
        raise ValueError("Need at least 5 matching actual/predicted values.")
    r = y - p
    rmse = float(np.sqrt(np.mean(r ** 2)))
    mean_err = float(r.mean())
    share = abs(mean_err) / rmse if rmse > 0 else 0.0

    sh_p = normal = None
    if y.size >= 8 and np.std(r) > 0:
        sh_p = float(stats.shapiro(r[:5000]).pvalue)
        normal = sh_p >= 0.05
    rho = hp = None
    grows = False
    if np.std(p) > 0 and np.std(np.abs(r)) > 0:
        res = stats.spearmanr(p, np.abs(r))
        rho, hp = float(res.statistic), float(res.pvalue)
        grows = bool(abs(rho) >= HETERO_RHO and hp < 0.05)
    worst = [int(i) for i in np.argsort(-np.abs(r))[:3]]
    return ResidualDiagnostics(
        n=int(y.size), mean_error=mean_err, rmse=rmse, bias_share=float(share), biased=bool(share > BIAS_SHARE),
        shapiro_p=sh_p, errors_look_normal=normal, hetero_rho=rho, hetero_p=hp, errors_grow=grows,
        interval95=float(1.96 * np.std(r, ddof=1)), worst_rows=worst)
