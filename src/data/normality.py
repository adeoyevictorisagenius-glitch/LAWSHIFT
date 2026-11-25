"""Describe ordinary tabular data and check how close each column is to a normal (bell-curve) distribution.

Everything is computed from the supplied numbers; no thresholds are tuned to any particular dataset.

Verdicts combine a *test* (Shapiro-Wilk) with the *shape* (skewness and excess kurtosis). Tests alone are
misleading: with many rows they flag harmless wiggles, with few rows they miss real problems. So the verdict
asks both "is the test happy?" and "is the shape actually bell-like?".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

SHAPIRO_MAX_N = 5000          # SciPy's p-value is only reliable up to about this size; larger columns are subsampled
SKEW_MILD, SKEW_STRONG = 0.5, 1.0
KURT_MILD = 1.0
ALPHA = 0.05

VERDICT_LABELS = {
    "normal": "looks normal",
    "nearly_normal": "nearly normal",
    "skewed_right": "skewed right",
    "skewed_left": "skewed left",
    "heavy_tails": "heavy tails",
    "light_tails": "flat / light tails",
    "not_normal": "not normal",
    "too_few": "too few values",
}


@dataclass
class ColumnProfile:
    name: str
    n: int
    mean: float
    std: float
    median: float
    minimum: float
    maximum: float
    q1: float
    q3: float
    skew: float
    excess_kurtosis: float
    shapiro_p: Optional[float]
    dagostino_p: Optional[float]
    within_1sd: float            # share of values within 1 / 2 / 3 standard deviations of the mean
    within_2sd: float
    within_3sd: float
    outliers_z3: int             # |z| > 3
    outliers_iqr: int            # outside 1.5 * IQR fences
    verdict: str                 # key of VERDICT_LABELS
    verdict_text: str            # one plain-language sentence
    suggestion: str              # what to do about it ("" if nothing)

    @property
    def label(self) -> str:
        return VERDICT_LABELS[self.verdict]


def normality_verdict(skew: float, kurt: float, p: Optional[float], n: int, positive: bool) -> Tuple[str, str, str]:
    """Return (code, plain sentence, suggestion)."""
    if n < 8 or p is None:
        return "too_few", "There are too few values to say anything reliable about the shape.", ""
    mild_shape = abs(skew) < SKEW_MILD and abs(kurt) < KURT_MILD
    if p >= ALPHA and mild_shape:
        return "normal", "The values look consistent with a normal (bell-shaped) distribution.", ""
    if mild_shape:
        return ("nearly_normal",
                "The shape is close to a bell curve; the formal test is just picky at this number of rows, "
                "so most normal-data methods will still work well.", "")
    if abs(skew) >= SKEW_STRONG:
        side = "right" if skew > 0 else "left"
        hint = ("A log transform often pulls a long right tail in and makes the column more bell-like."
                if skew > 0 and positive else
                "Consider a transform, or use methods that do not assume normality (medians, rank-based tests).")
        return (f"skewed_{side}",
                f"The distribution is clearly skewed to the {side}: a long tail stretches {side} while most values bunch up "
                f"on the other side.", hint)
    if kurt >= KURT_MILD:
        return ("heavy_tails",
                "The middle looks bell-like but there are more extreme values than a normal curve would give.",
                "Check the extreme values: they may be genuine, or they may be recording errors worth a second look.")
    if kurt <= -KURT_MILD:
        return ("light_tails",
                "The distribution is flatter than a bell curve, with fewer extreme values than normal data would show "
                "(it may be close to uniform).", "")
    return ("not_normal",
            "The shape departs from a bell curve in a mild but consistent way (see skew and kurtosis).",
            "Medians and rank-based methods are safer than means here.")


def profile_column(values, name: str = "x") -> ColumnProfile:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        raise ValueError(f"Column {name!r} has no finite values.")
    mean, std = float(x.mean()), float(x.std())
    q1, med, q3 = (float(v) for v in np.percentile(x, [25, 50, 75]))
    if std > 0 and n >= 3:
        skew = float(stats.skew(x))
        kurt = float(stats.kurtosis(x))
        z = np.abs((x - mean) / std)
        w1, w2, w3 = (float(np.mean(z <= k)) for k in (1, 2, 3))
        n_z3 = int(np.sum(z > 3))
    else:
        skew = kurt = 0.0
        w1 = w2 = w3 = 1.0
        n_z3 = 0
    iqr = q3 - q1
    n_iqr = int(np.sum((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr))) if iqr > 0 else 0

    p_sh = p_dag = None
    if n >= 8 and std > 0:
        sample = x if n <= SHAPIRO_MAX_N else np.random.default_rng(0).choice(x, SHAPIRO_MAX_N, replace=False)
        p_sh = float(stats.shapiro(sample).pvalue)
        p_dag = float(stats.normaltest(x).pvalue)
    verdict, text, hint = normality_verdict(skew, kurt, p_sh, n, positive=bool(x.min() > 0))
    return ColumnProfile(
        name=name, n=n, mean=mean, std=std, median=med, minimum=float(x.min()), maximum=float(x.max()), q1=q1, q3=q3,
        skew=skew, excess_kurtosis=kurt, shapiro_p=p_sh, dagostino_p=p_dag, within_1sd=w1, within_2sd=w2, within_3sd=w3,
        outliers_z3=n_z3, outliers_iqr=n_iqr, verdict=verdict, verdict_text=text, suggestion=hint)


def profile_table(df: pd.DataFrame) -> Tuple[List[ColumnProfile], pd.DataFrame]:
    """Profiles for every numeric column plus a compact summary table."""
    profiles = [profile_column(df[c].to_numpy(), c) for c in df.columns]
    table = pd.DataFrame([{
        "variable": p.name, "mean": p.mean, "std": p.std, "min": p.minimum, "median": p.median, "max": p.maximum,
        "skew": p.skew, "excess kurtosis": p.excess_kurtosis, "Shapiro-Wilk p": p.shapiro_p,
        "shape": p.label} for p in profiles])
    return profiles, table


def qq_points(values) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normal Q-Q data: (theoretical quantiles, ordered values, correlation of the Q-Q line)."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    (osm, osr), (_slope, _icpt, r) = stats.probplot(x, dist="norm")
    return np.asarray(osm), np.asarray(osr), float(r)


def strongest_pairs(corr: pd.DataFrame, k: int = 3) -> List[Tuple[str, str, float]]:
    """The k strongest pairwise correlations (by absolute value), each pair once."""
    cols = list(corr.columns)
    pairs = [(cols[i], cols[j], float(corr.iat[i, j])) for i in range(len(cols)) for j in range(i + 1, len(cols))
             if np.isfinite(corr.iat[i, j])]
    return sorted(pairs, key=lambda t: -abs(t[2]))[:k]
