"""Choose values for the independent variables that push the variable of interest where you want it.

You say how much (or how little) of the variable of interest you want:

    maximize   as high as possible
    minimize   as low as possible
    target     close to a specific value
    range      anywhere inside [low, high]

and optionally lock some variables at fixed values (things you cannot change, such as the weather).
The optimiser then searches the *fitted equation* for the best settings of the remaining variables.

Safeguards, because an equation is only trustworthy where the data are:

* every free variable stays inside the range observed in your data (no extrapolation);
* among settings that are equally good, the one closest to typical conditions wins (tiny tie-break);
* the answer is compared with the real rows: nearest observed row, best observed row, and whether the
  recommended *combination* of values has actually been seen;
* results are reported with the equation's held-out typical error, and `optimize_candidates` checks whether
  the top expressions agree with each other.

The recommendation describes the equation, not cause and effect: changing a variable in real life may
not move the outcome the way the equation (fitted on observations) suggests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

MODES = ("maximize", "minimize", "target", "range")
REG_WEIGHT = 1e-4          # strength of the pull toward typical conditions (tie-break only)
EDGE_FRACTION = 0.02       # within 2% of a bound = "at the edge of the data"
SENSITIVITY_STEP = 0.10    # robustness check: move each variable by 10% of its allowed span
TOUCHY_SHARE = 0.25        # a 10% step that moves the outcome by > 25% of its natural spread = "touchy"
FAR_FROM_DATA = 2.0        # farther than 2x the typical nearest-neighbour gap = "little data nearby"
AGREE_TOL = 0.15           # top expressions agree if their optima differ by < 15% of each variable's span
N_CURVE = 41


@dataclass
class Goal:
    mode: str
    target: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.mode == "target" and (self.target is None or not np.isfinite(self.target)):
            raise ValueError("A target value is required for mode 'target'.")
        if self.mode == "range":
            if self.low is None or self.high is None or not (np.isfinite(self.low) and np.isfinite(self.high)):
                raise ValueError("A low and a high value are required for mode 'range'.")
            if self.low > self.high:
                raise ValueError("The range's low value must not exceed its high value.")

    def gap(self, value: float) -> float:
        """Distance from the goal (0 = achieved). Always 0 for maximize / minimize."""
        if self.mode == "target":
            return abs(float(value) - float(self.target))
        if self.mode == "range":
            return max(0.0, float(self.low) - float(value), float(value) - float(self.high))
        return 0.0

    def describe(self, name: str) -> str:
        if self.mode == "maximize":
            return f"as much {name} as possible"
        if self.mode == "minimize":
            return f"as little {name} as possible"
        if self.mode == "target":
            return f"{name} as close as possible to {self.target:.4g}"
        return f"{name} between {self.low:.4g} and {self.high:.4g}"


@dataclass
class OptimizationResult:
    goal: Goal
    target_name: str
    names: List[str]
    values: Dict[str, float]            # recommended value of every variable (locked ones included)
    locked: Dict[str, float]
    bounds: Dict[str, Tuple[float, float]]
    typical: Dict[str, float]           # medians (locked variables at their locked value)
    predicted: float
    typical_prediction: float
    improvement: float                  # predicted - typical_prediction
    gap: float                          # distance from the goal (target / range), else 0
    reached: bool                       # goal achieved by the equation (target / range); True for max / min
    rmse: float
    edge: Dict[str, str]                # variable -> "lower edge" | "upper edge" for values pinned at a data bound
    sensitivity: List[dict]             # per free variable: change in outcome if the value moves down / up by 10% of its span
    touchy: List[str]                   # free variables where a 10% step moves the outcome a lot
    inert: List[str] = field(default_factory=list)   # free variables the equation barely responds to (any value will do)
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(repr=False, default_factory=dict)
    scale: float = 1.0                  # natural spread (std) of the outcome under the equation
    nearest_distance: Optional[float] = None
    typical_gap: Optional[float] = None
    far_from_data: bool = False
    nearest_value: Optional[float] = None       # observed outcome at the nearest real row
    best_observed: Optional[float] = None       # best observed outcome according to the goal
    observed_in_range: Optional[float] = None   # share of observed rows already inside the range (mode 'range')
    free: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def predicted_low(self) -> float:
        return self.predicted - self.rmse

    @property
    def predicted_high(self) -> float:
        return self.predicted + self.rmse


def _cost_of_outcome(goal: Goal, f: np.ndarray, scale: float) -> np.ndarray:
    if goal.mode == "maximize":
        return -f / scale
    if goal.mode == "minimize":
        return f / scale
    if goal.mode == "target":
        return ((f - goal.target) / scale) ** 2
    return (np.maximum(0.0, np.maximum(goal.low - f, f - goal.high)) / scale) ** 2


def optimize(
    predict: Callable[[np.ndarray], np.ndarray], X_ref: np.ndarray, names: Sequence[str], goal: Goal,
    target_name: str = "outcome", locked: Optional[Dict[str, float]] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None, y_ref: Optional[np.ndarray] = None,
    rmse: float = 0.0, seed: int = 0,
) -> OptimizationResult:
    """Find the variable settings that best meet `goal` according to the fitted equation `predict`."""
    goal.validate()
    X = np.asarray(X_ref, float)
    names = list(names)
    if X.ndim != 2 or X.shape[1] != len(names) or X.shape[0] < 5:
        raise ValueError("X_ref must be a 2-D array with one column per variable name and at least 5 rows.")
    notes: List[str] = []
    data_lo, data_hi = X.min(axis=0), X.max(axis=0)
    med = np.median(X, axis=0)

    lo, hi = data_lo.copy(), data_hi.copy()
    for k, (a, b) in (bounds or {}).items():
        if k not in names:
            raise ValueError(f"Unknown variable in bounds: {k!r}")
        j = names.index(k)
        a, b = float(a), float(b)
        if a > b:
            raise ValueError(f"Lower bound exceeds upper bound for {k!r}.")
        lo[j], hi[j] = max(a, data_lo[j]), min(b, data_hi[j])
        if lo[j] > hi[j]:
            raise ValueError(f"The allowed range for {k!r} lies outside the range seen in your data.")
        if (a < data_lo[j]) or (b > data_hi[j]):
            notes.append(f"{k}: the allowed range was cut back to the range seen in your data.")

    base = med.copy()
    locked_clean: Dict[str, float] = {}
    for k, v in (locked or {}).items():
        if k not in names:
            raise ValueError(f"Unknown variable in locked values: {k!r}")
        j = names.index(k)
        v = float(np.clip(float(v), data_lo[j], data_hi[j]))
        base[j] = v
        locked_clean[k] = v
    free_idx = [j for j in range(len(names)) if names[j] not in locked_clean and hi[j] > lo[j]]
    for j in range(len(names)):                       # typical value, kept inside the allowed range; no room = fixed
        if names[j] not in locked_clean:
            base[j] = float(np.clip(base[j], lo[j], hi[j]))
    span = np.where(hi > lo, hi - lo, 1.0)

    ref_pred = np.asarray(predict(X), float)
    scale = float(np.std(ref_pred))
    if scale == 0 and y_ref is not None:
        scale = float(np.std(y_ref))
    scale = scale or 1.0

    def rows(Xfree: np.ndarray) -> np.ndarray:        # Xfree: (k_free, S)  ->  (S, n_vars)
        S = Xfree.shape[1]
        M = np.tile(base, (S, 1))
        M[:, free_idx] = Xfree.T
        return M

    def cost(Xfree: np.ndarray) -> np.ndarray:
        M = rows(np.atleast_2d(Xfree))
        f = np.asarray(predict(M), float)
        z = (M[:, free_idx] - base[free_idx]) / span[free_idx]
        return _cost_of_outcome(goal, f, scale) + REG_WEIGHT * np.mean(z ** 2, axis=1)

    typical_pred = float(predict(base.reshape(1, -1))[0])
    if free_idx:
        box = [(float(lo[j]), float(hi[j])) for j in free_idx]
        de = differential_evolution(cost, box, seed=seed, popsize=20, maxiter=300, tol=1e-8, atol=0.0,
                                    mutation=(0.5, 1.0), recombination=0.7, vectorized=True, updating="deferred",
                                    polish=False)
        x_best, c_best = np.asarray(de.x, float), float(cost(np.asarray(de.x).reshape(-1, 1))[0])
        try:                                           # local polish, accepted only if it genuinely improves
            loc = minimize(lambda x: float(cost(x.reshape(-1, 1))[0]), x_best, method="L-BFGS-B", bounds=box)
            if np.all(np.isfinite(loc.x)) and float(loc.fun) < c_best:
                x_best = np.clip(loc.x, [b[0] for b in box], [b[1] for b in box])
        except Exception:
            pass
        best_row = rows(x_best.reshape(-1, 1))[0]
    else:
        best_row = base.copy()
        notes.append("Every variable is locked (or has no room to move), so there is nothing to optimise.")
    predicted = float(predict(best_row.reshape(1, -1))[0])

    values = {names[j]: float(best_row[j]) for j in range(len(names))}
    typical = {names[j]: float(base[j]) for j in range(len(names))}
    bounds_out = {names[j]: (float(lo[j]), float(hi[j])) for j in range(len(names))}

    edge: Dict[str, str] = {}
    for j in free_idx:
        if best_row[j] <= lo[j] + EDGE_FRACTION * span[j]:
            edge[names[j]] = "lower edge"
        elif best_row[j] >= hi[j] - EDGE_FRACTION * span[j]:
            edge[names[j]] = "upper edge"

    sens, touchy, inert, curves = [], [], [], {}
    for j in free_idx:
        step = SENSITIVITY_STEP * span[j]
        dn, up = best_row.copy(), best_row.copy()
        dn[j], up[j] = max(lo[j], best_row[j] - step), min(hi[j], best_row[j] + step)
        f_dn, f_up = (float(predict(r.reshape(1, -1))[0]) for r in (dn, up))
        sens.append({"variable": names[j], "step": float(step), "change_if_down": f_dn - predicted,
                     "change_if_up": f_up - predicted})
        if max(abs(f_dn - predicted), abs(f_up - predicted)) > TOUCHY_SHARE * scale:
            touchy.append(names[j])
        grid = np.linspace(lo[j], hi[j], N_CURVE)
        G = np.tile(best_row, (N_CURVE, 1))
        G[:, j] = grid
        curves[names[j]] = (grid, np.asarray(predict(G), float))
        if float(np.ptp(curves[names[j]][1])) < 0.01 * scale:
            inert.append(names[j])

    nearest_d = typical_gap = nearest_val = best_obs = in_range = None
    far = False
    if free_idx:
        Z = (X[:, free_idx] - best_row[free_idx]) / span[free_idx]
        d = np.sqrt(np.mean(Z ** 2, axis=1))
        i_near = int(np.argmin(d))
        nearest_d = float(d[i_near])
        sample = X if len(X) <= 300 else X[np.random.default_rng(0).choice(len(X), 300, replace=False)]
        S = (sample[:, free_idx] / span[free_idx])
        D = np.sqrt(np.mean((S[:, None, :] - S[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(D, np.inf)
        typical_gap = float(np.median(D.min(axis=1)))
        far = bool(typical_gap > 0 and nearest_d > FAR_FROM_DATA * typical_gap)
        if y_ref is not None:
            y = np.asarray(y_ref, float).ravel()
            nearest_val = float(y[i_near])
            if goal.mode == "maximize":
                best_obs = float(y.max())
            elif goal.mode == "minimize":
                best_obs = float(y.min())
            elif goal.mode == "target":
                best_obs = float(y[int(np.argmin(np.abs(y - goal.target)))])
            else:
                best_obs = float(y[int(np.argmin([goal.gap(v) for v in y]))])
                in_range = float(np.mean([goal.gap(v) == 0.0 for v in y]))

    gap = goal.gap(predicted)
    tol = 1e-3 * scale
    return OptimizationResult(
        goal=goal, target_name=target_name, names=names, values=values, locked=locked_clean, bounds=bounds_out,
        typical=typical, predicted=predicted, typical_prediction=typical_pred, improvement=predicted - typical_pred,
        gap=float(gap), reached=bool(gap <= tol), rmse=float(rmse), edge=edge, sensitivity=sens, touchy=touchy, inert=inert,
        curves=curves, scale=scale, nearest_distance=nearest_d, typical_gap=typical_gap, far_from_data=far,
        nearest_value=nearest_val, best_observed=best_obs, observed_in_range=in_range,
        free=[names[j] for j in free_idx], notes=notes)


@dataclass
class OptimizationComparison:
    results: List[OptimizationResult]
    disagreement: float                 # largest difference between any two optima, as a share of the variable's span
    agree: bool
    per_variable: Dict[str, List[float]]


def optimize_candidates(
    candidates, X_ref: np.ndarray, names: Sequence[str], goal: Goal, target_name: str = "outcome",
    locked: Optional[Dict[str, float]] = None, bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    y_ref: Optional[np.ndarray] = None, seed: int = 0,
) -> OptimizationComparison:
    """Run the optimiser on every candidate expression and check whether they point to the same settings."""
    if not candidates:
        raise ValueError("No candidate expressions to optimise.")
    results = [optimize(c.predict, X_ref, names, goal, target_name, locked, bounds, y_ref,
                        rmse=c.validation["rmse"], seed=seed) for c in candidates]
    free = results[0].free
    per = {k: [r.values[k] for r in results] for k in free}
    worst = 0.0
    for k in free:
        lo, hi = results[0].bounds[k]
        vals = np.asarray(per[k])
        worst = max(worst, float((vals.max() - vals.min()) / (hi - lo)))
    return OptimizationComparison(results=results, disagreement=worst, agree=bool(worst < AGREE_TOL), per_variable=per)
