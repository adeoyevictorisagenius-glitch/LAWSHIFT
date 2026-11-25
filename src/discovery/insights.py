"""Plain-language insights about a discovered equation.

Everything here is computed numerically from the fitted equation and the data - no
fixed text is attached to any dataset:

* importance  - permutation importance on HELD-OUT rows (how much predictions degrade
                when a variable's values are shuffled), normalised to shares.
* direction   - partial-effect curve: vary one variable over its observed 5th-95th
                percentile range, hold the others at their medians.
* shape       - whether the effect is roughly linear, flattening (diminishing returns)
                or accelerating.
* interaction - whether one variable's effect changes strongly with another.
* elasticity  - % change in the prediction for a +10% nudge to one variable (others at their medians).
* turning point / 80% point - where a sweet spot sits, and where most of a flattening gain has been collected.
* swing       - how much the prediction moves across the variable's 5th-95th percentile range, against the
                natural spread of the outcome.

These describe how the FITTED EQUATION behaves inside the observed data range. They are
associations, not proof of cause and effect, and say nothing outside that range.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

N_GRID = 21
N_REPEATS = 5
MEANINGFUL_SHARE = 0.05      # a variable needs >= 5% of total influence to be called a driver
NEGLIGIBLE_SHARE = 0.02

CAVEAT = ("These statements describe how the fitted equation behaves inside the observed data range. "
          "They show association in this dataset, not proven cause and effect, and do not apply outside that range.")


@dataclass
class VariableInsight:
    name: str
    used: bool                 # appears in the equation text
    importance: float          # share of total permutation importance (0..1)
    direction: str             # increasing | decreasing | rises then falls | falls then rises | oscillating | flat
    shape: str                 # roughly linear | flattens out | accelerates | ""
    low: float                 # 25th percentile of the variable
    high: float                # 75th percentile
    delta: float               # change in prediction from low to high (others at median)
    grid: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    curve: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    text: str = ""
    elasticity: Optional[float] = None     # % change in prediction for +10% in this variable (None if not meaningful)
    turning_x: Optional[float] = None      # location of the peak / dip, if the effect has one
    gain80_x: Optional[float] = None       # value of the variable by which 80% of its total effect has been reached
    swing: float = 0.0                     # peak-to-peak change of the prediction over the 5th-95th percentile range
    swing_share: float = 0.0               # swing as a multiple of std(outcome)


@dataclass
class ModelInsights:
    headline: str
    variables: List[VariableInsight]
    interaction: Optional[str]
    accuracy: str
    caveat: str = CAVEAT
    rmse: float = 0.0                      # held-out RMSE of the equation
    mean_abs_target: float = 0.0           # mean |outcome| on held-out rows (scale for 'x% of typical')
    target_std: float = 0.0                # std of the outcome on held-out rows

    def table(self) -> List[dict]:
        """One row per variable with the technical quantities behind the plain-language statements."""
        rows = []
        for v in sorted(self.variables, key=lambda v: -v.importance):
            rows.append({
                "variable": v.name, "influence share": v.importance, "in equation": v.used,
                "direction": v.direction, "shape": v.shape or "-",
                "change, 25th to 75th pct": v.delta, "elasticity (+10%) [%]": v.elasticity,
                "peak / dip at": v.turning_x, "80% of effect by": v.gain80_x, "swing / std(outcome)": v.swing_share})
        return rows

    def bullets(self) -> List[str]:
        out = [self.headline]
        out += [v.text for v in sorted(self.variables, key=lambda v: -v.importance)]
        if self.interaction:
            out.append(self.interaction)
        out.append(self.accuracy)
        return out


def _g(x: float) -> str:
    return f"{x:.3g}"


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _classify(curve: np.ndarray, scale: float):
    """Return (direction, shape) for a partial-effect curve."""
    rng_ = float(np.ptp(curve))
    if rng_ < 0.02 * scale:
        return "flat", ""
    d = np.diff(curve)
    tol = 0.01 * rng_
    signs = np.sign(np.where(np.abs(d) < tol, 0.0, d))
    signs = signs[signs != 0]
    turns = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    if turns >= 2:
        return "oscillating", ""
    if turns == 1:
        return ("rises then falls" if signs[0] > 0 else "falls then rises"), ""
    direction = "increasing" if signs[0] > 0 else "decreasing"
    mid = len(curve) // 2
    s1, s2 = abs(curve[mid] - curve[0]), abs(curve[-1] - curve[mid])
    if s1 + s2 < 1e-12:
        return direction, ""
    ratio = s2 / s1 if s1 > 0 else np.inf
    shape = "flattens out" if ratio < 0.7 else ("accelerates" if ratio > 1.4 else "roughly linear")
    return direction, shape


def _variable_text(v: VariableInsight, target: str) -> str:
    if not v.used:
        return (f"{v.name} is not used by this equation, so the search found no need for it "
                f"(this does not prove {v.name} is unrelated to {target}).")
    if v.importance < NEGLIGIBLE_SHARE or v.direction == "flat":
        return f"{v.name} appears in the equation but has a negligible effect on its predictions."
    span = f"from its 25th percentile ({_g(v.low)}) to its 75th ({_g(v.high)})"
    if v.direction in ("increasing", "decreasing"):
        verb = "rises" if v.delta > 0 else "falls"
        s = f"As {v.name} goes {span}, predicted {target} {verb} by about {_g(abs(v.delta))}"
        if v.shape == "flattens out":
            s += ", with diminishing returns (the effect flattens at higher values)"
        elif v.shape == "accelerates":
            s += ", and the effect gets stronger at higher values"
        elif v.shape == "roughly linear":
            s += ", roughly in proportion"
        return s + "."
    if v.direction == "rises then falls":
        return (f"{v.name} has a peak: predicted {target} first rises and then falls as {v.name} increases "
                f"(a sweet spot, not 'more is better').")
    if v.direction == "falls then rises":
        return (f"{v.name} has a dip: predicted {target} first falls and then rises as {v.name} increases.")
    return f"Predicted {target} oscillates repeatedly as {v.name} changes (a periodic pattern)."


def explain_model(
    predict: Callable[[np.ndarray], np.ndarray],
    X_ref: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    names: Sequence[str], target: str, used_indices: Sequence[int], seed: int = 0,
) -> ModelInsights:
    """Compute insights. X_ref (training rows) sets variable ranges/medians; X_val/y_val (held out) score importance."""
    X_ref, X_val, y_val = np.asarray(X_ref, float), np.asarray(X_val, float), np.asarray(y_val, float).ravel()
    n_vars = X_ref.shape[1]
    scale = float(np.std(y_val)) or 1.0
    med = np.median(X_ref, axis=0)
    rng = np.random.default_rng(seed)

    base = _rmse(y_val, predict(X_val))
    inc = np.zeros(n_vars)
    for j in range(n_vars):
        scores = []
        for _ in range(N_REPEATS):
            Xp = X_val.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            scores.append(_rmse(y_val, predict(Xp)))
        inc[j] = max(0.0, float(np.mean(scores)) - base)
    share = inc / inc.sum() if inc.sum() > 0 else np.zeros(n_vars)

    def at(j: int, value: float, fixed: Optional[dict] = None) -> float:
        row = med.copy()
        if fixed:
            for k, val in fixed.items():
                row[k] = val
        row[j] = value
        return float(predict(row.reshape(1, -1))[0])

    med_pred = at(0, float(med[0]))
    variables: List[VariableInsight] = []
    q25, q75 = np.percentile(X_ref, 25, axis=0), np.percentile(X_ref, 75, axis=0)
    for j in range(n_vars):
        lo, hi = np.percentile(X_ref[:, j], [5, 95])
        grid = np.linspace(lo, hi, N_GRID)
        G = np.tile(med, (N_GRID, 1))
        G[:, j] = grid
        curve = np.asarray(predict(G), float)
        direction, shape = _classify(curve, scale)
        delta = at(j, q75[j]) - at(j, q25[j])
        elasticity = None
        if med[j] != 0 and abs(med_pred) > 0.05 * scale:
            elasticity = 100.0 * (at(j, 1.1 * med[j]) - med_pred) / abs(med_pred)
        turning = None
        if direction == "rises then falls":
            turning = float(grid[int(np.argmax(curve))])
        elif direction == "falls then rises":
            turning = float(grid[int(np.argmin(curve))])
        gain80 = None
        total = float(curve[-1] - curve[0])
        if direction in ("increasing", "decreasing") and abs(total) > 0:
            frac = (curve - curve[0]) / total
            gain80 = float(grid[int(np.argmax(frac >= 0.8))])
        swing = float(np.ptp(curve))
        v = VariableInsight(name=names[j], used=j in set(used_indices), importance=float(share[j]),
                            direction=direction, shape=shape, low=float(q25[j]), high=float(q75[j]),
                            delta=float(delta), grid=grid, curve=curve, elasticity=elasticity, turning_x=turning,
                            gain80_x=gain80, swing=swing, swing_share=swing / scale)
        v.text = _variable_text(v, target)
        variables.append(v)

    # strongest interaction between two meaningful variables
    interaction = None
    best_score = 0.0
    active = [j for j in range(n_vars) if share[j] >= MEANINGFUL_SHARE]
    for a in active:
        for b in active:
            if a == b:
                continue
            d_low = at(a, q75[a], {b: q25[b]}) - at(a, q25[a], {b: q25[b]})
            d_high = at(a, q75[a], {b: q75[b]}) - at(a, q25[a], {b: q75[b]})
            big = max(abs(d_low), abs(d_high))
            if big < 0.02 * scale:
                continue
            rel = abs(d_high - d_low) / big
            if rel > 0.5 and rel * big > best_score:
                best_score = rel * big
                interaction = (f"The effect of {names[a]} depends on {names[b]}: raising {names[a]} from low to high "
                               f"changes predicted {target} by {d_low:+.3g} when {names[b]} is low, "
                               f"but by {d_high:+.3g} when {names[b]} is high.")

    ranked = sorted(range(n_vars), key=lambda j: -share[j])
    drivers = [j for j in ranked if share[j] >= MEANINGFUL_SHARE]
    if not drivers:
        headline = "No variable shows a meaningful effect on this equation's predictions."
    elif len(drivers) == 1:
        headline = f"{names[drivers[0]]} is the only driver ({share[drivers[0]]:.0%} of the equation's total sensitivity)."
    else:
        parts = ", ".join(f"{names[j]} ({share[j]:.0%})" for j in drivers[:3])
        headline = f"Main driver: {names[drivers[0]]}. Influence ranking: {parts}."

    denom = float(np.mean(np.abs(y_val)))
    pct = f" (about {100 * base / denom:.1f}% of the average {target})" if denom > 0 else ""
    accuracy = f"On unseen data the typical prediction error (RMSE) is {_g(base)} in {target}{pct}."
    return ModelInsights(headline=headline, variables=variables, interaction=interaction, accuracy=accuracy,
                         rmse=float(base), mean_abs_target=denom, target_std=float(np.std(y_val)))
