"""Turn search results into a short list of DISTINCT, ranked candidate equations.

Rank 1 is the best TRAINING fitness; the alternatives are the other non-dominated trade-offs between
training error and complexity (validation data never selects anything).
Two candidates that make practically the same predictions are treated as duplicates, so
the top 3 are genuinely different equations - typically a best fit plus alternatives
that trade accuracy against simplicity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.metrics import mae, nrmse, r2, rmse
from .expressions import Node, evaluate, size, used_variables
from .insights import ModelInsights, explain_model
from .latex import equation_latex, nonneg_from_data, sympy_latex
from .scaling import Scaler, unscale_tree
from .simplify import format_equation, simplify_tree, sympy_form

MIN_DIFF = 0.05      # candidates closer than 5% of std(y) (RMS difference) count as duplicates
MAX_LOSS = 0.9       # candidates explaining almost nothing (train NRMSE >= 0.9) are not worth showing


@dataclass
class Candidate:
    rank: int
    tree: Node
    expression: str                # "target = ..." in ORIGINAL units, simplified
    expanded: Optional[str]        # SymPy-expanded form if SymPy is installed
    size: int
    fitness: float                 # training fitness (error + complexity penalty)
    train: Dict[str, float]
    validation: Dict[str, float]
    pred_val: np.ndarray = field(repr=False)
    used_variables: List[str] = field(default_factory=list)
    insights: Optional[ModelInsights] = None
    note: str = ""
    run_index: int = 0
    latex: str = ""                       # the expression as LaTeX (`target = ...`)
    expanded_latex: Optional[str] = None   # SymPy-expanded form as LaTeX

    def predict(self, X: np.ndarray) -> np.ndarray:
        return evaluate(self.tree, X)


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {"r2": r2(y, p), "rmse": rmse(y, p), "mae": mae(y, p), "nrmse": nrmse(y, p)}


def _role_note(c: Candidate, best: Candidate) -> str:
    if c.rank == 1:
        return "Best training fit after the complexity penalty."
    d_size = c.size - best.size
    d_r2 = c.validation["r2"] - best.validation["r2"]
    shape = "simpler" if d_size < 0 else ("more complex" if d_size > 0 else "same size")
    acc = "less accurate" if d_r2 < -0.002 else ("more accurate" if d_r2 > 0.002 else "similar accuracy")
    return f"Alternative: {shape} ({c.size} vs {best.size} nodes), {acc} on held-out data (R² {c.validation['r2']:.4f} vs {best.validation['r2']:.4f})."


def build_candidates(
    hall: Sequence[Tuple[Node, float, int]], scaler: Scaler,
    X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
    names: Sequence[str], target: str, k: int = 3, seed: int = 0,
) -> List[Candidate]:
    """hall: (tree in SCALED space, training fitness, run index) triples from the searches."""
    sd_y = float(np.std(y_tr)) or 1.0
    ok: List[tuple] = []          # (fitness, tree, train_pred, run, train_loss, size)
    rejected: List[tuple] = []
    seen = set()
    for tree, fit, run in sorted(hall, key=lambda t: t[1]):
        t = simplify_tree(unscale_tree(tree, scaler))
        key = t.to_string(names)
        if key in seen:
            continue
        seen.add(key)
        p = evaluate(t, X_tr)
        if not np.all(np.isfinite(p)):
            continue
        loss = nrmse(y_tr, p)
        row = (fit, t, p, run, loss, size(t))
        (ok if (loss < MAX_LOSS and used_variables(t)) else rejected).append(row)

    chosen: List[tuple] = []
    if ok:
        chosen.append(ok[0])          # rank 1: best training fitness
        # Pareto front on TRAINING data: keep a candidate only if no simpler-or-equal one fits at least as well.
        front, best_loss = [], np.inf
        for row in sorted(ok, key=lambda r: (r[5], r[4])):
            if row[4] < best_loss * 0.99:
                front.append(row)
                best_loss = row[4]
        for row in sorted(front, key=lambda r: r[0]):
            if len(chosen) == k:
                break
            if all(np.sqrt(np.mean((row[2] - c[2]) ** 2)) / sd_y >= MIN_DIFF for c in chosen):
                chosen.append(row)
        for row in ok:                # still short of k: fill with the next-best distinct equations
            if len(chosen) == k:
                break
            if all(row is not c for c in chosen) and \
                    all(np.sqrt(np.mean((row[2] - c[2]) ** 2)) / sd_y >= MIN_DIFF for c in chosen):
                chosen.append(row)
    elif rejected:
        chosen = [rejected[0]]      # nothing fits well: still show the least bad one (its metrics will say so)

    cands: List[Candidate] = []
    nonneg = nonneg_from_data(X_tr)
    for rank, (fit, t, p_tr, run, _loss, _size) in enumerate(chosen, 1):
        p_va = evaluate(t, X_va)
        used_idx = sorted(used_variables(t))
        alg = sympy_form(t, list(names))
        c = Candidate(
            rank=rank, tree=t, expression=format_equation(t, list(names), target),
            expanded=f"{target} = {alg}" if alg else None, size=size(t), fitness=fit,
            train=_metrics(y_tr, p_tr), validation=_metrics(y_va, p_va), pred_val=p_va,
            used_variables=[names[i] for i in used_idx], run_index=run,
            latex=equation_latex(t, list(names), target, nonneg),
            expanded_latex=sympy_latex(t, list(names), target),
        )
        c.insights = explain_model(c.predict, X_tr, X_va, y_va, list(names), target, used_idx, seed=seed)
        cands.append(c)
    for c in cands:
        c.note = _role_note(c, cands[0])
    return cands
