"""Discovery on a USER-SUPPLIED table (for example an uploaded CSV).

The user chooses the variable of interest (the dependent variable / target); the other
numeric columns are the candidate independent variables. Variable names in the
discovered equations are the column headers (made safe for display, e.g. 'Length (m)'
-> 'Length_m').

Honest by construction:
* the engine only ever sees the TRAINING rows; held-out rows are used for reporting only;
* candidates are ranked on training fitness, never on validation scores;
* there is no hidden ground truth here, so agreement between independent searches and
  held-out accuracy are the evidence offered - and the report says what they cannot show.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.preprocessing import train_validation_split
from ..utils.metrics import mae, r2, rmse
from .candidates import Candidate, build_candidates
from .evolution import EvolutionConfig, run_evolution
from .expressions import evaluate
from .scaling import Scaler, unscale_tree
from .simplify import simplify_tree

MAX_FEATURES = 6
COMPLEX_NODE_LIMIT = 15
AGREE_TOL = 0.05     # two searches "agree" if their held-out predictions differ by < 5% of std(y)


@dataclass
class DatasetConfig:
    target: str
    features: List[str]
    val_fraction: float = 0.3
    seed: int = 0
    n_runs: int = 3
    top_k: int = 3
    evolution: EvolutionConfig = field(default_factory=lambda: EvolutionConfig(pop_size=150, generations=40))

    def validate(self, df: pd.DataFrame) -> None:
        if self.target not in df.columns:
            raise ValueError(f"Variable of interest {self.target!r} is not a column of the data.")
        if not self.features:
            raise ValueError("Select at least one independent variable.")
        if len(set(self.features)) != len(self.features):
            raise ValueError("Independent variables must be unique.")
        if self.target in self.features:
            raise ValueError("The variable of interest cannot also be an independent variable.")
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Independent variables not found in the data: {missing}")
        if len(self.features) > MAX_FEATURES:
            raise ValueError(f"At most {MAX_FEATURES} independent variables are supported; deselect some.")
        if not (1 <= self.n_runs <= 10):
            raise ValueError("n_runs must be between 1 and 10")
        if not (1 <= self.top_k <= 5):
            raise ValueError("top_k must be between 1 and 5")
        self.evolution.validate()


@dataclass
class DatasetResult:
    config: DatasetConfig
    target: str
    features: List[str]
    n_rows: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    candidates: List[Candidate]
    history: List[Dict]
    baselines: pd.DataFrame
    runs_agreeing: int
    n_runs: int
    notices: List[str]
    prep_report: Dict
    elapsed: float

    @property
    def best(self) -> Candidate:
        return self.candidates[0]

    def to_record(self) -> Dict:
        b = self.best
        return {
            "kind": "dataset", "target": self.target, "features": self.features, "n_rows": self.n_rows,
            "seed": self.config.seed, "n_runs": self.n_runs, "val_fraction": self.config.val_fraction,
            "pop_size": self.config.evolution.pop_size, "generations": self.config.evolution.generations,
            "expression": b.expression, "complexity": b.size, "train": b.train, "validation": b.validation,
            "runs_agreeing": self.runs_agreeing, "elapsed_seconds": self.elapsed,
            "candidates": [{"rank": c.rank, "expression": c.expression, "complexity": c.size,
                            "val_r2": c.validation["r2"], "val_rmse": c.validation["rmse"]} for c in self.candidates],
            "fitness_history": [{k: h[k] for k in ("generation", "best_fitness", "mean_fitness", "best_loss", "best_size")}
                                for h in self.history],
        }


def _linear_baseline(X_tr, y_tr, X_va, y_va) -> pd.DataFrame:
    A = np.column_stack([X_tr, np.ones(len(X_tr))])
    coef, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    p_lin = X_va @ coef[:-1] + coef[-1]
    p_mean = np.full(len(y_va), float(np.mean(y_tr)))
    return pd.DataFrame([
        {"model": "Mean baseline", "val_r2": r2(y_va, p_mean), "val_rmse": rmse(y_va, p_mean)},
        {"model": "Linear regression", "val_r2": r2(y_va, p_lin), "val_rmse": rmse(y_va, p_lin)},
    ])


def run_dataset_experiment(
    df: pd.DataFrame, cfg: DatasetConfig,
    progress_callback: Optional[Callable[[Dict], None]] = None, prep_report: Optional[Dict] = None,
) -> DatasetResult:
    t0 = time.time()
    cfg.validate(df)
    data = df[cfg.features + [cfg.target]].astype(float)
    if not np.all(np.isfinite(data.to_numpy())):
        raise ValueError("Data contains missing or infinite values; clean it first.")
    X, y = data[cfg.features].to_numpy(), data[cfg.target].to_numpy()
    X_tr, y_tr, X_va, y_va, _, _ = train_validation_split(X, y, cfg.val_fraction, cfg.seed)

    scaler = Scaler.fit(X_tr, y_tr, center=False)       # training statistics only
    Z_tr, zy_tr = scaler.transform_X(X_tr), scaler.transform_y(y_tr)

    outs = []
    for i in range(cfg.n_runs):
        ecfg = replace(cfg.evolution, seed=cfg.seed * 1000 + i)
        cb = (lambda rec, i=i: progress_callback({"run": i, "n_runs": cfg.n_runs, **rec})) if progress_callback else None
        outs.append(run_evolution(Z_tr, zy_tr, ecfg, var_names=cfg.features, progress_callback=cb))

    hall = [(ind.tree, ind.fitness, i) for i, o in enumerate(outs) for ind in o.hall_of_fame]
    cands = build_candidates(hall, scaler, X_tr, y_tr, X_va, y_va, cfg.features, cfg.target, k=cfg.top_k, seed=cfg.seed)
    if not cands:
        raise ValueError("The search found no usable equation. Try more generations or a larger population.")
    best_run = int(np.argmin([o.best.fitness for o in outs]))

    # reproducibility: how many independent searches reached a functionally equivalent best equation?
    top_pred = cands[0].pred_val
    sd = float(np.std(y_va)) or 1.0
    agree = 0
    for o in outs:
        p = evaluate(simplify_tree(unscale_tree(o.best.tree, scaler)), X_va)
        agree += int(np.sqrt(np.mean((p - top_pred) ** 2)) / sd < AGREE_TOL)

    baselines = _linear_baseline(X_tr, y_tr, X_va, y_va)
    b = cands[0]
    notices = [
        "There is no hidden ground truth for uploaded data. Held-out accuracy and agreement between independent "
        "searches are the evidence here, and neither proves the equation is a true law.",
        "Held-out rows come from the same range as the training rows: the equation has not been tested outside "
        "the range of your data.",
    ]
    if b.train["r2"] - b.validation["r2"] > 0.1:
        notices.append("The equation fits training data much better than held-out data: it may be over-fitted.")
    if b.validation["r2"] < 0.5:
        notices.append("Held-out R² is low: the selected variables may not explain the variable of interest, "
                       "or the relationship is too complex or too noisy for this operator set.")
    if b.size > COMPLEX_NODE_LIMIT:
        notices.append(f"The best equation is complex ({b.size} nodes); prefer a simpler alternative of similar accuracy.")
    if len(data) < 60:
        notices.append("Small sample: treat all results as tentative.")
    if cfg.n_runs > 1:
        notices.append(f"{agree} of {cfg.n_runs} independent searches reached an equivalent best equation."
                       + ("" if agree > 1 else " The result is not reproducible across searches; treat it with caution."))
    corr = np.corrcoef(X_tr, rowvar=False) if X_tr.shape[1] > 1 else None
    if corr is not None:
        for a in range(corr.shape[0]):
            for c in range(a + 1, corr.shape[0]):
                if abs(corr[a, c]) > 0.95:
                    notices.append(f"{cfg.features[a]} and {cfg.features[c]} are almost perfectly correlated "
                                   f"(r={corr[a, c]:.2f}); the equation cannot tell which of them matters.")

    return DatasetResult(
        config=cfg, target=cfg.target, features=list(cfg.features), n_rows=len(data), X_train=X_tr, y_train=y_tr,
        X_val=X_va, y_val=y_va, candidates=cands, history=outs[best_run].history, baselines=baselines,
        runs_agreeing=agree, n_runs=cfg.n_runs, notices=notices, prep_report=prep_report or {},
        elapsed=time.time() - t0,
    )
