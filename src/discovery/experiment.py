"""Experiment runner: one complete scientific discovery cycle.

    hidden law -> noisy observations -> train/validation split
      -> evolutionary search (engine sees ONLY training X, y)
      -> best discovery -> held-out validation -> post-experiment verification

Several independent searches (different seeds) are run. The "best" run is chosen by
its TRAINING fitness only, so the validation data is never used for model selection.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

import numpy as np

from ..data.generators import Observations, SystemSpec, generate_observations, get_system
from ..data.preprocessing import train_validation_split
from ..utils.metrics import mae, nrmse, r2, rmse
from .evolution import EvolutionConfig, run_evolution
from .expressions import Node, evaluate, size
from .latex import equation_latex, nonneg_from_data, sympy_latex
from .simplify import format_equation, simplify_tree, strip_outer_parens, sympy_form

NEAR_CEILING_TOL = 0.01   # a run "reaches the noise ceiling" if held-out R2 is within this
COMPLEX_NODE_LIMIT = 15


@dataclass
class ExperimentConfig:
    system_key: str = "projectile"
    n_samples: int = 200
    noise_level: float = 0.02
    val_fraction: float = 0.3
    seed: int = 0
    n_runs: int = 3
    n_jobs: int = 1
    evolution: EvolutionConfig = field(default_factory=lambda: EvolutionConfig(pop_size=150, generations=40))

    def validate(self) -> None:
        if not (1 <= self.n_runs <= 20):
            raise ValueError("n_runs must be between 1 and 20")
        if not (1 <= self.n_jobs <= 16):
            raise ValueError("n_jobs must be between 1 and 16")
        self.evolution.validate()


@dataclass
class RunResult:
    seed: int
    tree: Node
    expression: str            # simplified, with output name
    raw_expression: str
    size: int
    fitness: float
    train: Dict[str, float]
    validation: Dict[str, float]
    nrmse_vs_true_law: float   # error against the NOISE-FREE hidden law on validation inputs
    history: List[Dict]
    generations_run: int
    stopped_early: bool
    elapsed: float


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    spec: SystemSpec
    observations: Observations
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    y_val_clean: np.ndarray
    runs: List[RunResult]
    best_index: int
    noise_ceiling_r2: float
    run_disagreement: float
    interpretation: List[str]
    elapsed: float

    @property
    def best(self) -> RunResult:
        return self.runs[self.best_index]

    def predict_best(self, X: np.ndarray) -> np.ndarray:
        return evaluate(self.best.tree, X)

    def to_record(self, include_history: bool = True) -> Dict:
        b = self.best
        near = sum(r.validation["r2"] >= self.noise_ceiling_r2 - NEAR_CEILING_TOL for r in self.runs)
        rec = {
            "kind": "physics",
            "system": self.spec.key,
            "system_title": self.spec.title,
            "n_samples": self.config.n_samples,
            "noise_level": self.config.noise_level,
            "val_fraction": self.config.val_fraction,
            "seed": self.config.seed,
            "n_runs": self.config.n_runs,
            "pop_size": self.config.evolution.pop_size,
            "generations": self.config.evolution.generations,
            "parsimony": self.config.evolution.parsimony,
            "expression": b.expression,
            "complexity": b.size,
            "train": b.train,
            "validation": b.validation,
            "nrmse_vs_true_law": b.nrmse_vs_true_law,
            "noise_ceiling_r2": self.noise_ceiling_r2,
            "runs_near_ceiling": int(near),
            "elapsed_seconds": self.elapsed,
            "run_summaries": [
                {"seed": r.seed, "expression": r.expression, "complexity": r.size,
                 "val_r2": r.validation["r2"], "val_rmse": r.validation["rmse"]}
                for r in self.runs
            ],
        }
        if include_history:
            rec["fitness_history"] = [
                {k: h[k] for k in ("generation", "best_fitness", "mean_fitness", "best_loss", "best_size")}
                for h in b.history
            ]
        return rec


# --------------------------------------------------------------------------- internals
def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {"r2": r2(y, p), "rmse": rmse(y, p), "mae": mae(y, p), "nrmse": nrmse(y, p)}


def _search(args):
    """Top-level (picklable) worker. It receives ONLY training data and settings."""
    X_tr, y_tr, cfg, names = args
    t0 = time.time()
    res = run_evolution(X_tr, y_tr, cfg, var_names=names)
    return res, time.time() - t0


def _disagreement(preds: List[np.ndarray], scale: float) -> float:
    if len(preds) < 2 or scale <= 0:
        return 0.0
    worst = 0.0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            worst = max(worst, float(np.sqrt(np.mean((preds[i] - preds[j]) ** 2)) / scale))
    return worst


def _interpret(spec: SystemSpec, best: RunResult, ceiling: float, n_near: int, n_runs: int,
               disagreement: float) -> List[str]:
    gap = ceiling - best.validation["r2"]
    notes: List[str] = []
    if gap <= NEAR_CEILING_TOL and best.nrmse_vs_true_law <= 0.05:
        notes.append(
            "On unseen observations the discovered equation predicts about as well as the true "
            "law could given the measurement noise, and it agrees numerically with the hidden law."
        )
    elif best.validation["r2"] >= 0.9:
        notes.append(
            "The discovered equation predicts unseen observations well but falls short of what "
            "the true law achieves at this noise level; treat it as an approximation."
        )
    else:
        notes.append("The discovered equation generalises poorly to unseen observations; do not rely on it.")
    notes.append(f"{n_near} of {n_runs} independent searches reached the noise ceiling on held-out data.")
    if n_runs > 1:
        if disagreement <= 0.05:
            notes.append("All searches produced functionally equivalent predictions on held-out inputs.")
        else:
            notes.append(
                "Independent searches disagree on held-out inputs: the data may not pin down a "
                "unique equation. Compare the runs before trusting any single expression."
            )
    if best.size > COMPLEX_NODE_LIMIT:
        notes.append(
            f"The best equation is complex ({best.size} nodes). Good accuracy with a complicated form "
            "can be coincidental; prefer simpler candidates of similar accuracy."
        )
    notes.append(
        "Numerical agreement with a simulated law is not proof of a physical law. The data are "
        "simulated, not laboratory measurements."
    )
    return notes


# --------------------------------------------------------------------------- public API
def run_experiment(
    cfg: ExperimentConfig, progress_callback: Optional[Callable[[Dict], None]] = None
) -> ExperimentResult:
    cfg.validate()
    t_start = time.time()
    spec = get_system(cfg.system_key)
    obs = generate_observations(spec, cfg.n_samples, cfg.noise_level, cfg.seed)
    X_tr, y_tr, X_va, y_va, _, idx_va = train_validation_split(obs.X, obs.y_obs, cfg.val_fraction, cfg.seed)
    y_va_clean = obs.y_clean[idx_va]

    # Build one job per independent search. Jobs contain training data only.
    jobs = [
        (X_tr, y_tr, replace(cfg.evolution, seed=cfg.seed * 1000 + i), list(spec.input_names))
        for i in range(cfg.n_runs)
    ]
    outputs = []
    if cfg.n_jobs > 1 and cfg.n_runs > 1:
        from concurrent.futures import ProcessPoolExecutor
        try:
            with ProcessPoolExecutor(max_workers=min(cfg.n_jobs, cfg.n_runs)) as pool:
                for i, out in enumerate(pool.map(_search, jobs)):
                    outputs.append(out)
                    if progress_callback:
                        progress_callback({"run": i, "n_runs": cfg.n_runs, "done": True})
        except (OSError, RuntimeError):  # no multiprocessing available: fall back to sequential
            outputs = []
    if not outputs:
        for i, job in enumerate(jobs):
            if progress_callback:
                X_, y_, ecfg, names = job
                cb = lambda rec, i=i: progress_callback({"run": i, "n_runs": cfg.n_runs, **rec})
                t0 = time.time()
                outputs.append((run_evolution(X_, y_, ecfg, var_names=names, progress_callback=cb), time.time() - t0))
                progress_callback({"run": i, "n_runs": cfg.n_runs, "done": True})
            else:
                outputs.append(_search(job))

    runs: List[RunResult] = []
    for (res, elapsed), job in zip(outputs, jobs):
        tree = simplify_tree(res.best.tree)
        names = list(spec.input_names)
        p_tr, p_va = evaluate(tree, X_tr), evaluate(tree, X_va)
        runs.append(RunResult(
            seed=job[2].seed,
            tree=tree,
            expression=format_equation(tree, names, spec.output_name),
            raw_expression=f"{spec.output_name} = {strip_outer_parens(res.best.tree.to_string(names))}",
            size=size(tree),
            fitness=res.best.fitness,
            train=_metrics(y_tr, p_tr),
            validation=_metrics(y_va, p_va),
            nrmse_vs_true_law=nrmse(y_va_clean, p_va),
            history=res.history,
            generations_run=res.generations_run,
            stopped_early=res.stopped_early,
            elapsed=elapsed,
        ))

    best_index = int(np.argmin([r.fitness for r in runs]))   # selected on TRAINING fitness only
    ceiling = r2(y_va, y_va_clean)                           # what the true law scores on this noisy data
    disagreement = _disagreement([evaluate(r.tree, X_va) for r in runs], float(np.std(y_va_clean)))
    n_near = sum(r.validation["r2"] >= ceiling - NEAR_CEILING_TOL for r in runs)
    interpretation = _interpret(spec, runs[best_index], ceiling, n_near, len(runs), disagreement)

    return ExperimentResult(
        config=cfg, spec=spec, observations=obs,
        X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va, y_val_clean=y_va_clean,
        runs=runs, best_index=best_index, noise_ceiling_r2=ceiling,
        run_disagreement=disagreement, interpretation=interpretation,
        elapsed=time.time() - t_start,
    )


def algebraic_form(result: ExperimentResult) -> Optional[str]:
    """Optional SymPy-expanded form of the best equation (None if SymPy is not installed)."""
    s = sympy_form(result.best.tree, list(result.spec.input_names))
    return f"{result.spec.output_name} = {s}" if s else None


def best_latex(result: ExperimentResult) -> str:
    """The best equation as LaTeX."""
    names = list(result.spec.input_names)
    return equation_latex(result.best.tree, names, result.spec.output_name, nonneg_from_data(result.X_train))


def algebraic_latex(result: ExperimentResult) -> Optional[str]:
    """SymPy-expanded form of the best equation as LaTeX (None if SymPy is not installed)."""
    return sympy_latex(result.best.tree, list(result.spec.input_names), result.spec.output_name)
