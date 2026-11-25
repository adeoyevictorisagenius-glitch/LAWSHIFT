"""Business Pattern Lab: correlations, candidate equations, regression, anomalies.

IMPORTANT: everything here describes ASSOCIATION in the supplied data. None of it
establishes that one business activity caused a financial outcome.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..data.preprocessing import prepare_numeric_dataframe, train_validation_split
from ..discovery.evolution import EvolutionConfig, run_evolution
from ..discovery.expressions import evaluate, size
from ..discovery.latex import linear_latex
from ..discovery.candidates import Candidate, build_candidates
from ..discovery.scaling import Scaler
from ..utils.metrics import mae, r2, rmse

MIN_ROWS = 30
MAX_ROWS = 50_000
MAX_FEATURES = 6
ANOMALY_Z = 3.5
COMPLEX_NODE_LIMIT = 15

CAUSATION_NOTICE = (
    "Correlation is not causation. The relationships below describe how variables moved "
    "together in this dataset. They do not prove that advertising, spending or any other "
    "activity caused a change in revenue or profit: hidden factors, timing, seasonality and "
    "reverse causation can produce the same patterns."
)

DEMO_ANOMALY_ROWS = (17, 63, 111)
CENTER_BUSINESS_DATA = True    # measured: centring gave better held-out accuracy and clearer simple alternatives on the demo data


# --------------------------------------------------------------------------- data
def demo_dataset(n: int = 150, seed: int = 7) -> pd.DataFrame:
    """SYNTHETIC demonstration data (clearly not real company data).

    Units sold rise with advertising (with diminishing returns); revenue = price * units;
    operating expenses have a fixed part plus a part that scales with units; profit =
    revenue - operating expenses - advertising. A few rows get unusual profit shocks so
    the anomaly detector has something to find.
    """
    if n < MIN_ROWS:
        raise ValueError(f"n must be at least {MIN_ROWS}")
    rng = np.random.default_rng(seed)
    advertising = rng.uniform(500, 5000, n)
    units = 40 + 14 * np.sqrt(advertising) + rng.normal(0, 25, n)
    units = np.maximum(units, 1)
    revenue = 25.0 * units + rng.normal(0, 400, n)
    opex = 6000 + 4.0 * units + rng.normal(0, 300, n)
    profit = revenue - opex - advertising
    sd = float(np.std(profit))
    for k, row in enumerate(DEMO_ANOMALY_ROWS):
        if row < n:
            profit[row] += (6 if k % 2 == 0 else -6) * sd
    return pd.DataFrame({
        "advertising": advertising.round(2), "units_sold": units.round(1),
        "operating_expenses": opex.round(2), "revenue": revenue.round(2), "profit": profit.round(2),
    })


def prepare_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Keep usable numeric columns, drop invalid rows, and report what was done."""
    return prepare_numeric_dataframe(df, MIN_ROWS, MAX_ROWS)


# --------------------------------------------------------------------------- statistics
def target_correlations(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        if c == target:
            continue
        r, p = stats.pearsonr(df[c], df[target])
        rows.append({"variable": c, "pearson_r": float(r), "p_value": float(p), "n": len(df)})
    out = pd.DataFrame(rows)
    return out.reindex(out["pearson_r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def fit_linear(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def predict_linear(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ coef[:-1] + coef[-1]


def linear_equation(coef: np.ndarray, features: Sequence[str], target: str) -> str:
    terms = " ".join(f"{'+' if c >= 0 else '-'} {abs(c):.4g}*{f}" for c, f in zip(coef[:-1], features))
    b = coef[-1]
    return f"{target} = {terms} {'+' if b >= 0 else '-'} {abs(b):.4g}"


def find_anomalies(y: np.ndarray, pred: np.ndarray, threshold: float = ANOMALY_Z) -> Tuple[np.ndarray, np.ndarray]:
    """Robust z-score (median / MAD) of residuals. Returns (z_scores, flagged_mask)."""
    res = np.asarray(y, float) - np.asarray(pred, float)
    med = np.median(res)
    mad = np.median(np.abs(res - med))
    if mad == 0:
        mad = np.mean(np.abs(res - med)) or 1.0
    z = 0.6745 * (res - med) / mad
    return z, np.abs(z) > threshold


# --------------------------------------------------------------------------- analysis
@dataclass
class BusinessResult:
    target: str
    features: List[str]
    prep_report: Dict
    n_rows: int
    n_train: int
    n_val: int
    seed: int
    corr_matrix: pd.DataFrame
    target_corr: pd.DataFrame
    models: pd.DataFrame
    linear_equation: str
    linear_latex: str
    equation: str
    equation_size: int
    candidates: List[Candidate]
    val_table: pd.DataFrame
    anomalies: pd.DataFrame
    y_all: np.ndarray
    X_all: np.ndarray
    pred_linear_all: np.ndarray
    anomaly_mask: np.ndarray
    y_val: np.ndarray
    pred_val_equation: np.ndarray
    pred_val_linear: np.ndarray
    history: List[Dict]
    notices: List[str]
    elapsed: float

    def to_record(self) -> Dict:
        m = self.models.set_index("model")
        return {
            "kind": "business",
            "target": self.target,
            "features": self.features,
            "n_rows": self.n_rows,
            "seed": self.seed,
            "equation": self.equation,
            "linear_equation": self.linear_equation,
            "complexity": self.equation_size,
            "candidates": [{"rank": c.rank, "expression": c.expression, "complexity": c.size,
                            "val_r2": c.validation["r2"], "val_rmse": c.validation["rmse"]} for c in self.candidates],
            "val_r2": {k: float(v) for k, v in m["val_r2"].items()},
            "val_rmse": {k: float(v) for k, v in m["val_rmse"].items()},
            "n_unusual": int(self.anomaly_mask.sum()),
            "elapsed_seconds": self.elapsed,
        }


def _metrics(y, p) -> Dict[str, float]:
    return {"r2": r2(y, p), "rmse": rmse(y, p), "mae": mae(y, p)}


def analyze(
    df: pd.DataFrame, target: str, features: Sequence[str], seed: int = 0,
    val_fraction: float = 0.3, n_runs: int = 2, evolution: Optional[EvolutionConfig] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
    prep_report: Optional[Dict] = None, top_k: int = 3,
) -> BusinessResult:
    t0 = time.time()
    features = list(features)
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found.")
    if not features:
        raise ValueError("Select at least one predictor variable.")
    if len(set(features)) != len(features):
        raise ValueError("Predictor variables must be unique.")
    if target in features:
        raise ValueError("The target variable cannot also be a predictor.")
    if len(features) > MAX_FEATURES:
        raise ValueError(f"At most {MAX_FEATURES} predictors are supported.")
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Predictor columns not found: {missing}")
    if not (1 <= n_runs <= 10):
        raise ValueError("n_runs must be between 1 and 10")
    cfg = evolution or EvolutionConfig(pop_size=100, generations=30)
    cfg.validate()

    data = df[features + [target]].astype(float)
    if not np.all(np.isfinite(data.to_numpy())):
        raise ValueError("Data contains missing or infinite values; run prepare_dataframe first.")
    X, y = data[features].to_numpy(), data[target].to_numpy()
    X_tr, y_tr, X_va, y_va, i_tr, i_va = train_validation_split(X, y, val_fraction, seed)
    if np.std(y_tr) == 0:
        raise ValueError(f"The target {target!r} is constant in the training data.")

    # Search on rescaled columns (training statistics only), then restore the winners to original units exactly.
    scaler = Scaler.fit(X_tr, y_tr, center=CENTER_BUSINESS_DATA)
    Z_tr, zy_tr = scaler.transform_X(X_tr), scaler.transform_y(y_tr)
    ymu = float(y_tr.mean())

    outs = []
    for i in range(n_runs):
        ecfg = replace(cfg, seed=seed * 1000 + i)
        cb = (lambda rec, i=i: progress_callback({"run": i, "n_runs": n_runs, **rec})) if progress_callback else None
        outs.append(run_evolution(Z_tr, zy_tr, ecfg, var_names=features, progress_callback=cb))
    best = outs[int(np.argmin([o.best.fitness for o in outs]))]      # chosen on training fitness only
    hall = [(ind.tree, ind.fitness, i) for i, o in enumerate(outs) for ind in o.hall_of_fame]
    cands = build_candidates(hall, scaler, X_tr, y_tr, X_va, y_va, features, target, k=top_k, seed=seed)
    if not cands:
        raise ValueError("The search found no usable equation. Try more generations or a larger population.")
    tree, eq_str = cands[0].tree, cands[0].expression

    coef = fit_linear(X_tr, y_tr)
    models = {
        "Mean baseline": (lambda Xr: np.full(len(Xr), ymu), 1),
        "Linear regression": (lambda Xr: predict_linear(coef, Xr), len(coef)),
        "Evolved equation": (lambda Xr: evaluate(tree, Xr), size(tree)),
    }
    rows, preds_va = [], {}
    for name, (fn, cplx) in models.items():
        p_tr, p_va = fn(X_tr), fn(X_va)
        preds_va[name] = p_va
        mt, mv = _metrics(y_tr, p_tr), _metrics(y_va, p_va)
        rows.append({"model": name, "complexity": cplx,
                     "train_r2": mt["r2"], "val_r2": mv["r2"], "val_rmse": mv["rmse"], "val_mae": mv["mae"]})
    models_df = pd.DataFrame(rows)

    val_table = pd.DataFrame({
        "row": i_va, f"actual_{target}": y_va,
        "predicted_linear": preds_va["Linear regression"],
        "predicted_equation": preds_va["Evolved equation"],
    })
    val_table["abs_error_linear"] = (val_table[f"actual_{target}"] - val_table["predicted_linear"]).abs()
    val_table["abs_error_equation"] = (val_table[f"actual_{target}"] - val_table["predicted_equation"]).abs()

    pred_lin_all = predict_linear(coef, X)
    z_scores, mask = find_anomalies(y, pred_lin_all)
    an = data.copy()
    an["linear_prediction"] = pred_lin_all
    an["residual"] = y - pred_lin_all
    an["robust_z"] = z_scores
    anomalies = an[mask].sort_values("robust_z", key=np.abs, ascending=False)
    anomalies.insert(0, "row", anomalies.index)

    corr = data.corr()
    notices = [CAUSATION_NOTICE]
    mv = models_df.set_index("model")
    if mv.loc["Evolved equation", "val_rmse"] > mv.loc["Linear regression", "val_rmse"] * 1.05:
        notices.append("The evolved equation did not beat plain linear regression on held-out data here; "
                       "prefer the simpler linear model.")
    if size(tree) > COMPLEX_NODE_LIMIT:
        notices.append(f"The evolved equation is complex ({size(tree)} nodes) and may be fitting quirks of this sample.")
    fc = corr.loc[features, features].to_numpy() if len(features) > 1 else None
    if fc is not None:
        for a in range(len(features)):
            for b in range(a + 1, len(features)):
                if abs(fc[a, b]) > 0.9:
                    notices.append(f"{features[a]} and {features[b]} are highly correlated "
                                   f"(r={fc[a, b]:.2f}); their separate effects cannot be told apart.")
    if len(data) < 60:
        notices.append("Small sample: treat all results as tentative.")
    if mask.any():
        notices.append(f"{int(mask.sum())} unusual observation(s) flagged by a robust residual test. "
                       "These are statistical flags, not proof of error or fraud.")

    return BusinessResult(
        target=target, features=features, prep_report=prep_report or {}, n_rows=len(data),
        n_train=len(i_tr), n_val=len(i_va), seed=seed, corr_matrix=corr,
        target_corr=target_correlations(data, target), models=models_df,
        linear_equation=linear_equation(coef, features, target), linear_latex=linear_latex(coef, features, target),
        equation=eq_str,
        equation_size=size(tree), candidates=cands, val_table=val_table, anomalies=anomalies.reset_index(drop=True),
        y_all=y, X_all=X, pred_linear_all=pred_lin_all, anomaly_mask=mask, y_val=y_va,
        pred_val_equation=preds_va["Evolved equation"], pred_val_linear=preds_va["Linear regression"],
        history=best.history, notices=notices, elapsed=time.time() - t0,
    )
