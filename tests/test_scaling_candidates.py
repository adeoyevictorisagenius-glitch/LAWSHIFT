import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import clean_column_names, prepare_numeric_dataframe
from src.discovery.candidates import build_candidates
from src.discovery.dataset_experiment import DatasetConfig, run_dataset_experiment
from src.discovery.evolution import EvolutionConfig, run_evolution
from src.discovery.expressions import Binary, Const, Unary, Var, evaluate, random_tree
from src.discovery.insights import explain_model
from src.discovery.scaling import Scaler, unscale_tree
from src.discovery.simplify import simplify_tree

FAST = EvolutionConfig(pop_size=50, generations=10)


# ------------------------------------------------------------------ scaling
@pytest.mark.parametrize("center", [False, True])
def test_unscale_tree_is_exact(center):
    rng = np.random.default_rng(0)
    X = rng.uniform(100, 5000, (80, 3)) * np.array([1, 1e-3, 10])
    y = rng.normal(50, 20, 80)
    sc = Scaler.fit(X, y, center=center)
    Z = sc.transform_X(X)
    checked = 0
    for _ in range(150):
        t = random_tree(rng, 3, max_depth=4)
        inner = evaluate(t, Z)
        ok = np.abs(inner) < 1e6          # values near the 1e12 safety cap saturate at different points; skip those
        expected = inner * sc.sy + sc.my
        got = evaluate(unscale_tree(t, sc), X)
        np.testing.assert_allclose(got[ok], expected[ok], rtol=1e-8, atol=1e-8)
        checked += int(ok.sum())
    assert checked > 3000


def test_unscale_then_simplify_preserves_function():
    rng = np.random.default_rng(1)
    X = rng.uniform(1, 100, (60, 2))
    sc = Scaler.fit(X, rng.normal(size=60), center=True)
    for _ in range(100):
        t = unscale_tree(random_tree(rng, 2, max_depth=4), sc)
        a, b = evaluate(simplify_tree(t), X), evaluate(t, X)
        ok = np.abs(b) < 1e6
        np.testing.assert_allclose(a[ok], b[ok], rtol=1e-7, atol=1e-7)


def test_scaler_rejects_bad_data():
    X = np.column_stack([np.arange(20.0), np.ones(20)])
    with pytest.raises(ValueError):
        Scaler.fit(X, np.arange(20.0))                       # constant predictor
    with pytest.raises(ValueError):
        Scaler.fit(X[:, :1], np.ones(20))                    # constant target
    with pytest.raises(ValueError):
        Scaler.fit(X[:, :1] * 1e12, np.arange(20.0))         # too large


def test_ratio_stays_tidy_after_unscaling():
    sc = Scaler(sx=np.array([2.0, 4.0]), sy=3.0)
    t = unscale_tree(Binary("div", Var(0), Var(1)), sc)
    s = simplify_tree(t)
    assert s.to_string(["F", "m"]) == "(6 * (F / m))"        # 3 * (F/2) / (m/4) = 6 F/m


def test_new_simplifier_rules():
    assert simplify_tree(Unary("sq", Binary("mul", Const(3.0), Var(0)))).to_string() == "(9 * (x0)^2)"
    assert simplify_tree(Binary("mul", Binary("mul", Const(2.0), Var(0)), Binary("mul", Const(5.0), Var(1)))).to_string() == "(10 * (x0 * x1))"


# ------------------------------------------------------------------ header names / csv cleaning
def test_clean_column_names():
    assert clean_column_names(["Length (m)", "Mass (kg)", "1st", "", "Length (m)"]) == \
        ["Length_m", "Mass_kg", "v_1st", "col4", "Length_m_2"]


def test_prepare_reports_renamed_headers():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Force (N)": rng.uniform(1, 9, 50), "Mass (kg)": rng.uniform(1, 9, 50), "note": ["x"] * 50})
    clean, rep = prepare_numeric_dataframe(df)
    assert list(clean.columns) == ["Force_N", "Mass_kg"]
    assert rep["renamed_columns"] == {"Force (N)": "Force_N", "Mass (kg)": "Mass_kg"} and rep["ignored_columns"] == ["note"]


# ------------------------------------------------------------------ hall of fame + insights
def test_hall_of_fame_is_sorted_and_distinct_enough():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 5, (80, 2))
    y = X[:, 0] * X[:, 1]
    r = run_evolution(X, y, EvolutionConfig(pop_size=50, generations=8, seed=1))
    f = [i.fitness for i in r.hall_of_fame]
    assert r.hall_of_fame and f == sorted(f)
    assert r.hall_of_fame[0].fitness <= r.best.fitness + 1e-9 or abs(r.hall_of_fame[0].fitness - r.best.fitness) < 1e-6


def test_insights_importance_direction_and_shape():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 10, (300, 3))
    y_fn = lambda M: 5 * np.sqrt(M[:, 0]) - 2 * M[:, 1]          # third variable unused
    ins = explain_model(y_fn, X, X, y_fn(X), ["adv", "cost", "noise"], "profit", [0, 1], seed=0)
    v = {x.name: x for x in ins.variables}
    assert v["noise"].importance == pytest.approx(0.0, abs=1e-9) and not v["noise"].used
    assert v["adv"].direction == "increasing" and v["adv"].shape == "flattens out"
    assert v["cost"].direction == "decreasing" and v["cost"].delta < 0 and v["cost"].shape == "roughly linear"
    assert v["adv"].importance + v["cost"].importance == pytest.approx(1.0)
    assert "adv" in ins.headline
    assert any("diminishing returns" in b for b in ins.bullets())
    assert any("not used" in b for b in ins.bullets())


def test_insights_detects_peak_and_interaction():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 10, (400, 2))
    peak = lambda M: -(M[:, 0] - 5) ** 2 + 0 * M[:, 1]
    v = explain_model(peak, X, X, peak(X), ["a", "b"], "y", [0], seed=0).variables[0]
    assert v.direction == "rises then falls" and "peak" in v.text
    inter = lambda M: M[:, 0] * M[:, 1]
    ins = explain_model(inter, X, X, inter(X), ["a", "b"], "y", [0, 1], seed=0)
    assert ins.interaction and "depends on" in ins.interaction


def test_insights_flat_variable_and_caveat():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, (100, 1))
    ins = explain_model(lambda M: np.full(len(M), 3.0) + 0 * M[:, 0], X, X, rng.normal(3, 1, 100), ["a"], "y", [0])
    assert ins.variables[0].direction == "flat"
    assert "not proven cause and effect" in ins.caveat


# ------------------------------------------------------------------ dataset experiment
def _pendulum(n=150, seed=3):
    rng = np.random.default_rng(seed)
    L = rng.uniform(0.2, 2.0, n)
    return pd.DataFrame({"Length (m)": L, "Mass (kg)": rng.uniform(0.1, 5, n),
                         "Period (s)": 2 * np.pi * np.sqrt(L / 9.81) * (1 + rng.normal(0, 0.01, n))})


def test_dataset_experiment_uses_headers_and_finds_law():
    clean, rep = prepare_numeric_dataframe(_pendulum())
    res = run_dataset_experiment(
        clean, DatasetConfig(target="Period_s", features=["Length_m", "Mass_kg"], n_runs=2,
                             evolution=EvolutionConfig(pop_size=80, generations=25)), prep_report=rep)
    best = res.best
    assert best.expression.startswith("Period_s = ") and "Length_m" in best.expression
    assert "x0" not in best.expression and "Mass_kg" not in best.used_variables
    assert best.validation["r2"] > 0.99
    assert 1 <= len(res.candidates) <= 3 and [c.rank for c in res.candidates] == list(range(1, len(res.candidates) + 1))
    # predictions of the displayed equation on RAW data reproduce the reported validation metrics
    p = evaluate(best.tree, res.X_val)
    assert 1 - np.sum((res.y_val - p) ** 2) / np.sum((res.y_val - res.y_val.mean()) ** 2) == pytest.approx(best.validation["r2"])
    assert any("no hidden ground truth" in n for n in res.notices)
    assert res.to_record()["kind"] == "dataset" and res.to_record()["candidates"][0]["rank"] == 1


def test_dataset_experiment_handles_large_scale_data():
    rng = np.random.default_rng(4)
    n = 150
    df = pd.DataFrame({"x": rng.uniform(1e4, 9e4, n), "z": rng.uniform(0, 1e-3, n)})
    df["y"] = 0.002 * df["x"] + rng.normal(0, 2, n)
    res = run_dataset_experiment(df, DatasetConfig(target="y", features=["x", "z"], n_runs=1,
                                                   evolution=EvolutionConfig(pop_size=60, generations=15)))
    assert res.best.validation["r2"] > 0.95


def test_dataset_experiment_reproducible():
    clean, _ = prepare_numeric_dataframe(_pendulum())
    cfg = lambda: DatasetConfig(target="Period_s", features=["Length_m"], n_runs=1, seed=5, evolution=FAST)
    a, b = run_dataset_experiment(clean, cfg()), run_dataset_experiment(clean, cfg())
    assert [c.expression for c in a.candidates] == [c.expression for c in b.candidates]


@pytest.mark.parametrize("kw", [dict(target="nope", features=["Length_m"]), dict(target="Period_s", features=[]),
                                dict(target="Period_s", features=["Period_s"]),
                                dict(target="Period_s", features=["Length_m", "Length_m"]),
                                dict(target="Period_s", features=["ghost"]), dict(target="Period_s", features=["Length_m"], n_runs=0)])
def test_dataset_experiment_rejects_bad_selection(kw):
    clean, _ = prepare_numeric_dataframe(_pendulum())
    with pytest.raises(ValueError):
        run_dataset_experiment(clean, DatasetConfig(evolution=FAST, **kw))


def test_business_analysis_returns_distinct_candidates_with_insights():
    from src.business.analysis import analyze, demo_dataset
    res = analyze(demo_dataset(), "profit", ["advertising", "units_sold", "operating_expenses"], n_runs=2,
                  evolution=EvolutionConfig(pop_size=80, generations=20))
    assert 1 <= len(res.candidates) <= 3
    assert res.equation == res.candidates[0].expression and "z(" not in res.equation
    assert all(c.insights and c.insights.bullets() for c in res.candidates)
