import numpy as np
import pandas as pd
import pytest

from src.business.analysis import (
    CAUSATION_NOTICE, DEMO_ANOMALY_ROWS, analyze, demo_dataset, find_anomalies, fit_linear,
    predict_linear, prepare_dataframe, target_correlations,
)
from src.discovery.evolution import EvolutionConfig

FAST = EvolutionConfig(pop_size=50, generations=10)


def test_demo_dataset_reproducible_and_labelled_columns():
    a, b = demo_dataset(seed=7), demo_dataset(seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["advertising", "units_sold", "operating_expenses", "revenue", "profit"]
    assert a.notna().all().all() and len(a) == 150
    with pytest.raises(ValueError):
        demo_dataset(n=5)


def test_prepare_drops_bad_rows_and_ignores_text():
    df = demo_dataset(60)
    df.loc[3, "revenue"] = np.nan
    df.loc[4, "profit"] = np.inf
    df["region"] = ["north"] * 60
    clean, rep = prepare_dataframe(df)
    assert len(clean) == 58 and rep["dropped_rows"] == 2
    assert "region" in rep["ignored_columns"] and "region" not in clean.columns


def test_prepare_converts_numeric_strings_and_drops_constants():
    df = demo_dataset(60)
    df["units_sold"] = df["units_sold"].astype(str)
    df["flat"] = 5.0
    clean, rep = prepare_dataframe(df)
    assert "units_sold" in clean.columns and rep["constant_columns"] == ["flat"]


@pytest.mark.parametrize("make", [
    lambda: pd.DataFrame(),
    lambda: demo_dataset(60).iloc[:10],
    lambda: pd.DataFrame({"a": ["x"] * 50, "b": ["y"] * 50}),
    lambda: pd.DataFrame({"a": np.arange(50.0)}),
])
def test_prepare_rejects_unusable_data(make):
    with pytest.raises(ValueError):
        prepare_dataframe(make())


def test_linear_fit_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 5
    coef = fit_linear(X, y)
    np.testing.assert_allclose(coef, [3, -2, 5], atol=1e-9)
    np.testing.assert_allclose(predict_linear(coef, X), y, atol=1e-9)


def test_anomaly_detector_flags_outliers_only():
    rng = np.random.default_rng(1)
    pred = rng.normal(size=200)
    y = pred + rng.normal(0, 0.1, 200)
    y[[5, 50]] += 5
    _, mask = find_anomalies(y, pred)
    assert mask[5] and mask[50] and mask.sum() <= 4


def test_target_correlations_sorted_by_strength():
    rng = np.random.default_rng(2)
    a = rng.normal(size=100)
    df = pd.DataFrame({"a": a, "b": rng.normal(size=100), "y": 2 * a + rng.normal(0, 0.1, 100)})
    tc = target_correlations(df, "y")
    assert tc.iloc[0]["variable"] == "a" and tc.iloc[0]["pearson_r"] > 0.95


def test_full_analysis_on_demo_data():
    df, rep = prepare_dataframe(demo_dataset())
    res = analyze(df, "profit", ["advertising", "units_sold", "operating_expenses"], seed=0,
                  n_runs=1, evolution=FAST, prep_report=rep)
    m = res.models.set_index("model")
    assert set(m.index) == {"Mean baseline", "Linear regression", "Evolved equation"}
    assert m.loc["Linear regression", "val_r2"] > 0.8
    assert m.loc["Linear regression", "val_rmse"] < m.loc["Mean baseline", "val_rmse"]
    assert res.n_train + res.n_val == 150
    found = set(res.anomalies["row"])
    assert len(found & set(DEMO_ANOMALY_ROWS)) >= 2          # injected shocks are detected
    assert res.notices[0] == CAUSATION_NOTICE                 # causation warning is always first
    assert "Correlation is not causation" in res.notices[0]
    rec = res.to_record()
    assert rec["kind"] == "business" and rec["target"] == "profit"


@pytest.mark.parametrize("target,features", [
    ("nope", ["advertising"]), ("profit", []), ("profit", ["profit"]),
    ("profit", ["advertising", "advertising"]), ("profit", ["ghost"]),
])
def test_analyze_rejects_bad_selection(target, features):
    with pytest.raises(ValueError):
        analyze(demo_dataset(), target, features, evolution=FAST)


def test_analysis_reproducible():
    df = demo_dataset()
    kw = dict(target="revenue", features=["units_sold"], seed=3, n_runs=1, evolution=FAST)
    a, b = analyze(df, **kw), analyze(df, **kw)
    assert a.equation == b.equation
    pd.testing.assert_frame_equal(a.models, b.models)
