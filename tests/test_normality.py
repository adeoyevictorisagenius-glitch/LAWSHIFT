import numpy as np
import pandas as pd
import pytest

from src.data.normality import (VERDICT_LABELS, profile_column, profile_table, qq_points, strongest_pairs)
from src.discovery.diagnostics import residual_diagnostics


def test_verdicts_on_known_shapes():
    rng = np.random.default_rng(1)
    assert profile_column(rng.normal(50, 10, 400), "n").verdict in ("normal", "nearly_normal")
    ln = profile_column(rng.lognormal(3, 0.8, 400), "ln")
    assert ln.verdict == "skewed_right" and ln.skew > 1 and "log" in ln.suggestion.lower()
    assert profile_column(-rng.lognormal(3, 0.8, 400), "neg").verdict == "skewed_left"
    assert profile_column(rng.uniform(0, 1, 400), "u").verdict == "light_tails"
    assert profile_column(rng.standard_t(3, 600), "t").verdict in ("heavy_tails", "skewed_right", "skewed_left")


def test_profile_numbers_are_consistent():
    x = np.arange(1.0, 101.0)
    p = profile_column(x, "x")
    assert p.n == 100 and p.mean == pytest.approx(50.5) and p.median == pytest.approx(50.5)
    assert p.minimum == 1 and p.maximum == 100 and p.q1 < p.median < p.q3
    assert 0 < p.within_1sd < p.within_2sd <= p.within_3sd <= 1
    assert p.label == VERDICT_LABELS[p.verdict]


def test_outliers_are_counted():
    rng = np.random.default_rng(2)
    x = np.append(rng.normal(0, 1, 200), [15.0, -14.0])
    p = profile_column(x, "x")
    assert p.outliers_z3 >= 2 and p.outliers_iqr >= 2


def test_too_few_and_constant_values_are_handled():
    assert profile_column([1.0, 2.0, 3.0], "x").verdict == "too_few"
    p = profile_column(np.ones(50), "c")
    assert p.std == 0 and p.shapiro_p is None
    with pytest.raises(ValueError):
        profile_column([np.nan, np.inf], "bad")


def test_large_columns_are_subsampled_not_rejected():
    p = profile_column(np.random.default_rng(3).normal(size=20000), "big")
    assert p.shapiro_p is not None and p.n == 20000


def test_table_and_pairs_and_qq():
    rng = np.random.default_rng(4)
    a = rng.normal(size=150)
    df = pd.DataFrame({"a": a, "b": 2 * a + rng.normal(0, 0.1, 150), "c": rng.normal(size=150)})
    profiles, table = profile_table(df)
    assert [p.name for p in profiles] == ["a", "b", "c"] and len(table) == 3
    top = strongest_pairs(df.corr(), 2)
    assert {top[0][0], top[0][1]} == {"a", "b"} and top[0][2] > 0.99 and len(top) == 2
    osm, osr, r = qq_points(a)
    assert len(osm) == len(osr) == 150 and r > 0.98


def test_residual_diagnostics_detects_bias_and_growing_errors():
    rng = np.random.default_rng(5)
    y = rng.uniform(10, 100, 300)
    fair = residual_diagnostics(y, y + rng.normal(0, 2, 300))
    assert not fair.biased and not fair.errors_grow and fair.errors_look_normal
    biased = residual_diagnostics(y, y - 5 + rng.normal(0, 2, 300))
    assert biased.biased and biased.mean_error > 4
    pred = y + rng.normal(0, 1, 300) * (0.02 * y)           # error proportional to the prediction
    grows = residual_diagnostics(y, pred)
    assert grows.errors_grow and grows.hetero_rho > 0.3
    assert fair.interval95 == pytest.approx(1.96 * 2, rel=0.25)
    assert len(fair.worst_rows) == 3


def test_residual_diagnostics_rejects_tiny_input():
    with pytest.raises(ValueError):
        residual_diagnostics([1, 2, 3], [1, 2, 3])
