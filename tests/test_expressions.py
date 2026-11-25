import numpy as np
import pytest

from src.discovery.expressions import (
    Binary, Const, Unary, Var, MAX_ABS, copy_tree, depth, evaluate,
    get_subtrees, random_tree, size,
)
from src.utils.metrics import mse, r2, rmse


X = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 5.0]])


def test_basic_arithmetic():
    e = Binary("add", Binary("mul", Var(0), Var(1)), Const(1.0))   # x0*x1 + 1
    np.testing.assert_allclose(evaluate(e, X), [3.0, 13.0, 1.0])


def test_division_by_zero_is_safe():
    e = Binary("div", Var(1), Var(0))   # x1 / x0 with x0 = 0 in row 3
    out = evaluate(e, X)
    assert np.all(np.isfinite(out))
    assert out[2] == 1.0
    np.testing.assert_allclose(out[:2], [2.0, 4.0 / 3.0])


def test_exp_overflow_is_bounded():
    e = Unary("exp", Binary("mul", Const(1e6), Var(1)))
    out = evaluate(e, X)
    assert np.all(np.isfinite(out))
    assert np.all(np.abs(out) <= MAX_ABS)


def test_nested_overflow_is_bounded():
    e = Var(1)
    for _ in range(10):
        e = Binary("mul", e, Const(1e5))
    out = evaluate(e, X)
    assert np.all(np.isfinite(out)) and np.all(np.abs(out) <= MAX_ABS)


def test_unary_ops():
    np.testing.assert_allclose(evaluate(Unary("sq", Var(0)), X), [1, 9, 0])
    np.testing.assert_allclose(evaluate(Unary("sqrt", Const(-4.0)), X), [2, 2, 2])
    np.testing.assert_allclose(evaluate(Unary("sin", Var(0)), X), np.sin(X[:, 0]))
    np.testing.assert_allclose(evaluate(Unary("cos", Var(0)), X), np.cos(X[:, 0]))


def test_unknown_operator_rejected():
    with pytest.raises(ValueError):
        evaluate(Binary("pow", Var(0), Var(1)), X)
    with pytest.raises(ValueError):
        evaluate(Unary("tan", Var(0)), X)


def test_variable_out_of_range_rejected():
    with pytest.raises(ValueError):
        evaluate(Var(5), X)


def test_to_string_uses_names():
    e = Binary("mul", Var(0), Unary("sq", Var(1)))
    assert e.to_string(["m", "t"]) == "(m * (t)^2)"
    assert e.to_string() == "(x0 * (x1)^2)"


def test_size_depth_copy_independent():
    e = Binary("add", Var(0), Unary("sin", Const(2.0)))
    assert size(e) == 4 and depth(e) == 3
    c = copy_tree(e)
    c.left = Const(9.0)
    assert isinstance(e.left, Var)


def test_get_subtrees_count_matches_size():
    rng = np.random.default_rng(0)
    t = random_tree(rng, 2, max_depth=4)
    assert len(get_subtrees(t)) == size(t)


@pytest.mark.parametrize("method", ["grow", "full"])
def test_random_trees_valid_and_bounded(method):
    rng = np.random.default_rng(42)
    Xr = rng.normal(size=(50, 3)) * 100
    for _ in range(300):
        t = random_tree(rng, 3, max_depth=5, method=method)
        assert depth(t) <= 6
        out = evaluate(t, Xr)
        assert out.shape == (50,)
        assert np.all(np.isfinite(out)) and np.all(np.abs(out) <= MAX_ABS)


def test_random_tree_reproducible():
    a = random_tree(np.random.default_rng(7), 2, max_depth=4).to_string()
    b = random_tree(np.random.default_rng(7), 2, max_depth=4).to_string()
    assert a == b


def test_metrics():
    y = np.array([1.0, 2.0, 3.0])
    assert mse(y, y) == 0.0
    assert rmse(y, y + 1) == pytest.approx(1.0)
    assert r2(y, y) == pytest.approx(1.0)
    assert r2(np.ones(3), np.ones(3)) == 0.0
    with pytest.raises(ValueError):
        mse(y, y[:2])
