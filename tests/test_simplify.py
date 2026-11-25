import numpy as np
import pytest

from src.discovery.expressions import Binary, Const, Unary, Var, evaluate, random_tree, size
from src.discovery.simplify import format_equation, simplify_tree, strip_outer_parens, sympy_form


def test_constant_folding_and_identities():
    e = Binary("add", Binary("mul", Const(2.0), Const(3.0)), Var(0))      # 6 + x0
    s = simplify_tree(e)
    assert s.to_string() == "(6 + x0)"
    assert simplify_tree(Binary("mul", Var(0), Const(1.0))).to_string() == "x0"
    assert simplify_tree(Binary("add", Const(0.0), Var(0))).to_string() == "x0"
    assert simplify_tree(Binary("sub", Var(0), Var(0))).to_string() == "0"
    assert simplify_tree(Binary("div", Var(1), Var(1))).to_string() == "1"
    assert simplify_tree(Binary("mul", Var(0), Const(0.0))).to_string() == "0"


def test_division_by_constant_becomes_multiplication():
    e = Binary("div", Unary("cos", Binary("mul", Var(0), Const(3.0))), Const(0.5))
    s = simplify_tree(e)
    assert s.to_string() == "(2 * cos((3 * x0)))"


def test_nested_constant_multiplication_merges():
    e = Binary("mul", Const(2.0), Binary("mul", Const(3.0), Var(0)))
    assert simplify_tree(e).to_string() == "(6 * x0)"


def test_does_not_divide_by_tiny_constant():
    e = Binary("div", Var(0), Const(1e-12))
    s = simplify_tree(e)
    X = np.array([[1.0], [2.0]])
    np.testing.assert_allclose(evaluate(s, X), evaluate(e, X))


def test_simplification_preserves_function_on_random_trees():
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, (40, 2))
    for _ in range(300):
        t = random_tree(rng, 2, max_depth=5)
        s = simplify_tree(t)
        np.testing.assert_allclose(evaluate(s, X), evaluate(t, X), rtol=1e-9, atol=1e-9)
        assert size(s) <= size(t) + 1


def test_original_tree_not_mutated():
    e = Binary("div", Var(0), Const(2.0))
    before = e.to_string()
    simplify_tree(e)
    assert e.to_string() == before


def test_format_equation_and_parens():
    e = Binary("mul", Var(0), Var(1))
    assert format_equation(e, ["F", "m"], "a") == "a = F * m"
    assert strip_outer_parens("(a + b)") == "a + b"
    assert strip_outer_parens("(a) + (b)") == "(a) + (b)"


def test_sympy_form_expands_if_available():
    sp = pytest.importorskip("sympy")
    e = Binary("mul", Binary("sub", Var(1), Binary("mul", Var(0), Const(0.2))), Binary("mul", Var(1), Const(-4.9)))
    out = sympy_form(e, ["v0", "t"])
    assert out is not None and "t**2" in out
    # Result must be numerically equivalent
    v0, t = sp.symbols("v0 t", real=True)
    expr = sp.sympify(out, locals={"v0": v0, "t": t})
    expected = (t - 0.2 * v0) * (-4.9 * t)
    assert abs(float((expr - expected).subs({v0: 3.0, t: 1.7}))) < 1e-2
