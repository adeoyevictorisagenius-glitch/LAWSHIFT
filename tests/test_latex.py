import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from matplotlib.mathtext import MathTextParser

from src.discovery.expressions import Binary, Const, Unary, Var, random_tree
from src.discovery.latex import (FORMULAS, OBJECTIVES, equation_latex, latex_name, latex_number, linear_latex,
                                 nonneg_from_data, sympy_latex, tree_latex)

_PARSER = MathTextParser("path")
NAMES = ["Temperature_K", "Volume_L", "t", "v0"]


def parses(tex: str) -> bool:
    """matplotlib's mathtext is an independent LaTeX-subset parser: it raises on malformed input."""
    _PARSER.parse(f"${tex}$")
    return True


def test_names_and_numbers():
    assert latex_name("F") == "F"
    assert latex_name("v0") == "v_{0}"
    assert latex_name("Pressure_kPa") == r"\mathrm{Pressure\_kPa}"
    assert latex_number(4.905) == "4.905"
    assert latex_number(0) == "0"
    assert "10^{-7}" in latex_number(1.5e-7)


def test_gas_law_is_a_fraction():
    t = Binary("mul", Const(8.314), Binary("div", Binary("mul", Var(1), Var(0)), Var(2)))
    tex = equation_latex(t, ["T", "n", "V"], "P")
    assert r"\frac{" in tex and tex.startswith("P = 8.314")
    assert parses(tex)


def test_projectile_signs_and_powers():
    t = Binary("sub", Binary("mul", Var(0), Var(1)), Binary("mul", Const(4.905), Unary("sq", Var(1))))
    tex = equation_latex(t, ["v0", "t"], "y")
    assert "t^{2}" in tex and " - 4.905" in tex and "v_{0}" in tex
    assert parses(tex)


def test_negative_constants_do_not_leave_double_signs():
    tex = tree_latex(Binary("add", Binary("mul", Const(-3.2), Var(0)), Const(-7.0)), ["x"])
    assert "+ -" not in tex and "- -" not in tex and "--" not in tex
    assert parses(tex)


def test_coefficient_that_displays_as_one_is_dropped():
    tex = tree_latex(Binary("mul", Const(1.00001), Binary("add", Var(0), Var(1))), ["a", "b"])
    assert tex == "a + b"
    tex = tree_latex(Binary("mul", Const(-1.00001), Binary("add", Var(0), Var(1))), ["a", "b"])
    assert tex == r"-\left(a + b\right)"


def test_sqrt_bars_only_when_sign_is_unknown():
    t = Unary("sqrt", Var(0))
    assert r"\left|" in tree_latex(t, ["L"])                       # unknown sign: evaluator uses |x|
    assert r"\left|" not in tree_latex(t, ["L"], nonneg_vars={0})  # data are all >= 0: bars are redundant
    X = np.array([[1.0, -2.0], [3.0, 4.0]])
    assert nonneg_from_data(X) == {0}


def test_exp_square_are_bracketed_safely():
    assert tree_latex(Unary("sq", Binary("add", Var(0), Var(1))), ["a", "b"]).startswith(r"\left(")
    tex = tree_latex(Unary("sq", Unary("exp", Var(0))), ["a"])      # e^{a}^{2} would be invalid LaTeX
    assert tex.startswith(r"\left(e^{a}") and parses(tex)


def test_random_trees_always_produce_valid_latex():
    rng = np.random.default_rng(0)
    for _ in range(300):
        tex = equation_latex(random_tree(rng, 4, max_depth=5), NAMES, "y", nonneg_vars={0, 1})
        assert tex.count("{") == tex.count("}") and parses(tex)


def test_sympy_latex_and_linear_latex():
    pytest.importorskip("sympy")
    t = Binary("sub", Binary("mul", Var(0), Var(1)), Binary("mul", Const(4.905), Unary("sq", Var(1))))
    tex = sympy_latex(t, ["v0", "t"], "y")
    assert tex is not None and "4.905" in tex and parses(tex)
    lin = linear_latex([2.5, -0.31, 12.0], ["a", "Bb"], "y")
    assert lin == r"y = 2.5\,a - 0.31\,\mathrm{Bb} + 12" and parses(lin)


def test_tiny_coefficients_survive_sympy_rounding():
    pytest.importorskip("sympy")
    from src.discovery.simplify import sympy_form
    assert "e-7" in sympy_form(Binary("mul", Const(1.5e-7), Var(0)), ["x"])


def test_every_explanatory_formula_is_valid_latex():
    for key, (label, tex) in {**FORMULAS, **OBJECTIVES}.items():
        assert label and parses(tex), key
