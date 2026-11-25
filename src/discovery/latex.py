"""LaTeX output for equations (display only - nothing here is used by the search).

* `equation_latex`   - a discovered expression tree as `target = ...` in LaTeX
* `sympy_latex`      - the algebraically expanded form (needs SymPy; None if unavailable)
* `linear_latex`     - a linear-regression line
* `FORMULAS`         - the definitions of the metrics the app reports, for the "show the maths" panels

The renderer mirrors the evaluator exactly. The evaluator's square root acts on |x|, so the bars are
drawn unless every quantity under the root is known to be non-negative (variables whose observed
values are all >= 0). Division is shown as an ordinary fraction; the evaluator's protection against
dividing by (nearly) zero is an implementation detail and is not part of the displayed law.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np

from .expressions import Binary, Const, Node, Unary, Var
from .simplify import simplify_tree, sympy_expr

_LEFT, _RIGHT = r"\left(", r"\right)"


# --------------------------------------------------------------------------- atoms
def latex_name(name: str) -> str:
    """Variable name -> LaTeX. 'F' -> F, 'v0' -> v_{0}, 'Pressure_kPa' -> \\mathrm{Pressure\\_kPa}."""
    if re.fullmatch(r"[A-Za-z]", name):
        return name
    m = re.fullmatch(r"([A-Za-z])(\d+)", name)
    if m:
        return f"{m.group(1)}_{{{m.group(2)}}}"
    return r"\mathrm{" + name.replace("_", r"\_") + "}"


def latex_number(v: float, digits: int = 4) -> str:
    """Compact number: 4.905, 0.001234, 1.5 \\times 10^{-7}."""
    if v == 0:
        return "0"
    s = f"{v:.{digits}g}"
    if "e" in s:
        mant, exp = s.split("e")
        return rf"{mant} \times 10^{{{int(exp)}}}"
    return s


def _is_nonneg(node: Node, nonneg: Set[int]) -> bool:
    if isinstance(node, Var):
        return node.index in nonneg
    if isinstance(node, Const):
        return node.value >= 0
    if isinstance(node, Unary):
        return node.op in ("sq", "sqrt", "exp")
    if node.op in ("mul", "div", "add"):
        return _is_nonneg(node.left, nonneg) and _is_nonneg(node.right, nonneg)
    return False


def _wrap(s: str) -> str:
    return f"{_LEFT}{s}{_RIGHT}"


# --------------------------------------------------------------------------- tree -> LaTeX
# precedence of a rendered piece: 0 = sum (needs brackets inside a product), 1 = product, 2 = atom
def _render(node: Node, names: Sequence[str], nonneg: Set[int]) -> Tuple[str, int]:
    if isinstance(node, Var):
        return latex_name(names[node.index]), 2
    if isinstance(node, Const):
        return latex_number(node.value), (2 if node.value >= 0 else 1)

    if isinstance(node, Unary):
        inner, _ = _render(node.child, names, nonneg)
        if node.op in ("sin", "cos"):
            return rf"\{node.op}{_wrap(inner)}", 2
        if node.op == "exp":
            return f"e^{{{inner}}}", 2
        if node.op == "sqrt":
            body = inner if _is_nonneg(node.child, nonneg) else rf"\left|{inner}\right|"
            return rf"\sqrt{{{body}}}", 2
        # square
        c = node.child
        bare = isinstance(c, Var) or (isinstance(c, Const) and c.value >= 0) or \
            (isinstance(c, Unary) and c.op in ("sin", "cos"))
        return f"{inner if bare else _wrap(inner)}^{{2}}", 2

    op = node.op
    if op == "div":
        a, _ = _render(node.left, names, nonneg)
        b, _ = _render(node.right, names, nonneg)
        return rf"\frac{{{a}}}{{{b}}}", 2

    if op in ("add", "sub"):
        a, _ = _render(node.left, names, nonneg)
        b, pb = _render(node.right, names, nonneg)
        negative = b.startswith("-")
        if op == "sub":
            if pb == 0 and not negative:
                b = _wrap(b)
            elif negative:
                return f"{a} + {b[1:]}", 0
            return f"{a} - {b}", 0
        return (f"{a} - {b[1:]}" if negative else f"{a} + {b}"), 0

    # product
    a, pa = _render(node.left, names, nonneg)
    b, pb = _render(node.right, names, nonneg)
    if isinstance(node.left, Const):
        coef = latex_number(node.left.value)
        if coef == "1":                                   # a coefficient that displays as 1 is not shown
            return b, pb
        if coef == "-1":
            return (b[1:] if b.startswith("-") else f"-{_wrap(b) if pb == 0 else b}"), 1
    if pa == 0:
        a = _wrap(a)
    if pb == 0 or b.startswith("-"):
        b = _wrap(b)
    sep = r" \cdot " if b[:1].isdigit() else r"\,"
    return f"{a}{sep}{b}", 1


def tree_latex(tree: Node, names: Sequence[str], nonneg_vars: Optional[Set[int]] = None) -> str:
    """Right-hand side only, after simplification."""
    text, _ = _render(simplify_tree(tree), names, set(nonneg_vars or ()))
    return text


def equation_latex(tree: Node, names: Sequence[str], target: str, nonneg_vars: Optional[Set[int]] = None) -> str:
    """`target = ...` in LaTeX."""
    return f"{latex_name(target)} = {tree_latex(tree, names, nonneg_vars)}"


def nonneg_from_data(X: np.ndarray) -> Set[int]:
    """Indices of columns whose observed values are all >= 0 (lets sqrt drop its absolute-value bars)."""
    X = np.asarray(X, float)
    return {j for j in range(X.shape[1]) if float(np.min(X[:, j])) >= 0.0}


# --------------------------------------------------------------------------- expanded form
def sympy_latex(tree: Node, names: Sequence[str], target: str, digits: int = 4) -> Optional[str]:
    """Expanded form in LaTeX via SymPy, or None if SymPy is missing or fails."""
    expr = sympy_expr(tree, names, digits)
    if expr is None:
        return None
    try:
        import sympy as sp
        mapping = {sp.Symbol(n, real=True): latex_name(n) for n in names}
        return f"{latex_name(target)} = {sp.latex(expr, symbol_names=mapping)}"
    except Exception:
        return None


# --------------------------------------------------------------------------- linear model
def linear_latex(coef: Sequence[float], features: Sequence[str], target: str) -> str:
    """y = b + c1 x1 + c2 x2 ... for a regression whose last coefficient is the intercept."""
    coef = list(map(float, coef))
    parts = []
    for c, f in zip(coef[:-1], features):
        mag = latex_number(abs(c))
        parts.append((c < 0, rf"{mag}\,{latex_name(f)}"))
    b = coef[-1]
    parts.append((b < 0, latex_number(abs(b))))
    out = ""
    for i, (neg, text) in enumerate(parts):
        out += (f"-{text}" if neg else text) if i == 0 else (f" - {text}" if neg else f" + {text}")
    return f"{latex_name(target)} = {out}"


# --------------------------------------------------------------------------- explanatory formulas
FORMULAS: Dict[str, Tuple[str, str]] = {
    # key: (plain-language label, LaTeX)
    "r2": ("R² - share of the ups and downs the equation explains",
           r"R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}"),
    "rmse": ("RMSE - typical size of a miss, in the units of your variable",
             r"\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}"),
    "mae": ("MAE - average size of a miss",
            r"\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}\left|y_i-\hat y_i\right|"),
    "nrmse": ("NRMSE - RMSE compared with the natural spread of the data",
              r"\mathrm{NRMSE} = \frac{\mathrm{RMSE}}{\sigma_y}"),
    "fitness": ("Fitness - what the evolutionary search minimises (accuracy plus a penalty for size)",
                r"\mathrm{fitness} = \mathrm{NRMSE} + \lambda\,N_{\mathrm{nodes}},\qquad \lambda = 0.001"),
    "importance": ("Permutation importance - how much shuffling one variable hurts the predictions",
                   r"I_j=\max\left(0,\ \mathrm{RMSE}_{\mathrm{shuffled}\ j}-\mathrm{RMSE}_{\mathrm{base}}\right),"
                   r"\qquad s_j=\frac{I_j}{\sum_k I_k}"),
    "elasticity": ("Elasticity - % change in the outcome for a 10% nudge to one variable",
                   r"E_j = 100\cdot\frac{f(1.1\,x_j)-f(x_j)}{\left|f(x_j)\right|}\ \ [\%]"),
    "interval": ("Rough 95% range for a single prediction",
                 r"\hat y \pm 1.96\,\hat\sigma_{\mathrm{resid}}"),
    "zscore": ("z-score - how many standard deviations a value sits from the mean",
               r"z_i = \frac{x_i-\bar x}{s}"),
    "normal_pdf": ("Normal (bell-curve) density",
                   r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}\,e^{-\frac{(x-\mu)^2}{2\sigma^2}}"),
    "empirical_rule": ("Empirical rule - what a normal variable should look like",
                       r"P(\mu-k\sigma<X<\mu+k\sigma)\approx 68\%,\ 95\%,\ 99.7\%\quad (k=1,2,3)"),
    "skewness": ("Skewness - 0 for a symmetric shape, positive when a long tail points right",
                 r"g_1=\frac{\frac{1}{n}\sum_i (x_i-\bar x)^3}{s^3}"),
    "kurtosis": ("Excess kurtosis - 0 for a bell curve, positive when tails are heavy",
                 r"g_2=\frac{\frac{1}{n}\sum_i (x_i-\bar x)^4}{s^4}-3"),
    "shapiro": ("Shapiro-Wilk statistic - close to 1 means 'looks normal'",
                r"W=\frac{\left(\sum_i a_i\,x_{(i)}\right)^2}{\sum_i (x_i-\bar x)^2}"),
    "robust_z": ("Robust z-score used to flag unusual observations",
                 r"z^{*}_i = 0.6745\,\frac{r_i-\mathrm{median}(r)}{\mathrm{MAD}(r)}"),
}

OBJECTIVES: Dict[str, Tuple[str, str]] = {
    "maximize": ("As high as possible", r"\max_{x\in B}\ f(x)"),
    "minimize": ("As low as possible", r"\min_{x\in B}\ f(x)"),
    "target": ("Close to a specific value t", r"\min_{x\in B}\ \left(f(x)-t\right)^2"),
    "range": ("Inside the range [a, b]", r"\min_{x\in B}\ \left(\max\left(0,\ a-f(x),\ f(x)-b\right)\right)^2"),
}
