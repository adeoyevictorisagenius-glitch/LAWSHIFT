"""Equation simplification and display.

Two levels:

1. `simplify_tree` - pure NumPy/Python, always available. Folds constants, removes
   identities (x+0, x*1, x*0, x/1, x-x, x/x) and moves constants to the front
   (e.g. cos(3t)/0.4998 -> 2.001 * cos(3t)). It preserves the function exactly, which
   the tests check numerically.
2. `sympy_form` - optional. If SymPy is installed it expands the expression
   algebraically (e.g. (t - 0.2*v0)*(-4.9*t) -> 0.98*t*v0 - 4.9*t**2). Returns None if
   SymPy is missing or conversion fails. Display only: never used for the search.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .expressions import DIV_EPS, Binary, Const, Node, Unary, Var, apply_binary, apply_unary, copy_tree


def _fold_unary(op: str, c: Const) -> Const:
    return Const(float(apply_unary(op, np.array([c.value]))[0]))


def _fold_binary(op: str, a: Const, b: Const) -> Const:
    return Const(float(apply_binary(op, np.array([a.value]), np.array([b.value]))[0]))


def _split_const(n: Node) -> Tuple[float, Node]:
    """Split a node into (leading constant factor, remainder): c * rest -> (c, rest)."""
    if isinstance(n, Binary) and n.op == "mul" and isinstance(n.left, Const):
        return n.left.value, n.right
    return 1.0, n


def _with_const(c: float, rest: Node) -> Node:
    return rest if c == 1.0 else Binary("mul", Const(c), rest)


def _simp(n: Node) -> Node:
    if isinstance(n, (Var, Const)):
        return copy_tree(n)
    if isinstance(n, Unary):
        c = _simp(n.child)
        if isinstance(c, Const):
            return _fold_unary(n.op, c)
        # pull constant factors outside squares and roots: (k*x)^2 = k^2 * x^2, sqrt|k*x| = sqrt|k| * sqrt|x|
        if n.op in ("sq", "sqrt"):
            k, rest = _split_const(c)
            if k != 1.0 and rest is not c:
                kk = k * k if n.op == "sq" else float(np.sqrt(abs(k)))
                return _with_const(kk, Unary(n.op, rest))
        return Unary(n.op, c)

    l, r = _simp(n.left), _simp(n.right)
    op = n.op
    if isinstance(l, Const) and isinstance(r, Const):
        return _fold_binary(op, l, r)

    if op == "add":
        if isinstance(l, Const) and l.value == 0:
            return r
        if isinstance(r, Const) and r.value == 0:
            return l
        if isinstance(r, Const):          # constants first
            l, r = r, l
    elif op == "sub":
        if isinstance(r, Const) and r.value == 0:
            return l
        if l.to_string() == r.to_string():
            return Const(0.0)
    elif op == "mul":
        if (isinstance(l, Const) and l.value == 0) or (isinstance(r, Const) and r.value == 0):
            return Const(0.0)
        if isinstance(l, Const) and l.value == 1:
            return r
        if isinstance(r, Const) and r.value == 1:
            return l
        if isinstance(r, Const):
            l, r = r, l
        if isinstance(l, Const) and isinstance(r, Binary) and r.op == "mul" and isinstance(r.left, Const):
            return _simp(Binary("mul", _fold_binary("mul", l, r.left), r.right))
        # (c1*a) * (c2*b) -> (c1*c2) * (a*b);  a * (c*b) -> c * (a*b)
        k1, a = _split_const(l)
        k2, b = _split_const(r)
        if (a is not l or b is not r) and not isinstance(l, Const):
            return _simp(_with_const(k1 * k2, Binary("mul", a, b)))
    elif op == "div":
        if isinstance(r, Const) and r.value == 1:
            return l
        if isinstance(r, Const) and abs(r.value) >= DIV_EPS:
            return _simp(Binary("mul", Const(1.0 / r.value), l))
        if l.to_string() == r.to_string():
            return Const(1.0)   # safe division returns 1 for x/x everywhere, including x=0
        # (c1*a) / (c2*b) -> (c1/c2) * (a/b)   (equal except where the denominator is exactly 0)
        k1, a = _split_const(l)
        k2, b = _split_const(r)
        if (a is not l or b is not r) and abs(k2) >= 1e-6:
            return _simp(_with_const(k1 / k2, Binary("div", a, b)))
    return Binary(op, l, r)


def simplify_tree(tree: Node, max_passes: int = 6) -> Node:
    """Return an equivalent, tidier copy of the tree."""
    cur = copy_tree(tree)
    for _ in range(max_passes):
        nxt = _simp(cur)
        if nxt.to_string() == cur.to_string():
            return nxt
        cur = nxt
    return cur


def strip_outer_parens(s: str) -> str:
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        depth = 0
        for i, ch in enumerate(s):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and i < len(s) - 1:
                return s
        return s[1:-1]
    return s


def format_equation(tree: Node, names: Optional[Sequence[str]], output_name: str) -> str:
    return f"{output_name} = {strip_outer_parens(simplify_tree(tree).to_string(names))}"


def _to_sympy(node: Node, names: Sequence[str], sp):
    if isinstance(node, Var):
        return sp.Symbol(names[node.index], real=True)
    if isinstance(node, Const):
        return sp.Float(node.value)
    if isinstance(node, Unary):
        a = _to_sympy(node.child, names, sp)
        return {
            "sin": sp.sin, "cos": sp.cos, "exp": sp.exp,
            "sq": lambda z: z ** 2, "sqrt": lambda z: sp.sqrt(sp.Abs(z)),
        }[node.op](a)
    a, b = _to_sympy(node.left, names, sp), _to_sympy(node.right, names, sp)
    return {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[node.op]


def sympy_expr(tree: Node, names: Sequence[str], digits: int = 4):
    """Algebraically expanded SymPy expression (floats rounded), or None if SymPy is unavailable/fails."""
    try:
        import sympy as sp
    except ImportError:
        return None
    try:
        expr = sp.expand(_to_sympy(simplify_tree(tree), names, sp))
        def tidy(f):                      # `digits` significant digits; values that round to an integer become integers
            r = float(f"{float(f):.{digits}g}")
            return sp.Integer(int(r)) if r == int(r) and abs(r) < 1e15 else sp.Float(r, digits)

        return expr.xreplace({f: tidy(f) for f in expr.atoms(sp.Float)})
    except Exception:
        return None


def sympy_form(tree: Node, names: Sequence[str], digits: int = 4) -> Optional[str]:
    """Algebraically expanded form via SymPy, or None if SymPy is unavailable/fails."""
    expr = sympy_expr(tree, names, digits)
    return None if expr is None else str(expr)
