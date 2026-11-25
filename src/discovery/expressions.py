"""Safe symbolic expression trees for LAWSHIFT.

Expressions are plain Python objects (no eval, no string execution). Evaluation is
vectorised with NumPy and numerically protected:

* division by (near) zero returns 1.0 for that element
* exp() input is clipped, sqrt() acts on |x|
* every result is clipped to +/- MAX_ABS and NaN/inf are replaced

An expression is a tree of:
    Var(index)            -> input variable x{index}
    Const(value)          -> numeric constant
    Unary(op, child)      -> sin, cos, exp, sq, sqrt
    Binary(op, left, right) -> add, sub, mul, div
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

MAX_ABS = 1e12          # hard bound on any intermediate value
DIV_EPS = 1e-9          # |denominator| below this is treated as zero
EXP_CLIP = 30.0         # exp argument is clipped to [-EXP_CLIP, EXP_CLIP]
MAX_DEPTH_DEFAULT = 6

BINARY_OPS = ("add", "sub", "mul", "div")
UNARY_OPS = ("sin", "cos", "exp", "sq", "sqrt")
DEFAULT_BINARY = BINARY_OPS
DEFAULT_UNARY = UNARY_OPS

_BIN_SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def _clean(a: np.ndarray) -> np.ndarray:
    """Replace NaN/inf and clip to the allowed magnitude."""
    a = np.nan_to_num(a, nan=0.0, posinf=MAX_ABS, neginf=-MAX_ABS)
    return np.clip(a, -MAX_ABS, MAX_ABS)


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    small = np.abs(b) < DIV_EPS
    safe_b = np.where(small, 1.0, b)
    return np.where(small, 1.0, a / safe_b)


def apply_binary(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        if op == "add":
            r = a + b
        elif op == "sub":
            r = a - b
        elif op == "mul":
            r = a * b
        elif op == "div":
            r = safe_div(a, b)
        else:
            raise ValueError(f"Unknown binary operator: {op!r}")
    return _clean(r)


def apply_unary(op: str, a: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        if op == "sin":
            r = np.sin(a)
        elif op == "cos":
            r = np.cos(a)
        elif op == "exp":
            r = np.exp(np.clip(a, -EXP_CLIP, EXP_CLIP))
        elif op == "sq":
            r = a * a
        elif op == "sqrt":
            r = np.sqrt(np.abs(a))
        else:
            raise ValueError(f"Unknown unary operator: {op!r}")
    return _clean(r)


# --------------------------------------------------------------------------- nodes
@dataclass
class Var:
    index: int

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if self.index >= X.shape[1]:
            raise ValueError(f"Expression uses x{self.index} but data has {X.shape[1]} variables")
        return X[:, self.index].astype(float)

    def to_string(self, names: Optional[Sequence[str]] = None) -> str:
        return names[self.index] if names else f"x{self.index}"


@dataclass
class Const:
    value: float

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], float(np.clip(self.value, -MAX_ABS, MAX_ABS)))

    def to_string(self, names: Optional[Sequence[str]] = None) -> str:
        return f"{self.value:.4g}"


@dataclass
class Unary:
    op: str
    child: "Node"

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return apply_unary(self.op, self.child.evaluate(X))

    def to_string(self, names: Optional[Sequence[str]] = None) -> str:
        inner = self.child.to_string(names)
        if self.op == "sq":
            return f"({inner})^2"
        return f"{self.op}({inner})"


@dataclass
class Binary:
    op: str
    left: "Node"
    right: "Node"

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return apply_binary(self.op, self.left.evaluate(X), self.right.evaluate(X))

    def to_string(self, names: Optional[Sequence[str]] = None) -> str:
        return f"({self.left.to_string(names)} {_BIN_SYMBOL[self.op]} {self.right.to_string(names)})"


Node = Union[Var, Const, Unary, Binary]


# --------------------------------------------------------------------------- tree utilities
def size(node: Node) -> int:
    """Number of nodes in the tree (the complexity measure)."""
    if isinstance(node, (Var, Const)):
        return 1
    if isinstance(node, Unary):
        return 1 + size(node.child)
    return 1 + size(node.left) + size(node.right)


def depth(node: Node) -> int:
    if isinstance(node, (Var, Const)):
        return 1
    if isinstance(node, Unary):
        return 1 + depth(node.child)
    return 1 + max(depth(node.left), depth(node.right))


def copy_tree(node: Node) -> Node:
    if isinstance(node, Var):
        return Var(node.index)
    if isinstance(node, Const):
        return Const(node.value)
    if isinstance(node, Unary):
        return Unary(node.op, copy_tree(node.child))
    return Binary(node.op, copy_tree(node.left), copy_tree(node.right))


def get_subtrees(node: Node) -> List[Tuple[Node, Optional[Node], str]]:
    """All (subtree, parent, slot) triples in pre-order. slot in {'', 'child', 'left', 'right'}."""
    out: List[Tuple[Node, Optional[Node], str]] = []

    def walk(n: Node, parent: Optional[Node], slot: str) -> None:
        out.append((n, parent, slot))
        if isinstance(n, Unary):
            walk(n.child, n, "child")
        elif isinstance(n, Binary):
            walk(n.left, n, "left")
            walk(n.right, n, "right")

    walk(node, None, "")
    return out


def replace_subtree(root: Node, target: Node, parent: Optional[Node], slot: str, new: Node) -> Node:
    """Replace `target` (located via parent/slot) with `new`. Returns the (possibly new) root."""
    if parent is None:
        return new
    setattr(parent, slot, new)
    return root


def get_constants(node: Node) -> List[Const]:
    return [n for n, _, _ in get_subtrees(node) if isinstance(n, Const)]


def used_variables(node: Node) -> set:
    return {n.index for n, _, _ in get_subtrees(node) if isinstance(n, Var)}


# --------------------------------------------------------------------------- random generation
def random_tree(
    rng: np.random.Generator,
    n_vars: int,
    max_depth: int = 4,
    method: str = "grow",
    binary_ops: Sequence[str] = DEFAULT_BINARY,
    unary_ops: Sequence[str] = DEFAULT_UNARY,
    const_range: Tuple[float, float] = (-5.0, 5.0),
    p_terminal: float = 0.3,
) -> Node:
    """Grow or full random tree. Depth never exceeds max_depth."""
    if n_vars < 1:
        raise ValueError("n_vars must be >= 1")

    def terminal() -> Node:
        if rng.random() < 0.7:
            return Var(int(rng.integers(n_vars)))
        return Const(float(rng.uniform(*const_range)))

    def build(d: int) -> Node:
        if d >= max_depth:
            return terminal()
        if method == "grow" and d > 0 and rng.random() < p_terminal:
            return terminal()
        use_unary = len(unary_ops) > 0 and (len(binary_ops) == 0 or rng.random() < 0.25)
        if use_unary:
            return Unary(str(rng.choice(list(unary_ops))), build(d + 1))
        return Binary(str(rng.choice(list(binary_ops))), build(d + 1), build(d + 1))

    return build(0)


def evaluate(node: Node, X: np.ndarray) -> np.ndarray:
    """Evaluate on a 2-D array (n_samples, n_vars). Always returns finite, bounded values."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return _clean(node.evaluate(X))
