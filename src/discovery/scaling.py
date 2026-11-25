"""Exact conversion between original units and the scaled space the search runs in.

Real datasets can have columns of any magnitude (1e-3 ... 1e6), which makes random
search hard. The engine therefore searches on rescaled columns, and the winning tree is
rewritten EXACTLY in the original units:

    z_i = (x_i - mx_i) / sx_i            (mx_i = 0 when centring is off)
    y   = sy * f(z) + my

`unscale_tree` substitutes these expressions into the tree, so the returned tree can be
evaluated directly on raw data and gives the same predictions as sy * f(z) + my.
Scaling statistics must come from TRAINING rows only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .expressions import Binary, Const, Node, Var, copy_tree

MAX_ABS_VALUE = 1e9     # data beyond this risks hitting the evaluator's overflow bound


@dataclass
class Scaler:
    sx: np.ndarray                       # per-input scale (std)
    sy: float                            # output scale (std)
    mx: Optional[np.ndarray] = None      # per-input centre (None = no centring)
    my: float = 0.0                      # output centre

    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray, center: bool = False) -> "Scaler":
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        if np.max(np.abs(X)) > MAX_ABS_VALUE or np.max(np.abs(y)) > MAX_ABS_VALUE:
            raise ValueError(f"Values larger than {MAX_ABS_VALUE:g} are not supported; "
                             "rescale the data (for example, change units).")
        sx = X.std(axis=0)
        if np.any(sx == 0):
            raise ValueError("A predictor is constant in the training data and cannot be used.")
        sy = float(y.std())
        if sy == 0:
            raise ValueError("The target is constant in the training data.")
        return Scaler(sx=sx, sy=sy, mx=X.mean(axis=0) if center else None, my=float(y.mean()) if center else 0.0)

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        return (X - (self.mx if self.mx is not None else 0.0)) / self.sx

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, float) - self.my) / self.sy


def unscale_tree(tree: Node, scaler: Scaler) -> Node:
    """Return a new tree on ORIGINAL units, equal to  scaler.sy * tree(z) + scaler.my."""
    new = copy_tree(tree)

    def sub(node: Node) -> Node:
        if isinstance(node, Var):
            v: Node = Var(node.index)
            if scaler.mx is not None:
                v = Binary("sub", v, Const(float(scaler.mx[node.index])))
            return Binary("mul", v, Const(1.0 / float(scaler.sx[node.index])))
        if isinstance(node, Const):
            return node
        if hasattr(node, "child"):
            node.child = sub(node.child)
            return node
        node.left, node.right = sub(node.left), sub(node.right)
        return node

    out: Node = Binary("mul", Const(scaler.sy), sub(new))
    if scaler.mx is not None or scaler.my != 0.0:
        out = Binary("add", out, Const(scaler.my))
    return out
