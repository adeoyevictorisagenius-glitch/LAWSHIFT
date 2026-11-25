"""Initial population: ramped half-and-half generation for structural diversity."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .expressions import DEFAULT_BINARY, DEFAULT_UNARY, Node, random_tree


def initial_population(
    rng: np.random.Generator,
    size: int,
    n_vars: int,
    max_depth: int = 4,
    binary_ops: Sequence[str] = DEFAULT_BINARY,
    unary_ops: Sequence[str] = DEFAULT_UNARY,
) -> List[Node]:
    if size < 2:
        raise ValueError("Population size must be at least 2")
    trees: List[Node] = []
    depths = list(range(2, max(max_depth, 2) + 1))
    for i in range(size):
        d = depths[i % len(depths)]
        method = "full" if (i // len(depths)) % 2 == 0 else "grow"
        trees.append(random_tree(rng, n_vars, d, method, binary_ops, unary_ops))
    return trees
