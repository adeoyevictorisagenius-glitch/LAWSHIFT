"""Subtree crossover with a depth limit."""
from __future__ import annotations

import numpy as np

from .expressions import Node, copy_tree, depth, get_subtrees, replace_subtree


def _pick_subtree(rng: np.random.Generator, tree: Node):
    subs = get_subtrees(tree)
    # prefer non-terminal points (90%) so crossover exchanges real structure
    nonterm = [s for s in subs if hasattr(s[0], "op")]
    pool = nonterm if nonterm and rng.random() < 0.9 else subs
    return pool[int(rng.integers(len(pool)))]


def subtree_crossover(
    rng: np.random.Generator, parent1: Node, parent2: Node, max_depth: int = 6, attempts: int = 5
) -> Node:
    """Return one child: a copy of parent1 with a random subtree replaced by one from parent2.

    If every attempt exceeds max_depth, a copy of parent1 is returned unchanged.
    """
    for _ in range(attempts):
        child = copy_tree(parent1)
        donor = copy_tree(parent2)
        target, parent, slot = _pick_subtree(rng, child)
        piece, _, _ = _pick_subtree(rng, donor)
        child = replace_subtree(child, target, parent, slot, piece)
        if depth(child) <= max_depth:
            return child
    return copy_tree(parent1)
