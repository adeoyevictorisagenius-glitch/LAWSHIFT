"""Mutation operators: subtree, point, constant perturbation and hoist."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .expressions import (
    BINARY_OPS, UNARY_OPS, Binary, Const, DEFAULT_BINARY, DEFAULT_UNARY, Node, Unary, Var,
    copy_tree, depth, get_subtrees, random_tree, replace_subtree,
)


def subtree_mutation(rng, tree, n_vars, max_depth=6, binary_ops=DEFAULT_BINARY, unary_ops=DEFAULT_UNARY) -> Node:
    child = copy_tree(tree)
    subs = get_subtrees(child)
    target, parent, slot = subs[int(rng.integers(len(subs)))]
    new = random_tree(rng, n_vars, max_depth=3, method="grow", binary_ops=binary_ops, unary_ops=unary_ops)
    child = replace_subtree(child, target, parent, slot, new)
    return child if depth(child) <= max_depth else copy_tree(tree)


def point_mutation(rng, tree, n_vars, binary_ops=DEFAULT_BINARY, unary_ops=DEFAULT_UNARY) -> Node:
    """Change one node without altering the tree's shape."""
    child = copy_tree(tree)
    subs = get_subtrees(child)
    node, parent, slot = subs[int(rng.integers(len(subs)))]
    if isinstance(node, Var):
        node.index = int(rng.integers(n_vars))
    elif isinstance(node, Const):
        node.value = float(rng.uniform(-5, 5))
    elif isinstance(node, Unary) and unary_ops:
        node.op = str(rng.choice(list(unary_ops)))
    elif isinstance(node, Binary) and binary_ops:
        node.op = str(rng.choice(list(binary_ops)))
    return child


def constant_mutation(rng, tree) -> Node:
    """Perturb one constant (multiplicative + small additive Gaussian noise)."""
    child = copy_tree(tree)
    consts = [n for n, _, _ in get_subtrees(child) if isinstance(n, Const)]
    if not consts:
        return child
    c = consts[int(rng.integers(len(consts)))]
    c.value = float(c.value * (1.0 + rng.normal(0, 0.2)) + rng.normal(0, 0.1))
    return child


def hoist_mutation(rng, tree) -> Node:
    """Replace the tree by one of its own subtrees (a strong anti-bloat operator)."""
    subs = get_subtrees(tree)
    node, _, _ = subs[int(rng.integers(len(subs)))]
    return copy_tree(node)


def mutate(rng, tree, n_vars, max_depth=6, binary_ops=DEFAULT_BINARY, unary_ops=DEFAULT_UNARY) -> Node:
    r = rng.random()
    if r < 0.35:
        return subtree_mutation(rng, tree, n_vars, max_depth, binary_ops, unary_ops)
    if r < 0.60:
        return point_mutation(rng, tree, n_vars, binary_ops, unary_ops)
    if r < 0.85:
        return constant_mutation(rng, tree)
    return hoist_mutation(rng, tree)
