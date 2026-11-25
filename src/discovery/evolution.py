"""Evolutionary symbolic regression engine.

Generational GA with tournament selection, subtree crossover, several mutation
operators, elitism and a complexity penalty. The engine only ever sees (X, y):
it is never given the generating equation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .crossover import subtree_crossover
from .expressions import DEFAULT_BINARY, DEFAULT_UNARY, Node, copy_tree, evaluate
from .fitness import BAD_FITNESS, Individual, evaluate_individual
from .mutation import mutate
from .population import initial_population
from .selection import tournament_select

BINARY_ALLOWED = {"add", "sub", "mul", "div"}
UNARY_ALLOWED = {"sin", "cos", "exp", "sq", "sqrt"}


@dataclass
class EvolutionConfig:
    pop_size: int = 200
    generations: int = 40
    tournament_size: int = 5
    crossover_rate: float = 0.7
    mutation_rate: float = 0.25
    elite_count: int = 2
    max_depth: int = 6
    init_depth: int = 4
    parsimony: float = 0.001
    tune_constants: bool = True
    patience: int = 15              # stop after this many generations without improvement
    min_improvement: float = 1e-6
    seed: int = 0
    binary_ops: Sequence[str] = DEFAULT_BINARY
    unary_ops: Sequence[str] = DEFAULT_UNARY

    def validate(self) -> None:
        if self.pop_size < 4:
            raise ValueError("pop_size must be at least 4")
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if not (0 <= self.crossover_rate <= 1 and 0 <= self.mutation_rate <= 1):
            raise ValueError("crossover_rate and mutation_rate must be in [0, 1]")
        if self.crossover_rate + self.mutation_rate > 1:
            raise ValueError("crossover_rate + mutation_rate must not exceed 1")
        if not (0 <= self.elite_count < self.pop_size):
            raise ValueError("elite_count must be in [0, pop_size)")
        if self.max_depth < 2 or self.init_depth < 2:
            raise ValueError("depths must be at least 2")
        if self.parsimony < 0:
            raise ValueError("parsimony must be >= 0")
        if not set(self.binary_ops) <= BINARY_ALLOWED:
            raise ValueError(f"Unsupported binary operators: {set(self.binary_ops) - BINARY_ALLOWED}")
        if not set(self.unary_ops) <= UNARY_ALLOWED:
            raise ValueError(f"Unsupported unary operators: {set(self.unary_ops) - UNARY_ALLOWED}")
        if not self.binary_ops:
            raise ValueError("At least one binary operator is required")


@dataclass
class EvolutionResult:
    best: Individual
    history: List[Dict] = field(default_factory=list)
    var_names: Optional[Sequence[str]] = None
    generations_run: int = 0
    stopped_early: bool = False
    hall_of_fame: List[Individual] = field(default_factory=list)   # best distinct candidates seen, best first

    @property
    def expression(self) -> str:
        return self.best.tree.to_string(self.var_names)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return evaluate(self.best.tree, X)


def _validate_data(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional (samples x variables)")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]} values")
    if X.shape[0] < 10:
        raise ValueError("At least 10 training observations are required")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise ValueError("Data contains NaN or infinite values; clean it before searching")
    if np.std(y) == 0:
        raise ValueError("The target variable is constant; there is nothing to discover")
    return X, y


def run_evolution(
    X,
    y,
    config: Optional[EvolutionConfig] = None,
    var_names: Optional[Sequence[str]] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> EvolutionResult:
    """Search for an expression f(X) ~ y. Returns the best candidate and fitness history."""
    cfg = config or EvolutionConfig()
    cfg.validate()
    X, y = _validate_data(X, y)
    n_vars = X.shape[1]
    if var_names is not None and len(var_names) != n_vars:
        raise ValueError("var_names length must match the number of input columns")

    rng = np.random.default_rng(cfg.seed)

    def ev(tree: Node) -> Individual:
        return evaluate_individual(tree, X, y, cfg.parsimony, cfg.tune_constants)

    trees = initial_population(rng, cfg.pop_size, n_vars, cfg.init_depth, cfg.binary_ops, cfg.unary_ops)
    population = [ev(t) for t in trees]

    best = min(population, key=lambda i: i.fitness)
    best = Individual(copy_tree(best.tree), best.loss, best.fitness, best.size)
    result = EvolutionResult(best=best, var_names=var_names)
    stale = 0
    archive: Dict[str, Individual] = {}     # hall of fame, keyed by expression text

    def archive_top(pop: List[Individual], n: int = 30) -> None:
        for ind in pop[:n]:
            if ind.fitness >= BAD_FITNESS:
                continue
            key = ind.tree.to_string()
            if key not in archive or ind.fitness < archive[key].fitness:
                archive[key] = Individual(copy_tree(ind.tree), ind.loss, ind.fitness, ind.size)
        if len(archive) > 300:
            keep = sorted(archive.items(), key=lambda kv: kv[1].fitness)[:150]
            archive.clear()
            archive.update(keep)

    for gen in range(cfg.generations + 1):
        population.sort(key=lambda i: i.fitness)
        archive_top(population)
        gen_best = population[0]
        improved = gen_best.fitness < best.fitness - cfg.min_improvement
        if gen_best.fitness < best.fitness:
            best = Individual(copy_tree(gen_best.tree), gen_best.loss, gen_best.fitness, gen_best.size)
        stale = 0 if improved or gen == 0 else stale + 1

        finite = [i.fitness for i in population if i.fitness < 1e8]
        record = {
            "generation": gen,
            "best_fitness": best.fitness,
            "best_loss": best.loss,
            "best_size": best.size,
            "mean_fitness": float(np.mean(finite)) if finite else float("nan"),
            "mean_size": float(np.mean([i.size for i in population])),
            "best_expression": best.tree.to_string(var_names),
        }
        result.history.append(record)
        result.best = best
        result.generations_run = gen
        if progress_callback is not None:
            progress_callback(record)

        if gen == cfg.generations:
            break
        if stale >= cfg.patience:
            result.stopped_early = True
            break

        # --- next generation -------------------------------------------------
        next_pop: List[Individual] = [population[i] for i in range(cfg.elite_count)]  # elitism
        while len(next_pop) < cfg.pop_size:
            r = rng.random()
            if r < cfg.crossover_rate:
                p1 = tournament_select(rng, population, cfg.tournament_size)
                p2 = tournament_select(rng, population, cfg.tournament_size)
                child = subtree_crossover(rng, p1.tree, p2.tree, cfg.max_depth)
            elif r < cfg.crossover_rate + cfg.mutation_rate:
                p = tournament_select(rng, population, cfg.tournament_size)
                child = mutate(rng, p.tree, n_vars, cfg.max_depth, cfg.binary_ops, cfg.unary_ops)
            else:  # reproduction: reuse the already-evaluated parent (no re-tuning)
                p = tournament_select(rng, population, cfg.tournament_size)
                next_pop.append(Individual(copy_tree(p.tree), p.loss, p.fitness, p.size))
                continue
            next_pop.append(ev(child))
        population = next_pop

    # Keep variety: one entry per (complexity, ~1% error) bucket, so the list is not 60 copies of the winner.
    buckets: Dict[tuple, Individual] = {}
    for ind in sorted(archive.values(), key=lambda i: i.fitness):
        buckets.setdefault((ind.size, round(ind.loss, 2)), ind)
    result.hall_of_fame = list(buckets.values())[:80]
    return result
