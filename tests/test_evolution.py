import numpy as np
import pytest

from src.discovery.crossover import subtree_crossover
from src.discovery.evolution import EvolutionConfig, run_evolution
from src.discovery.expressions import Binary, Const, Var, depth, evaluate, random_tree
from src.discovery.fitness import BAD_FITNESS, evaluate_individual, optimize_constants
from src.discovery.mutation import mutate
from src.discovery.population import initial_population
from src.discovery.selection import tournament_select
from src.utils.metrics import r2


def _newton_data(n=120, noise=0.02, seed=1):
    rng = np.random.default_rng(seed)
    m = rng.uniform(1, 10, n)
    a = rng.uniform(1, 10, n)
    F = m * a
    F = F + rng.normal(0, noise * np.std(F), n)
    return np.column_stack([m, a]), F


def test_constant_optimisation_recovers_known_constant():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 5, (60, 1))
    y = 3.7 * X[:, 0]
    tree = Binary("mul", Const(1.0), Var(0))
    optimize_constants(tree, X, y)
    assert tree.left.value == pytest.approx(3.7, rel=1e-3)


def test_fitness_is_finite_for_pathological_tree():
    X, y = _newton_data()
    bad = Binary("div", Var(0), Binary("sub", Var(1), Var(1)))  # x0 / 0
    ind = evaluate_individual(bad, X, y)
    assert np.isfinite(ind.fitness) and ind.fitness <= BAD_FITNESS


def test_complexity_penalty_prefers_smaller_equal_fit():
    X, y = _newton_data(noise=0.0)
    small = Binary("mul", Var(0), Var(1))
    big = Binary("add", Binary("mul", Var(0), Var(1)), Binary("sub", Var(0), Var(0)))
    f_small = evaluate_individual(small, X, y, parsimony=0.01).fitness
    f_big = evaluate_individual(big, X, y, parsimony=0.01).fitness
    assert f_small < f_big


def test_operators_respect_depth_limit():
    rng = np.random.default_rng(3)
    pop = initial_population(rng, 40, 2, 4)
    for _ in range(300):
        a, b = pop[rng.integers(40)], pop[rng.integers(40)]
        assert depth(subtree_crossover(rng, a, b, max_depth=6)) <= 6
        assert depth(mutate(rng, a, 2, max_depth=6)) <= 6


def test_parents_not_modified_by_variation():
    rng = np.random.default_rng(5)
    a = random_tree(rng, 2, 4)
    b = random_tree(rng, 2, 4)
    sa, sb = a.to_string(), b.to_string()
    for _ in range(50):
        subtree_crossover(rng, a, b)
        mutate(rng, a, 2)
    assert a.to_string() == sa and b.to_string() == sb


def test_tournament_prefers_fitter():
    X, y = _newton_data()
    rng = np.random.default_rng(0)
    good = evaluate_individual(Binary("mul", Var(0), Var(1)), X, y)
    bad = evaluate_individual(Const(0.5), X, y)
    pop = [good] + [bad] * 9
    wins = sum(tournament_select(rng, pop, k=10) is good for _ in range(20))
    assert wins == 20


def test_same_seed_same_result():
    X, y = _newton_data()
    cfg = dict(pop_size=40, generations=6, seed=11)
    r1 = run_evolution(X, y, EvolutionConfig(**cfg))
    r2_ = run_evolution(X, y, EvolutionConfig(**cfg))
    assert r1.expression == r2_.expression
    assert [h["best_fitness"] for h in r1.history] == [h["best_fitness"] for h in r2_.history]


def test_elitism_best_fitness_never_worsens():
    X, y = _newton_data()
    res = run_evolution(X, y, EvolutionConfig(pop_size=60, generations=10, seed=2))
    f = [h["best_fitness"] for h in res.history]
    assert all(f[i + 1] <= f[i] + 1e-12 for i in range(len(f) - 1))


def test_rediscovers_newton_second_law_on_held_out_data():
    """The engine sees only (m, a) -> F. It must generalise to unseen observations."""
    X, y = _newton_data(n=200, noise=0.02, seed=1)
    X_tr, y_tr, X_va, y_va = X[:140], y[:140], X[140:], y[140:]
    res = run_evolution(
        X_tr, y_tr, EvolutionConfig(pop_size=150, generations=40, seed=0), var_names=["m", "a"]
    )
    assert r2(y_va, res.predict(X_va)) > 0.98


@pytest.mark.parametrize(
    "kwargs",
    [dict(pop_size=2), dict(generations=0), dict(crossover_rate=0.8, mutation_rate=0.5),
     dict(binary_ops=("pow",)), dict(elite_count=500)],
)
def test_invalid_config_rejected(kwargs):
    X, y = _newton_data()
    with pytest.raises(ValueError):
        run_evolution(X, y, EvolutionConfig(**kwargs))


def test_invalid_data_rejected():
    X, y = _newton_data()
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        run_evolution(bad, y)
    with pytest.raises(ValueError):
        run_evolution(X, np.ones(len(y)))
    with pytest.raises(ValueError):
        run_evolution(X, y[:-1])
    with pytest.raises(ValueError):
        run_evolution(X[:5], y[:5])
