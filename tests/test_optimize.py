import numpy as np
import pytest

from src.discovery.optimize import Goal, optimize, optimize_candidates

NAMES = ["a", "b", "c"]


def f(Z):                       # a: increasing with diminishing returns (max at 10); b: sweet spot at 7.25; c: irrelevant
    return 5 * np.sqrt(Z[:, 0]) + 0.5 * Z[:, 1] - 0.2 * (Z[:, 1] - 6) ** 2


def data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(0, 10, n), rng.uniform(0, 10, n), rng.uniform(20, 40, n)])
    return X, f(X)


def test_maximize_finds_the_true_optimum():
    X, y = data()
    r = optimize(f, X, NAMES, Goal("maximize"), "y", y_ref=y, rmse=0.2)
    assert r.values["b"] == pytest.approx(7.25, abs=0.05)
    assert r.values["a"] > 9.8 and r.edge.get("a") == "upper edge"
    assert r.predicted == pytest.approx(f(np.array([[10.0, 7.25, 30.0]]))[0], abs=0.2)
    assert r.improvement > 0 and r.reached and "c" in r.inert


def test_minimize_goes_to_the_low_end():
    X, y = data()
    r = optimize(f, X, NAMES, Goal("minimize"), "y")
    assert r.values["a"] < 0.3 and r.values["b"] < 0.3 and r.improvement < 0


def test_target_and_range_are_met():
    X, y = data()
    t = optimize(f, X, NAMES, Goal("target", target=10.0), "y")
    assert abs(t.predicted - 10.0) < 0.05 and t.reached
    rg = optimize(f, X, NAMES, Goal("range", low=11, high=12), "y")
    assert 10.95 <= rg.predicted <= 12.05 and rg.reached


def test_unreachable_goal_is_reported_honestly():
    X, y = data()
    r = optimize(f, X, NAMES, Goal("target", target=500.0), "y")
    assert not r.reached and r.gap > 400


def test_range_prefers_the_smallest_change_from_typical():
    X, y = data()
    typical = float(f(np.median(X, axis=0).reshape(1, -1))[0])
    r = optimize(f, X, NAMES, Goal("range", low=typical - 1, high=typical + 1), "y")
    assert r.reached
    assert abs(r.values["c"] - np.median(X[:, 2])) < 2.0           # an irrelevant variable stays near typical


def test_locked_variables_are_respected():
    X, y = data()
    r = optimize(f, X, NAMES, Goal("maximize"), "y", locked={"a": 3.0})
    assert r.values["a"] == 3.0 and "a" in r.locked and "a" not in r.free
    assert r.values["b"] == pytest.approx(7.25, abs=0.05)


def test_bounds_are_respected_and_never_exceed_the_data():
    X, y = data()
    r = optimize(f, X, NAMES, Goal("maximize"), "y", bounds={"b": (0.0, 5.0), "a": (-100.0, 100.0)})
    assert r.values["b"] <= 5.0 + 1e-9 and r.edge.get("b") == "upper edge"
    assert r.bounds["a"] == (float(X[:, 0].min()), float(X[:, 0].max()))
    assert any("cut back" in n for n in r.notes)
    for j, nm in enumerate(NAMES):
        assert X[:, j].min() - 1e-9 <= r.values[nm] <= X[:, j].max() + 1e-9


def test_everything_locked_is_handled():
    X, y = data()
    r = optimize(f, X, NAMES, Goal("maximize"), "y", locked={"a": 1.0, "b": 2.0, "c": 30.0})
    assert not r.free and r.notes and r.values == {"a": 1.0, "b": 2.0, "c": 30.0}


def test_reality_check_and_far_from_data_flag():
    X, y = data()
    r = optimize(f, X, NAMES, Goal("maximize"), "y", y_ref=y)
    assert r.best_observed == pytest.approx(y.max()) and r.nearest_value is not None
    assert not r.far_from_data
    # data lie on a thin diagonal strip: the optimum of a corner-seeking goal is far from any real row
    rng = np.random.default_rng(1)
    u = rng.uniform(0, 10, 300)
    Xs = np.column_stack([u, u + rng.normal(0, 0.2, 300)])
    far = optimize(lambda Z: Z[:, 0] - Z[:, 1], Xs, ["p", "q"], Goal("maximize"), "d", y_ref=Xs[:, 0] - Xs[:, 1])
    assert far.far_from_data


def test_touchy_variable_is_flagged():
    X, y = data()
    r = optimize(lambda Z: 100 * Z[:, 0] + 0.01 * Z[:, 1], X[:, :2], ["a", "b"], Goal("target", target=500.0), "y")
    assert "a" in r.touchy and "b" not in r.touchy


def test_reproducible():
    X, y = data()
    a = optimize(f, X, NAMES, Goal("maximize"), "y", seed=3)
    b = optimize(f, X, NAMES, Goal("maximize"), "y", seed=3)
    assert a.values == b.values


@pytest.mark.parametrize("kw", [
    dict(goal=Goal("sideways")), dict(goal=Goal("target")), dict(goal=Goal("range", low=5, high=1)),
    dict(goal=Goal("maximize"), locked={"zzz": 1.0}), dict(goal=Goal("maximize"), bounds={"zzz": (0, 1)}),
    dict(goal=Goal("maximize"), bounds={"a": (6, 4)}), dict(goal=Goal("maximize"), bounds={"a": (50, 60)}),
])
def test_bad_inputs_are_rejected(kw):
    X, y = data()
    with pytest.raises(ValueError):
        optimize(f, X, NAMES, kw.pop("goal"), "y", **kw)


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        optimize(f, np.ones((3, 3)), NAMES, Goal("maximize"))
    with pytest.raises(ValueError):
        optimize(f, np.ones((10, 2)), NAMES, Goal("maximize"))


class _Cand:
    def __init__(self, fn, rmse):
        self.predict, self.validation = fn, {"rmse": rmse}


def test_candidates_agree_and_disagree():
    X, y = data()
    same = optimize_candidates([_Cand(f, 0.1), _Cand(lambda Z: f(Z) + 0.3, 0.2)], X, NAMES, Goal("maximize"), "y")
    assert same.agree and same.disagreement < 0.15
    other = _Cand(lambda Z: 5 * np.sqrt(Z[:, 0]) + 0.5 * Z[:, 1] - 0.2 * (Z[:, 1] - 2) ** 2, 0.2)   # sweet spot elsewhere
    diff = optimize_candidates([_Cand(f, 0.1), other], X, NAMES, Goal("maximize"), "y")
    assert not diff.agree and diff.disagreement > 0.3
    with pytest.raises(ValueError):
        optimize_candidates([], X, NAMES, Goal("maximize"))
