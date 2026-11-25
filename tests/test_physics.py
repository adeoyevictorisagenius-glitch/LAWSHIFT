import numpy as np
import pytest

from src.data.generators import available_systems, generate_observations, get_system
from src.data.preprocessing import drop_invalid_rows, train_validation_split
from src.physics.motion import acceleration
from src.physics.oscillation import displacement
from src.physics.projectile import vertical_displacement


def test_newton_values():
    assert acceleration(np.array([10.0]), np.array([2.0]))[0] == pytest.approx(5.0)
    with pytest.raises(ValueError):
        acceleration(np.array([1.0]), np.array([0.0]))


def test_harmonic_values_and_periodicity():
    t = np.array([0.0, np.pi / 3.0 / 2.0])          # omega=3 -> quarter period at pi/6
    x = displacement(t, 2.0, 3.0)
    assert x[0] == pytest.approx(2.0)
    assert x[1] == pytest.approx(0.0, abs=1e-12)
    period = 2 * np.pi / 3.0
    np.testing.assert_allclose(displacement(np.array([0.7]), 2, 3), displacement(np.array([0.7 + period]), 2, 3))
    with pytest.raises(ValueError):
        displacement(t, -1.0, 3.0)


def test_projectile_values():
    # v0=20 m/s, t=2 s: 40 - 0.5*9.81*4 = 20.38
    assert vertical_displacement(np.array([20.0]), np.array([2.0]))[0] == pytest.approx(20.38)
    # apex time v0/g: y = v0^2 / (2g)
    v0 = 15.0
    assert vertical_displacement(np.array([v0]), np.array([v0 / 9.81]))[0] == pytest.approx(v0 ** 2 / (2 * 9.81))
    with pytest.raises(ValueError):
        vertical_displacement(np.array([1.0]), np.array([-1.0]))


def test_three_systems_registered_with_assumptions():
    systems = available_systems()
    assert set(systems) == {"newtonian", "harmonic", "projectile"}
    for s in systems.values():
        assert s.assumptions and s.true_equation and len(s.input_ranges) == len(s.input_names)
    with pytest.raises(ValueError):
        get_system("nope")


@pytest.mark.parametrize("key", ["newtonian", "harmonic", "projectile"])
def test_generation_reproducible_and_in_range(key):
    spec = get_system(key)
    a = generate_observations(spec, 100, 0.05, seed=3)
    b = generate_observations(spec, 100, 0.05, seed=3)
    c = generate_observations(spec, 100, 0.05, seed=4)
    np.testing.assert_array_equal(a.y_obs, b.y_obs)
    assert not np.array_equal(a.y_obs, c.y_obs)
    for j, (lo, hi) in enumerate(spec.input_ranges):
        assert a.X[:, j].min() >= lo and a.X[:, j].max() <= hi
    np.testing.assert_allclose(a.y_clean, spec.true_fn(a.X))


def test_noise_level_is_respected():
    spec = get_system("projectile")
    zero = generate_observations(spec, 500, 0.0, seed=1)
    np.testing.assert_array_equal(zero.y_obs, zero.y_clean)
    noisy = generate_observations(spec, 5000, 0.10, seed=1)
    ratio = np.std(noisy.y_obs - noisy.y_clean) / np.std(noisy.y_clean)
    assert ratio == pytest.approx(0.10, rel=0.1)


@pytest.mark.parametrize("n,noise", [(5, 0.1), (100, -0.1), (100, 2.0), (100, float("nan"))])
def test_generation_rejects_bad_args(n, noise):
    with pytest.raises(ValueError):
        generate_observations(get_system("newtonian"), n, noise, 0)


def test_split_disjoint_complete_reproducible():
    X = np.arange(200, dtype=float).reshape(100, 2)
    y = np.arange(100, dtype=float)
    X_tr, y_tr, X_va, y_va, i_tr, i_va = train_validation_split(X, y, 0.3, seed=9)
    assert len(i_va) == 30 and len(i_tr) == 70
    assert set(i_tr).isdisjoint(i_va) and set(i_tr) | set(i_va) == set(range(100))
    np.testing.assert_array_equal(y_tr, y[i_tr])
    again = train_validation_split(X, y, 0.3, seed=9)
    np.testing.assert_array_equal(i_va, again[5])
    with pytest.raises(ValueError):
        train_validation_split(X, y, 0.9)
    with pytest.raises(ValueError):
        train_validation_split(X, y[:-1])


def test_drop_invalid_rows():
    X = np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0, np.inf])
    Xc, yc, dropped = drop_invalid_rows(X, y)
    assert dropped == 2 and Xc.shape == (1, 2) and yc[0] == 1.0
