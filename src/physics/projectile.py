"""System C: projectile motion under a documented simplified model."""
from __future__ import annotations

import numpy as np

from ..data.generators import SystemSpec

G = 9.81  # m/s^2, standard gravity


def vertical_displacement(v0: np.ndarray, t: np.ndarray, g: float = G) -> np.ndarray:
    """y = v0 * t - 0.5 * g * t^2, vertical displacement from the launch point."""
    if g <= 0:
        raise ValueError("g must be positive")
    v0 = np.asarray(v0, dtype=float)
    t = np.asarray(t, dtype=float)
    if np.any(t < 0):
        raise ValueError("Time must be non-negative")
    return v0 * t - 0.5 * g * t ** 2


def projectile_spec(g: float = G) -> SystemSpec:
    return SystemSpec(
        key="projectile",
        title="Projectile motion (vertical launch)",
        description=(
            "An object is launched straight upward with initial speed v0. The experiment "
            "records its vertical displacement y from the launch point at time t."
        ),
        input_names=("v0", "t"),
        output_name="y",
        input_ranges=((10.0, 30.0), (0.1, 2.5)),
        units={"v0": "m/s", "t": "s", "y": "m"},
        true_fn=lambda X: vertical_displacement(X[:, 0], X[:, 1], g),
        true_equation=f"y = v0 * t - 0.5 * {g:g} * t^2",
        true_latex=rf"y = v_{{0}}\,t - \frac{{1}}{{2}}\cdot {g:g}\,t^{{2}}",
        assumptions=(
            "Vertical launch only (one dimension); no horizontal motion.",
            "Constant gravitational acceleration; no air resistance.",
            "Flat-Earth approximation near the surface; y is measured from the launch point "
            "and may be negative (below it) for late times and low v0.",
            "Not a substitute for laboratory measurement or certified ballistics analysis.",
        ),
    )
