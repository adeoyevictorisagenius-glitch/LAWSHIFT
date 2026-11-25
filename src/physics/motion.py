"""System A: Newtonian motion (Newton's second law)."""
from __future__ import annotations

import numpy as np

from ..data.generators import SystemSpec


def acceleration(force: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """a = F / m for a single body with constant mass."""
    mass = np.asarray(mass, dtype=float)
    if np.any(mass <= 0):
        raise ValueError("Mass must be positive")
    return np.asarray(force, dtype=float) / mass


def newtonian_spec() -> SystemSpec:
    return SystemSpec(
        key="newtonian",
        title="Newtonian motion",
        description=(
            "A body of mass m is pushed by a net force F. The experiment records the "
            "resulting acceleration a."
        ),
        input_names=("F", "m"),
        output_name="a",
        input_ranges=((1.0, 100.0), (1.0, 20.0)),
        units={"F": "N", "m": "kg", "a": "m/s^2"},
        true_fn=lambda X: acceleration(X[:, 0], X[:, 1]),
        true_equation="a = F / m",
        true_latex=r"a = \frac{F}{m}",
        assumptions=(
            "Single rigid body with constant mass.",
            "F is the net force along one axis (friction folded into F).",
            "Non-relativistic speeds.",
        ),
    )
