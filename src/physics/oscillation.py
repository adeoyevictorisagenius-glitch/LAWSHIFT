"""System B: simple harmonic motion."""
from __future__ import annotations

import numpy as np

from ..data.generators import SystemSpec


def displacement(t: np.ndarray, amplitude: float, omega: float, phase: float = 0.0) -> np.ndarray:
    """x(t) = A cos(omega t + phase)."""
    if amplitude <= 0 or omega <= 0:
        raise ValueError("amplitude and omega must be positive")
    return amplitude * np.cos(omega * np.asarray(t, dtype=float) + phase)


def harmonic_spec(amplitude: float = 2.0, omega: float = 3.0, phase: float = 0.0) -> SystemSpec:
    eq = f"x = {amplitude:g} * cos({omega:g} * t)" if phase == 0 else \
        f"x = {amplitude:g} * cos({omega:g} * t + {phase:g})"
    tex = rf"x = {amplitude:g}\,\cos\left({omega:g}\,t\right)" if phase == 0 else \
        rf"x = {amplitude:g}\,\cos\left({omega:g}\,t + {phase:g}\right)"
    return SystemSpec(
        key="harmonic",
        title="Simple harmonic motion",
        description=(
            "A mass on an ideal spring oscillates about its equilibrium position. "
            "The experiment records displacement x at different times t."
        ),
        input_names=("t",),
        output_name="x",
        input_ranges=((0.0, 6.0),),
        units={"t": "s", "x": "m"},
        true_fn=lambda X: displacement(X[:, 0], amplitude, omega, phase),
        true_equation=eq,
        true_latex=tex,
        assumptions=(
            "Ideal linear spring (Hooke's law); no damping or driving force.",
            "Hidden amplitude and angular frequency are fixed within one experiment.",
            "Observation window covers about three periods for the default parameters.",
        ),
    )
