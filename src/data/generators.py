"""System specifications and observation generation.

A SystemSpec describes a simulated physical system: its input variables and ranges,
the hidden law used to generate data, and its documented assumptions. The hidden law
(`true_fn`, `true_equation`) is used ONLY to generate observations and for the
post-experiment comparison. It is never passed to the discovery engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SystemSpec:
    key: str
    title: str
    description: str
    input_names: Tuple[str, ...]
    output_name: str
    input_ranges: Tuple[Tuple[float, float], ...]
    units: Dict[str, str]
    true_fn: Callable[[np.ndarray], np.ndarray]
    true_equation: str
    assumptions: Tuple[str, ...] = field(default_factory=tuple)
    true_latex: str = ""       # the hidden law as LaTeX (display only)


@dataclass
class Observations:
    spec_key: str
    input_names: Tuple[str, ...]
    output_name: str
    X: np.ndarray            # (n, n_inputs)
    y_clean: np.ndarray      # noise-free values from the hidden law
    y_obs: np.ndarray        # what the experimenter "measures" (clean + noise)
    noise_level: float
    seed: int


def validate_generation_args(n_samples: int, noise_level: float) -> None:
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 30:
        raise ValueError("n_samples must be an integer of at least 30")
    if n_samples > 20000:
        raise ValueError("n_samples must not exceed 20000")
    if not np.isfinite(noise_level) or not (0.0 <= noise_level <= 1.0):
        raise ValueError("noise_level must be between 0 and 1 (fraction of the signal's std)")


def generate_observations(spec: SystemSpec, n_samples: int, noise_level: float, seed: int) -> Observations:
    """Sample inputs uniformly, evaluate the hidden law, add Gaussian measurement noise.

    noise_level is the noise standard deviation as a fraction of std(y_clean),
    e.g. 0.02 means roughly 2% measurement noise.
    """
    validate_generation_args(n_samples, noise_level)
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(lo, hi, n_samples) for lo, hi in spec.input_ranges])
    y_clean = np.asarray(spec.true_fn(X), dtype=float)
    if not np.all(np.isfinite(y_clean)):
        raise ValueError(f"System {spec.key!r} produced non-finite values")
    noise = rng.normal(0.0, noise_level * float(np.std(y_clean)), n_samples) if noise_level > 0 else 0.0
    return Observations(
        spec_key=spec.key,
        input_names=spec.input_names,
        output_name=spec.output_name,
        X=X,
        y_clean=y_clean,
        y_obs=y_clean + noise,
        noise_level=float(noise_level),
        seed=int(seed),
    )


# --------------------------------------------------------------------------- registry
def available_systems() -> Dict[str, SystemSpec]:
    # imported here to avoid circular imports
    from ..physics.motion import newtonian_spec
    from ..physics.oscillation import harmonic_spec
    from ..physics.projectile import projectile_spec

    specs: List[SystemSpec] = [newtonian_spec(), harmonic_spec(), projectile_spec()]
    return {s.key: s for s in specs}


def get_system(key: str) -> SystemSpec:
    systems = available_systems()
    if key not in systems:
        raise ValueError(f"Unknown system {key!r}. Available: {sorted(systems)}")
    return systems[key]
