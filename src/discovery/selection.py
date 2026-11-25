"""Tournament selection (lower fitness wins)."""
from __future__ import annotations

from typing import List

import numpy as np

from .fitness import Individual


def tournament_select(rng: np.random.Generator, population: List[Individual], k: int = 5) -> Individual:
    if not population:
        raise ValueError("Cannot select from an empty population")
    k = max(1, min(k, len(population)))
    idx = rng.choice(len(population), size=k, replace=False)
    return min((population[i] for i in idx), key=lambda ind: ind.fitness)
