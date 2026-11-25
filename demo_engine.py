"""Stage 2 demo: rediscover F = m * a from noisy data, then validate on unseen data.

Run from the project root:  python demo_engine.py
The engine only receives (m, a) -> F observations. The true law appears below only
to generate the data and for the post-experiment comparison.
"""
import time

import numpy as np

from src.discovery.evolution import EvolutionConfig, run_evolution
from src.utils.metrics import mae, r2, rmse

SEED = 42
rng = np.random.default_rng(SEED)

# 1. Generate observations from a hidden law, with 2% measurement noise
n = 200
m = rng.uniform(1, 10, n)
a = rng.uniform(1, 10, n)
F_true = m * a
F_obs = F_true + rng.normal(0, 0.02 * np.std(F_true), n)
X = np.column_stack([m, a])

# 2. Hold out 30% of the observations; the engine never sees them
split = int(0.7 * n)
X_tr, y_tr, X_va, y_va = X[:split], F_obs[:split], X[split:], F_obs[split:]

# 3. Search
cfg = EvolutionConfig(pop_size=150, generations=40, seed=SEED)
t0 = time.time()
res = run_evolution(
    X_tr, y_tr, cfg, var_names=["m", "a"],
    progress_callback=lambda r: print(
        f"gen {r['generation']:3d}  loss={r['best_loss']:.5f}  size={r['best_size']:2d}  {r['best_expression']}"
    ) if r["generation"] % 5 == 0 else None,
)

# 4. Report
pred_va = res.predict(X_va)
print("\n--- RESULT ---")
print("Discovered :", res.expression)
print(f"Complexity : {res.best.size} nodes")
print(f"Train R2   : {r2(y_tr, res.predict(X_tr)):.5f}")
print(f"Held-out R2: {r2(y_va, pred_va):.5f}   RMSE={rmse(y_va, pred_va):.4f}   MAE={mae(y_va, pred_va):.4f}")
print(f"Generations: {res.generations_run}  (stopped early: {res.stopped_early})   time={time.time() - t0:.1f}s")
print("\nPost-experiment verification (hidden law used to generate the data): F = m * a")
