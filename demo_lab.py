"""Stage 3 demo: run all three discovery experiments and save them to the history.

    python demo_lab.py                 # all systems
    python demo_lab.py projectile      # one system
    python demo_lab.py --jobs 4        # use 4 CPU cores for the independent searches
"""
import argparse

from src.discovery.evolution import EvolutionConfig
from src.discovery.experiment import ExperimentConfig, algebraic_form, run_experiment
from src.utils.history import HistoryStore, summary_stats

ap = argparse.ArgumentParser()
ap.add_argument("systems", nargs="*", default=["newtonian", "harmonic", "projectile"])
ap.add_argument("--jobs", type=int, default=2)
ap.add_argument("--noise", type=float, default=0.02)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

store = HistoryStore()  # runs/experiment_history.jsonl
for key in args.systems:
    cfg = ExperimentConfig(
        system_key=key, noise_level=args.noise, seed=args.seed, n_runs=3, n_jobs=args.jobs,
        evolution=EvolutionConfig(pop_size=150, generations=40),
    )
    res = run_experiment(cfg)
    print("=" * 72)
    print(f"{res.spec.title}   (noise {cfg.noise_level:.0%}, {res.X_train.shape[0]} train / {res.X_val.shape[0]} validation)")
    print("-" * 72)
    for i, r in enumerate(res.runs):
        star = "*" if i == res.best_index else " "
        print(f" {star} run {i}: {r.expression:<48} nodes={r.size:<3} heldout R2={r.validation['r2']:.4f}")
    b = res.best
    print(f"\nDiscovered : {b.expression}")
    alg = algebraic_form(res)
    if alg:
        print(f"Expanded   : {alg}")
    print(f"Held-out   : R2={b.validation['r2']:.4f}  RMSE={b.validation['rmse']:.4f}  MAE={b.validation['mae']:.4f}")
    print(f"Noise ceiling (true law on same noisy data): R2={res.noise_ceiling_r2:.4f}")
    print(f"Error vs noise-free true law: {b.nrmse_vs_true_law:.4f} (NRMSE)")
    print(f"\nPOST-EXPERIMENT VERIFICATION - hidden law: {res.spec.true_equation}")
    for line in res.interpretation:
        print("  -", line)
    print(f"  time: {res.elapsed:.1f}s")
    store.append(res.to_record())

print("\nHistory summary:", summary_stats(store.load()))
