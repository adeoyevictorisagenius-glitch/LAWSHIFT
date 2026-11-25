import numpy as np
import pytest

import src.discovery.experiment as experiment
from src.discovery.evolution import EvolutionConfig
from src.discovery.experiment import ExperimentConfig, run_experiment
from src.utils.history import HistoryStore, summary_stats


def _cfg(system="newtonian", **kw):
    base = dict(system_key=system, n_samples=120, noise_level=0.02, seed=1, n_runs=2,
                evolution=EvolutionConfig(pop_size=60, generations=12))
    base.update(kw)
    return ExperimentConfig(**base)


def test_engine_never_receives_the_hidden_law(monkeypatch):
    """The only thing handed to the search is training X, y, config and variable names."""
    seen = []
    real = experiment.run_evolution

    def spy(X, y, cfg=None, var_names=None, progress_callback=None):
        seen.append((X, y, var_names))
        return real(X, y, cfg, var_names, progress_callback)

    monkeypatch.setattr(experiment, "run_evolution", spy)
    res = run_experiment(_cfg())
    assert seen
    for X, y, names in seen:
        assert X.shape == res.X_train.shape and y.shape == res.y_train.shape
        assert names == ["F", "m"]
    # the validation rows were never given to the search
    assert res.X_val.shape[0] == 36 and res.X_train.shape[0] == 84


def test_full_cycle_newtonian_generalises():
    res = run_experiment(_cfg())
    assert res.best.validation["r2"] > 0.97
    assert res.best.validation["r2"] <= 1.0
    assert res.noise_ceiling_r2 > 0.97
    assert res.best.nrmse_vs_true_law < 0.1
    assert len(res.runs) == 2 and res.interpretation
    assert any("simulated" in line for line in res.interpretation)
    assert res.spec.true_equation == "a = F / m"      # available only for post-hoc verification


def test_best_run_selected_by_training_fitness():
    res = run_experiment(_cfg())
    assert res.best.fitness == min(r.fitness for r in res.runs)


def test_experiment_reproducible():
    a = run_experiment(_cfg(n_runs=1))
    b = run_experiment(_cfg(n_runs=1))
    assert a.best.expression == b.best.expression
    assert a.best.validation["r2"] == b.best.validation["r2"]


def test_parallel_matches_sequential():
    seq = run_experiment(_cfg(n_jobs=1))
    par = run_experiment(_cfg(n_jobs=2))
    assert [r.expression for r in seq.runs] == [r.expression for r in par.runs]


def test_progress_callback_reports_runs():
    events = []
    run_experiment(_cfg(n_runs=2), progress_callback=events.append)
    assert {e["run"] for e in events} == {0, 1}
    assert any(e.get("done") for e in events) and any("generation" in e for e in events)


@pytest.mark.parametrize("kw", [dict(n_runs=0), dict(n_jobs=0), dict(noise_level=3.0), dict(n_samples=10),
                                dict(val_fraction=0.9)])
def test_invalid_experiment_config_rejected(kw):
    with pytest.raises(ValueError):
        run_experiment(_cfg(**kw))


def test_unknown_system_rejected():
    with pytest.raises(ValueError):
        run_experiment(_cfg(system="warp_drive"))


def test_history_roundtrip_and_stats(tmp_path):
    store = HistoryStore(tmp_path / "h.jsonl")
    assert store.load() == [] and summary_stats([])["n_experiments"] == 0
    res = run_experiment(_cfg(n_runs=1))
    rec = store.append(res.to_record())
    assert rec["id"] and rec["timestamp"]
    loaded = store.load()
    assert len(loaded) == 1 and loaded[0]["expression"] == res.best.expression
    assert loaded[0]["fitness_history"][0]["generation"] == 0
    stats = summary_stats(loaded)
    assert stats["n_experiments"] == 1 and stats["by_system"] == {"newtonian": 1}
    assert stats["best_val_r2"] == pytest.approx(res.best.validation["r2"])
    # a corrupted line is skipped, not fatal
    with open(tmp_path / "h.jsonl", "a") as fh:
        fh.write("{not json\n")
    assert len(store.load()) == 1
