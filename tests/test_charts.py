import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.discovery.evolution import EvolutionConfig
from src.discovery.experiment import ExperimentConfig, run_experiment
from src.visualization import charts


@pytest.fixture(scope="module")
def result():
    return run_experiment(ExperimentConfig(
        system_key="harmonic", n_samples=100, seed=2, n_runs=2,
        evolution=EvolutionConfig(pop_size=50, generations=8)))


def _render(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    assert buf.tell() > 3000      # a non-trivial image was drawn


def test_all_physics_charts_render(result):
    r, s = result, result.spec
    pred = r.predict_best(r.X_val)
    _render(charts.raw_observations(s.input_names, r.X_train, r.y_train, s.output_name, r.X_val, r.y_val, s.units))
    _render(charts.actual_vs_predicted(r.y_val, pred))
    _render(charts.residuals(r.y_val, pred))
    _render(charts.fitness_history(r.best.history))
    _render(charts.run_comparison([f"run {i}" for i in range(len(r.runs))],
                                  [x.validation["rmse"] for x in r.runs], [x.size for x in r.runs], r.best_index))
    _render(charts.curve_overlay(r.X_train, r.y_train, r.X_val, r.y_val, r.predict_best, "t", "x", s.true_fn))


def test_chart_data_comes_from_inputs(result):
    """The scatter in actual-vs-predicted must contain exactly the supplied values."""
    pred = result.predict_best(result.X_val)
    fig = charts.actual_vs_predicted(result.y_val, pred)
    pts = fig.axes[0].collections[0].get_offsets()
    np.testing.assert_allclose(pts[:, 0], result.y_val)
    np.testing.assert_allclose(pts[:, 1], pred)
    plt.close(fig)


def test_fitness_chart_rejects_empty_history():
    with pytest.raises(ValueError):
        charts.fitness_history([])


def test_history_scatter_handles_empty_and_real(result):
    _render(charts.history_scatter([]))
    _render(charts.history_scatter([result.to_record()]))


def test_business_charts_render():
    import pandas as pd
    from src.business.analysis import analyze, demo_dataset
    res = analyze(demo_dataset(), "profit", ["advertising", "units_sold"], n_runs=1,
                  evolution=EvolutionConfig(pop_size=40, generations=6))
    _render(charts.correlation_heatmap(res.corr_matrix))
    _render(charts.anomaly_plot(res.y_all, res.pred_linear_all, res.anomaly_mask, "profit"))
    m = res.models
    _render(charts.model_comparison(list(m["model"]), list(m["val_rmse"]), list(m["val_r2"])))


def test_insight_and_candidate_charts_render():
    import pandas as pd
    from src.data.preprocessing import prepare_numeric_dataframe
    from src.discovery.dataset_experiment import DatasetConfig, run_dataset_experiment
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.uniform(1, 9, 120), "b": rng.uniform(1, 9, 120)})
    df["y"] = df["a"] * 2 + rng.normal(0, 0.1, 120)
    res = run_dataset_experiment(df, DatasetConfig(target="y", features=["a", "b"], n_runs=1,
                                                   evolution=EvolutionConfig(pop_size=40, generations=6)))
    ins = res.best.insights
    _render(charts.variable_importance([v.name for v in ins.variables], [v.importance for v in ins.variables]))
    _render(charts.partial_effects(ins.variables, "y"))
    _render(charts.candidates_comparison([f"#{c.rank}" for c in res.candidates],
                                         [c.validation["r2"] for c in res.candidates], [c.size for c in res.candidates]))
