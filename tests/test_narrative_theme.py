import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import theme
from src.data.normality import profile_column, profile_table, strongest_pairs
from src.discovery.dataset_experiment import DatasetConfig, run_dataset_experiment
from src.discovery.evolution import EvolutionConfig
from src.discovery.narrative import (ReadoutContext, readout_candidate, readout_column, readout_optimization,
                                     readout_run, readout_table)
from src.discovery.optimize import Goal, optimize_candidates
from src.visualization import charts


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"Alpha": rng.uniform(1, 10, n), "Beta": rng.uniform(0, 10, n), "Gamma": rng.uniform(0, 5, n)})
    df["Out"] = 10 + 4 * np.sqrt(df["Alpha"]) - 0.5 * (df["Beta"] - 5) ** 2 + rng.normal(0, 0.3, n)
    return run_dataset_experiment(df, DatasetConfig(target="Out", features=["Alpha", "Beta", "Gamma"], n_runs=2,
                                                    evolution=EvolutionConfig(pop_size=80, generations=25)))


# ------------------------------------------------------------------ colour system
def _lum(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _contrast(a, b):
    hi, lo = sorted([_lum(a), _lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_text_colours_are_readable():
    for fg, bg in [(theme.INK, theme.CREAM), (theme.INK, theme.PAPER), (theme.MUTED, theme.CREAM), (theme.MUTED, theme.PAPER),
                   (theme.SAGE_DEEP, theme.SAGE_SOFT), (theme.SAGE_TEXT, theme.CREAM), ("#FFFFFF", theme.TERRACOTTA),
                   (theme.INK, theme.TERRACOTTA_SOFT)]:
        assert _contrast(fg, bg) >= 4.5, (fg, bg)


def test_palette_is_soft_not_neon():
    import colorsys

    def hsv(hexcode):
        h = hexcode.lstrip("#")
        return colorsys.rgb_to_hsv(*(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)))

    for name in ("CREAM", "PAPER", "LINEN", "SAGE", "SAGE_DEEP", "SAGE_MID", "SAGE_SOFT", "TAUPE", "TERRACOTTA_SOFT"):
        assert hsv(getattr(theme, name))[1] < 0.5, name        # neutrals and sages are muted
    _, s, v = hsv(theme.TERRACOTTA)                           # the accent is earthy: allowed to be warm, never bright
    assert s < 0.85 and v < 0.8
    assert hsv("#00F0FF")[2] > 0.95 and hsv("#00F0FF")[1] > 0.95   # (what the old neon cyan looked like, for contrast)


def test_css_is_complete_and_not_sci_fi():
    css = theme.build_css()
    assert not re.search(r"__[A-Z_]+__", css)
    for banned in ("Orbitron", "Share Tech Mono", "text-shadow", "scanline", "#00F0FF", "#FF3DDC", "@keyframes"):
        assert banned not in css
    assert theme.TERRACOTTA in css and theme.SAGE in css and theme.CREAM in css


def test_accent_is_used_sparingly_in_the_stylesheet():
    css = theme.build_css()
    assert css.count("var(--accent)") <= 12            # the 10%: buttons, rank-1 badge, a few highlights


def test_charts_use_the_theme_palette():
    assert charts.ACCENT == theme.TERRACOTTA and charts.SAGE == theme.SAGE and charts.BG == theme.CREAM
    forbidden = {"#00F0FF", "#FF3DDC", "#9B6BFF", "#060B1F"}
    assert not forbidden & {getattr(charts, n).upper() for n in dir(charts) if n.isupper() and isinstance(getattr(charts, n), str)}


# ------------------------------------------------------------------ candidates carry LaTeX
def test_candidates_have_latex_and_insight_extras(result):
    for c in result.candidates:
        assert c.latex.startswith(r"\mathrm{Out} =") and c.latex.count("{") == c.latex.count("}")
        for v in c.insights.variables:
            assert np.isfinite(v.swing) and v.swing_share >= 0
    rows = result.best.insights.table()
    assert {r["variable"] for r in rows} == {"Alpha", "Beta", "Gamma"} and "elasticity (+10%) [%]" in rows[0]
    assert result.best.insights.rmse > 0 and result.best.insights.mean_abs_target > 0


# ------------------------------------------------------------------ read-outs
def test_candidate_readout_is_plain_checkable_and_deterministic(result):
    c = result.best
    ctx = ReadoutContext(n_rows=result.n_rows, n_runs=result.n_runs, runs_agreeing=result.runs_agreeing, linear_val_r2=0.5)
    a = readout_candidate(c, "Out", result.y_val, ctx)
    b = readout_candidate(c, "Out", result.y_val, ctx)
    assert a.paragraphs == b.paragraphs and a.next_steps == b.next_steps
    text = "\n".join(a.paragraphs)
    assert f"{c.validation['r2']:.3f}" in text                      # quotes the real held-out R²
    assert "`" in text and "How far I would trust it" in text and "mistakes" in text
    assert "$" not in text                                          # would be read as LaTeX by Markdown
    assert a.next_steps and "Optimise" in a.next_steps[0] and a.cautions


def test_readout_reports_unused_variables_and_weak_fits(result):
    c = result.best
    unused = [v.name for v in c.insights.variables if not v.used]
    text = "\n".join(readout_candidate(c, "Out", result.y_val).paragraphs)
    for u in unused:
        assert f"`{u}`" in text
    import copy
    weak = copy.copy(c)
    weak.validation = {**c.validation, "r2": 0.2}
    r = readout_candidate(weak, "Out", result.y_val)
    assert any("weak" in x for x in r.cautions) and "poorly" in r.paragraphs[0]


def test_run_readout_compares_alternatives(result):
    r = readout_run(result.candidates, "Out")
    assert r.paragraphs and (f"{len(result.candidates)} genuinely different" in r.paragraphs[0] or len(result.candidates) == 1)


def test_column_and_table_readouts():
    rng = np.random.default_rng(1)
    skewed = profile_column(rng.lognormal(3, 0.8, 300), "Screen_time")
    text = readout_column(skewed)
    assert "`Screen_time`" in text and "skewed to the right" in text and "log transform" in text and "68%" in text
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.lognormal(size=200)})
    df["c"] = 2 * df["a"] + rng.normal(0, 0.1, 200)
    profiles, _ = profile_table(df)
    r = readout_table(profiles, strongest_pairs(df.corr(), 2), len(df))
    assert "3 numeric columns" in r.paragraphs[0] and "`b`" in r.paragraphs[0] and "not that one causes" in r.paragraphs[1]


def test_optimisation_readout(result):
    X = np.vstack([result.X_train, result.X_val])
    y = np.concatenate([result.y_train, result.y_val])
    cmp_ = optimize_candidates(result.candidates, X, ["Alpha", "Beta", "Gamma"], Goal("maximize"), "Out", y_ref=y)
    res = cmp_.results[0]
    r = readout_optimization(res, cmp_)
    text = "\n".join(r.paragraphs)
    assert "as much `Out` as possible" in text and "Reality check" in text and "association, not cause" in " ".join(r.cautions)
    assert f"{res.predicted:.3g}"[:3] in text or "≈" in text
    locked = optimize_candidates(result.candidates, X, ["Alpha", "Beta", "Gamma"], Goal("range", low=y.min(), high=y.max()), "Out",
                                 locked={"Alpha": float(np.median(X[:, 0]))}, y_ref=y)
    assert "held at" in "\n".join(readout_optimization(locked.results[0], locked).paragraphs)


# ------------------------------------------------------------------ new charts
def _render(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    assert buf.tell() > 3000


def test_optimiser_and_normality_charts_render(result):
    X = np.vstack([result.X_train, result.X_val])
    y = np.concatenate([result.y_train, result.y_val])
    names = ["Alpha", "Beta", "Gamma"]
    for goal in (Goal("maximize"), Goal("target", target=float(np.median(y))), Goal("range", low=float(np.percentile(y, 40)), high=float(np.percentile(y, 60)))):
        cmp_ = optimize_candidates(result.candidates, X, names, goal, "Out", y_ref=y)
        _render(charts.optimizer_curves(cmp_.results[0]))
        _render(charts.optimum_agreement(cmp_.per_variable, cmp_.results[0].bounds))
    df = pd.DataFrame(X, columns=names)
    p = profile_column(df["Alpha"].to_numpy(), "Alpha")
    _render(charts.distribution_panel(df["Alpha"], "Alpha", p.mean, p.std))
    _render(charts.distribution_overview(df, {"Alpha": "x"}))
    _render(charts.pair_scatter(df["Alpha"], df["Beta"], "Alpha", "Beta", float(df["Alpha"].corr(df["Beta"]))))
    _render(charts.correlation_heatmap(df.corr()))


def test_optimiser_chart_rejects_nothing_to_plot(result):
    X = np.vstack([result.X_train, result.X_val])
    res = optimize_candidates(result.candidates, X, ["Alpha", "Beta", "Gamma"], Goal("maximize"), "Out",
                              locked={"Alpha": 5.0, "Beta": 5.0, "Gamma": 2.0}).results[0]
    with pytest.raises(ValueError):
        charts.optimizer_curves(res)
