"""LAWSHIFT - find the equation hiding in your data (DEMO PROTOTYPE).

Run from the project root:   streamlit run app.py
"""
from __future__ import annotations

import html
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.business.analysis import CAUSATION_NOTICE, analyze, demo_dataset, prepare_dataframe
from src.data.generators import available_systems
from src.data.normality import profile_table, strongest_pairs
from src.data.preprocessing import prepare_numeric_dataframe
from src.discovery.dataset_experiment import MAX_FEATURES, DatasetConfig, run_dataset_experiment
from src.discovery.evolution import EvolutionConfig
from src.discovery.experiment import ExperimentConfig, algebraic_latex, best_latex, run_experiment
from src.discovery.latex import FORMULAS, OBJECTIVES
from src.discovery.narrative import (Readout, ReadoutContext, readout_candidate, readout_column, readout_optimization,
                                     readout_run, readout_table)
from src.discovery.optimize import Goal, optimize_candidates
from src.theme import build_css
from src.utils.history import HistoryStore, summary_stats
from src.visualization import charts

st.set_page_config(page_title="LAWSHIFT", page_icon="🔬", layout="wide")

PAGES = ["Home", "Scientific Discovery Lab", "Everyday Data Lab", "Business Pattern Lab", "Experiment History",
         "About LAWSHIFT"]
PRESETS = {
    "Quick demo": dict(pop_size=80, generations=25, n_runs=2),
    "Standard": dict(pop_size=150, generations=40, n_runs=3),
    "Thorough": dict(pop_size=250, generations=60, n_runs=3),
}
MAX_UPLOAD_MB = 5
SAMPLE_DIR = Path(__file__).parent / "sample_data"
EVERYDAY_SAMPLES = {"coffee_shop", "study_habits"}     # shown in the Everyday Data Lab; the rest feed the physics lab
READOUT_NOTE = ("Written on this machine from the numbers on this page: a rule-based plain-English summary, not an "
                "external AI. Every statement can be checked against the technical tabs.")

st.markdown(build_css(), unsafe_allow_html=True)

store = HistoryStore()
esc = html.escape


# --------------------------------------------------------------------------- helpers
def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<h1 class="page-title">{esc(title)}</h1><div class="page-sub">{esc(subtitle)}</div>',
                unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f'<div class="section-label">{esc(label)}</div>', unsafe_allow_html=True)


def demo_banner(extra: str = "") -> None:
    st.markdown(
        '<div class="demo-banner"><b>Demo prototype.</b> Simulated systems use documented simplified models; sample files '
        'are synthetic; business data is synthetic unless you upload your own. Results are not laboratory measurements '
        f'or certified engineering/financial analysis. {esc(extra)}</div>', unsafe_allow_html=True)


def readout_panel(r: Readout) -> None:
    """The conversational 'what your data says' panel, followed by cautions."""
    with st.chat_message("assistant", avatar="🔬"):
        for p in r.paragraphs:
            st.markdown(p)
        if r.next_steps:
            st.markdown("**What I would do next**")
            for x in r.next_steps:
                st.markdown(f"- {x}")
    st.caption(READOUT_NOTE)
    if r.cautions:
        with st.expander("Things to keep in mind"):
            for c in r.cautions:
                st.markdown(f"- {c}")


def formulas_panel(keys) -> None:
    for k in keys:
        label, tex = FORMULAS[k]
        st.markdown(f"**{label}**")
        st.latex(tex)


def run_with_progress(cfg: ExperimentConfig, label: str = ""):
    bar = st.progress(0.0, text=f"{label}Starting search...")
    total = cfg.n_runs * (cfg.evolution.generations + 1)
    done = {"n": 0}

    def cb(rec):
        if "generation" in rec:
            done["n"] += 1
            bar.progress(min(done["n"] / total, 1.0),
                         text=f"{label}search {rec['run'] + 1}/{rec['n_runs']} - generation "
                              f"{rec['generation']}/{cfg.evolution.generations} - training error {rec['best_loss']:.4f}")

    res = run_experiment(cfg, progress_callback=cb)
    bar.empty()
    return res


def make_progress(n_runs: int, generations: int, label: str = "Searching"):
    """Return (bar, callback) for searches that report {'run','n_runs','generation','best_loss'}."""
    bar = st.progress(0.0, text=f"{label}...")
    total = max(1, n_runs * (generations + 1))
    done = {"n": 0}

    def cb(rec):
        if "generation" in rec:
            done["n"] += 1
            bar.progress(min(done["n"] / total, 1.0),
                         text=f"{label}: search {rec['run'] + 1}/{rec['n_runs']} - generation "
                              f"{rec['generation']}/{generations} - training error {rec['best_loss']:.4f}")

    return bar, cb


def make_config(key, noise, n, valf, seed, preset) -> ExperimentConfig:
    p = PRESETS[preset]
    return ExperimentConfig(
        system_key=key, n_samples=int(n), noise_level=float(noise), val_fraction=float(valf), seed=int(seed),
        n_runs=p["n_runs"], evolution=EvolutionConfig(pop_size=p["pop_size"], generations=p["generations"]))


def ctx_from_dataset(res) -> dict:
    lin = float(res.baselines.set_index("model").loc["Linear regression", "val_r2"])
    return {"X_all": np.vstack([res.X_train, res.X_val]), "y_all": np.concatenate([res.y_train, res.y_val]),
            "features": list(res.features),
            "readout": ReadoutContext(n_rows=res.n_rows, n_runs=res.n_runs, runs_agreeing=res.runs_agreeing,
                                      linear_val_r2=lin)}


def ctx_from_business(res) -> dict:
    lin = float(res.models.set_index("model").loc["Linear regression", "val_r2"])
    return {"X_all": res.X_all, "y_all": res.y_all, "features": list(res.features),
            "readout": ReadoutContext(n_rows=res.n_rows, linear_val_r2=lin)}


# --------------------------------------------------------------------------- shared components
def candidate_card(c) -> None:
    chips = [("Held-out R²", f"{c.validation['r2']:.4f}"), ("RMSE", f"{c.validation['rmse']:.4g}"),
             ("Size", f"{c.size} nodes"), ("Uses", ", ".join(c.used_variables) or "-")]
    chip_html = "".join(f'<span class="chip"><span class="chip-l">{esc(l)}</span><span class="chip-v">{esc(v)}</span></span>'
                        for l, v in chips)
    with st.container(border=True):
        a, b = st.columns([1, 14])
        a.markdown(f'<span class="rank{" r1" if c.rank == 1 else ""}">{c.rank}</span>', unsafe_allow_html=True)
        with b:
            st.latex(c.latex or c.expression)
            st.markdown(f'<div class="cand-chips">{chip_html}</div><div class="cand-note">{esc(c.note)}</div>',
                        unsafe_allow_html=True)


def render_insights(c, target: str) -> None:
    """Technical insights: the precise statements and numbers behind the plain-English read-out."""
    ins = c.insights
    if ins is None:
        st.info("No insights available for this expression.")
        return
    st.markdown(f'<div class="insight-head">{esc(ins.headline)}</div>', unsafe_allow_html=True)
    for v in sorted(ins.variables, key=lambda v: -v.importance):
        cls = "insight warn" if not v.used else "insight"
        st.markdown(f'<div class="{cls}">{esc(v.text)}</div>', unsafe_allow_html=True)
    if ins.interaction:
        st.markdown(f'<div class="insight">{esc(ins.interaction)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight">{esc(ins.accuracy)}</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(ins.table()), hide_index=True)
    c1, c2 = st.columns([2, 3])
    with c1:
        show(charts.variable_importance([v.name for v in ins.variables], [v.importance for v in ins.variables]))
    with c2:
        st.caption("Influence share = permutation importance on held-out rows. Elasticity = % change in the outcome for a "
                   "+10% nudge to one variable. Swing = how far the prediction moves across the variable's 5th-95th "
                   "percentile range, in units of the outcome's standard deviation.")
    show(charts.partial_effects(ins.variables, target))
    st.caption(ins.caveat)


def render_maths(c) -> None:
    st.markdown("**The expression**")
    st.latex(c.latex or c.expression)
    if c.expanded_latex:
        st.markdown("**Expanded form** (SymPy; the same function, multiplied out)")
        st.latex(c.expanded_latex)
    st.markdown("**Plain text, to copy into code or a spreadsheet**")
    st.code(c.expression, language=None)
    if c.expanded:
        st.code(c.expanded, language=None)
    st.caption("The evaluator treats square roots as acting on |x| and guards against dividing by zero; the displayed form drops "
               "those safeguards wherever your data make them irrelevant.")
    with st.expander("How the numbers are defined"):
        formulas_panel(["r2", "rmse", "mae", "nrmse", "fitness", "importance", "elasticity", "interval"])


def _sig(*parts) -> str:
    return format(zlib.crc32(repr(parts).encode("utf-8")), "x")


def render_optimizer(cands, c, ctx: dict, target: str, key: str) -> None:
    """Pick the best settings of the variables for a goal on the variable of interest."""
    X, y, names = ctx["X_all"], ctx["y_all"], ctx["features"]
    data_sig = format(zlib.crc32(np.ascontiguousarray(X).tobytes()), "x")
    k = f"{key}_{data_sig}"
    st.markdown(f"Tell me how much (or how little) **{target}** you want and I will look for settings of the other variables "
                f"that get there, using the expression you selected. I will never go outside the range your data cover.")
    q25, med, q75 = (float(v) for v in np.percentile(y, [25, 50, 75]))
    st.caption(f"In your data {target} runs from {y.min():.4g} to {y.max():.4g} (middle half: {q25:.4g} to {q75:.4g}, median {med:.4g}).")

    labels = {"maximize": "As high as possible", "minimize": "As low as possible",
              "target": "Close to a specific value", "range": "Inside a range"}
    mode = st.radio(f"I want {target} to be", list(labels), format_func=labels.get, horizontal=True, key=f"{k}_mode")
    goal = Goal(mode)
    if mode == "target":
        goal.target = float(st.number_input("Target value", value=med, format="%.6g", key=f"{k}_t"))
    elif mode == "range":
        a, b = st.columns(2)
        goal.low = float(a.number_input("From", value=q25, format="%.6g", key=f"{k}_lo"))
        goal.high = float(b.number_input("To", value=q75, format="%.6g", key=f"{k}_hi"))
    label, tex = OBJECTIVES[mode]
    st.latex(tex)

    st.markdown('<div class="small-label">Variables you can change</div>', unsafe_allow_html=True)
    locked, bounds = {}, {}
    for j, nm in enumerate(names):
        lo, hi = float(X[:, j].min()), float(X[:, j].max())
        c1, c2, c3 = st.columns([2, 1.3, 4])
        c1.markdown(f"`{nm}`")
        c1.caption(f"seen from {lo:.4g} to {hi:.4g}")
        if hi <= lo:
            c3.caption("constant in your data")
            continue
        fixed = c2.checkbox("Keep fixed", key=f"{k}_fix_{nm}", help="Use this for things you cannot change, like the weather.")
        if fixed:
            locked[nm] = float(c3.slider("Fixed at", lo, hi, float(np.median(X[:, j])), key=f"{k}_val_{nm}",
                                         label_visibility="collapsed"))
        else:
            bounds[nm] = tuple(float(v) for v in c3.slider("Allowed range", lo, hi, (lo, hi), key=f"{k}_rng_{nm}",
                                                           label_visibility="collapsed"))

    sig = _sig(mode, goal.target, goal.low, goal.high, sorted(locked.items()), sorted(bounds.items()), len(cands))
    if st.button("Find the best settings", type="primary", key=f"{k}_go"):
        try:
            goal.validate()
            with st.spinner("Searching the equation for the best settings..."):
                cmp_ = optimize_candidates(cands, X, names, goal, target, locked or None, bounds or None, y_ref=y)
            st.session_state[f"{k}_result"] = (sig, goal, cmp_)
        except ValueError as e:
            st.error(f"Could not optimise: {e}")
    stored = st.session_state.get(f"{k}_result")
    if stored is None:
        return
    if stored[0] != sig:
        st.info("You changed the goal or the variables since the last search. Press the button to update the recommendation.")
        return
    _, goal_used, cmp_ = stored
    res = cmp_.results[min(c.rank, len(cmp_.results)) - 1]

    section("Recommended settings")
    m = st.columns(3)
    m[0].metric(f"Predicted {target}", f"{res.predicted:.4g}", help=f"Typical error of this expression on unseen data: ±{res.rmse:.3g}")
    m[1].metric("At typical settings", f"{res.typical_prediction:.4g}")
    if goal_used.mode in ("maximize", "minimize"):
        m[2].metric("Change", f"{res.improvement:+.4g}")
    else:
        m[2].metric("Distance from goal", f"{res.gap:.4g}")
    rows = []
    for nm in res.names:
        if nm in res.locked:
            note = "kept fixed"
        elif nm in res.inert:
            note = "barely matters: any value will do"
        elif nm in res.edge:
            note = f"at the {res.edge[nm]} of your data"
        elif nm in res.touchy:
            note = "touchy: set carefully"
        else:
            note = ""
        rows.append({"variable": nm, "recommended": res.values[nm], "typical": res.typical[nm],
                     "allowed from": res.bounds[nm][0], "allowed to": res.bounds[nm][1], "note": note})
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True)
    st.download_button("Download these settings (CSV)", table.to_csv(index=False), file_name="lawshift_recommended_settings.csv",
                       key=f"{k}_dl")
    readout_panel(readout_optimization(res, cmp_))
    if res.free:
        show(charts.optimizer_curves(res))
        if len(cmp_.results) > 1:
            show(charts.optimum_agreement(cmp_.per_variable, res.bounds))
    with st.expander("Technical details"):
        st.markdown(f"Objective: {labels[goal_used.mode].lower()}")
        st.latex(OBJECTIVES[goal_used.mode][1])
        st.caption("Search: differential evolution over the allowed ranges (global), then a local polish. Among equally good "
                   "settings the one closest to typical conditions is chosen. B = the box of allowed ranges.")
        if res.sensitivity:
            st.markdown("Robustness: change in the predicted outcome if one variable moves by 10% of its allowed range")
            st.dataframe(pd.DataFrame(res.sensitivity), hide_index=True)
        if res.nearest_distance is not None:
            st.caption(f"Distance to the nearest real row: {res.nearest_distance:.3f} (typical neighbour gap in your data: "
                       f"{res.typical_gap:.3f}), in units of each variable's allowed range.")
        st.caption(f"Disagreement between the {len(cmp_.results)} expressions' optima: {cmp_.disagreement:.1%} of a variable's range.")


def render_candidates(cands, y_val, target: str, key: str, history=None, ctx: dict | None = None) -> None:
    section("The best expressions I found")
    for c in cands:
        candidate_card(c)
    if len(cands) < 3:
        st.caption(f"Only {len(cands)} distinct expression(s) met the quality bar; weaker or near-duplicate "
                   "candidates are not shown.")
    labels = [f"#{c.rank}" for c in cands]
    pick = st.radio("Look closer at expression", labels, horizontal=True, key=f"pick_{key}")
    c = cands[labels.index(pick)]
    names = ["In plain English", "Technical insights"] + (["Optimise"] if ctx else []) + ["Validation", "Compare", "The maths"]
    if history:
        names.append("Evolution")
    tabs = dict(zip(names, st.tabs(names)))
    with tabs["In plain English"]:
        readout_panel(readout_candidate(c, target, y_val, ctx["readout"] if ctx else None))
    with tabs["Technical insights"]:
        render_insights(c, target)
    if ctx:
        with tabs["Optimise"]:
            render_optimizer(cands, c, ctx, target, key)
    with tabs["Validation"]:
        a, b = st.columns(2)
        with a:
            show(charts.actual_vs_predicted(y_val, c.pred_val, f"#{c.rank}: actual vs predicted (held out)"))
        with b:
            show(charts.residuals(y_val, c.pred_val))
        st.dataframe(pd.DataFrame({"": ["Training", "Validation (held out)"],
                                   "R²": [c.train["r2"], c.validation["r2"]],
                                   "RMSE": [c.train["rmse"], c.validation["rmse"]],
                                   "MAE": [c.train["mae"], c.validation["mae"]]}), hide_index=True)
    with tabs["Compare"]:
        readout_panel(readout_run(cands, target))
        st.dataframe(pd.DataFrame([{"rank": f"#{x.rank}", "expression": x.expression, "nodes": x.size,
                                    "held-out R²": x.validation["r2"], "held-out RMSE": x.validation["rmse"]}
                                   for x in cands]), hide_index=True)
        show(charts.candidates_comparison(labels, [x.validation["r2"] for x in cands], [x.size for x in cands]))
        st.caption("Rank 1 is the best TRAINING fit; the others are alternatives that trade accuracy against "
                   "simplicity. Validation data is never used to choose or rank them.")
    with tabs["The maths"]:
        render_maths(c)
    if history:
        with tabs["Evolution"]:
            show(charts.fitness_history(history))


# --------------------------------------------------------------------------- Home
def page_home() -> None:
    page_header("LAWSHIFT", "Find the equation hiding in your data, test it on rows it has never seen, and use it to choose better settings.")
    demo_banner()
    records = store.load()
    stats = summary_stats(records)
    n_business = sum(1 for r in records if r.get("kind") == "business")
    n_data = sum(1 for r in records if r.get("kind") == "dataset")

    c = st.columns(5)
    c[0].metric("Simulated experiments", stats["n_experiments"])
    c[1].metric("Your-data analyses", n_data)
    c[2].metric("Business analyses", n_business)
    c[3].metric("Median held-out R²", "-" if stats["median_val_r2"] is None else f"{stats['median_val_r2']:.4f}")
    c[4].metric("Noise ceiling reached", f"{stats['reached_ceiling']}/{stats['n_experiments']}")
    st.caption("Every figure is computed from experiments stored on this machine. 'Noise ceiling' = the score the true law "
               "itself gets on the same noisy held-out data (simulated experiments only).")

    if not records:
        st.info("Nothing recorded yet. Open the Everyday Data Lab and load a sample, or run the quick suite below.")
    left, right = st.columns([3, 2])
    with left:
        section("Accuracy vs noise (stored simulations)")
        show(charts.history_scatter(records))
    with right:
        section("Try it in one click")
        st.write("Runs the three simulated physics experiments with the Quick preset (seed 0, 2% noise). "
                 "The searches run live, so it takes a little while.")
        if st.button("Run quick discovery suite", type="primary"):
            for key in available_systems():
                res = run_with_progress(make_config(key, 0.02, 200, 0.3, 0, "Quick demo"), f"{key}: ")
                store.append(res.to_record())
                st.success(f"{res.spec.title}: {res.best.expression}  (held-out R² {res.best.validation['r2']:.4f})")
            st.rerun()
        section("Where to start")
        st.markdown("1. **Everyday Data Lab**: explore any table, check how normal it is, find the pattern, pick the best settings\n"
                    "2. **Scientific Discovery Lab**: hidden-law experiments, or your own measurements\n"
                    "3. **Business Pattern Lab**: advertising, costs and outcomes, with unusual-row detection")


# --------------------------------------------------------------------------- shared data loading + discovery
def load_data_block(ns: str, everyday: bool):
    """Upload or pick a sample, clean it, preview it. Returns (df, report, name) or None."""
    section("1. Load your data")
    sample_label = "Sample dataset"
    source = st.radio("Data source", ["Upload CSV", sample_label], horizontal=True, key=f"{ns}_source")
    raw, name = None, None
    if source == "Upload CSV":
        up = st.file_uploader("CSV file: a header row, numeric columns, at least 30 complete rows", type=["csv"], key=f"{ns}_upload")
        if up is not None:
            if up.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"File is larger than {MAX_UPLOAD_MB} MB.")
            else:
                try:
                    raw, name = pd.read_csv(up), up.name
                except Exception as e:      # unreadable or malformed CSV
                    st.error(f"Could not read the CSV file: {e}")
    else:
        samples = [p for p in sorted(SAMPLE_DIR.glob("*.csv")) if (p.stem in EVERYDAY_SAMPLES) == everyday]
        if not samples:
            st.info("No sample files found in the sample_data folder. Run `python make_sample_data.py`.")
            return None
        choice = st.selectbox("Sample dataset", samples, format_func=lambda p: p.stem.replace("_", " ").title(), key=f"{ns}_sample")
        try:
            raw, name = pd.read_csv(choice), choice.name
        except Exception as e:
            st.error(f"Could not read the sample file: {e}")
        st.caption("SYNTHETIC data generated from a known law plus noise (see make_sample_data.py). The engine is never told the law.")
    if raw is None:
        st.info("Load a CSV to begin. Variable names in the discovered equations will come from your column headers.")
        return None
    try:
        df, report = prepare_numeric_dataframe(raw)
    except ValueError as e:
        st.error(str(e))
        return None
    bits = []
    if report["renamed_columns"]:
        bits.append("headers used as variable names: " + ", ".join(f"'{o}' → {n}" for o, n in report["renamed_columns"].items()))
    if report["dropped_rows"]:
        bits.append(f"{report['dropped_rows']} row(s) with missing/invalid values dropped")
    if report["ignored_columns"]:
        bits.append(f"ignored non-numeric columns: {report['ignored_columns']}")
    if report["constant_columns"]:
        bits.append(f"constant columns removed: {report['constant_columns']}")
    if bits:
        st.caption("Data cleaning: " + "; ".join(bits) + ".")
    st.dataframe(df.head(8), hide_index=True)
    st.caption(f"{len(df)} usable rows, {df.shape[1]} numeric columns.")
    return df, report, name


def render_dataset_result(res, ns: str) -> None:
    b = res.best
    section("Result")
    m = st.columns(4)
    m[0].metric("Best held-out R²", f"{b.validation['r2']:.4f}")
    m[1].metric("Best held-out RMSE", f"{b.validation['rmse']:.4g}")
    m[2].metric("Complexity", f"{b.size} nodes")
    m[3].metric("Searches in agreement", f"{res.runs_agreeing}/{res.n_runs}")
    render_candidates(res.candidates, res.y_val, res.target, ns, history=res.history, ctx=ctx_from_dataset(res))

    section("Your data")
    show(charts.raw_observations(res.features, res.X_train, res.y_train, res.target, res.X_val, res.y_val))
    st.dataframe(res.baselines, hide_index=True)
    st.caption("Baselines on the same held-out rows: the evolved expressions should beat the mean and be competitive with linear regression.")

    section("The honest report")
    for note in res.notices:
        st.info(note)
    st.caption(f"Saved to Experiment History. Total time {res.elapsed:.1f}s.")


def discovery_section(df: pd.DataFrame, report: dict, name: str, ns: str) -> None:
    """Choose the variable of interest, search for expressions, show results."""
    section("2. Choose the variable of interest")
    cols = list(df.columns)
    target = st.selectbox("Variable of interest (the quantity the equation should predict)", cols,
                          index=len(cols) - 1, key=f"{ns}_target_{name}")
    options = [c for c in cols if c != target]
    corr = df.corr()[target].abs().drop(target).sort_values(ascending=False)
    default = list(corr.index[:MAX_FEATURES])
    feats = st.multiselect(f"Independent variables (up to {MAX_FEATURES}; all other columns by default)", options,
                           default=default, key=f"{ns}_feats_{name}_{target}")
    if len(options) > MAX_FEATURES:
        st.caption(f"{len(options)} candidate columns: the {MAX_FEATURES} most correlated with {target} are pre-selected.")
    st.caption("You decide which variable depends on which. The engine cannot infer direction from data alone.")

    section("3. Search for expressions")
    c1, c2, c3 = st.columns(3)
    valf = c1.slider("Held-out validation fraction", 0.2, 0.5, 0.3, 0.05, key=f"{ns}_valf")
    seed = c2.number_input("Random seed", min_value=0, max_value=99999, value=0, step=1, key=f"{ns}_seed")
    effort = c3.selectbox("Search effort", list(PRESETS), index=0, key=f"{ns}_effort")
    p = PRESETS[effort]
    st.caption(f"{p['n_runs']} independent searches, population {p['pop_size']}, up to {p['generations']} generations.")
    if st.button("Discover expressions", type="primary", key=f"{ns}_run"):
        if not feats:
            st.error("Select at least one independent variable.")
        else:
            bar, cb = make_progress(p["n_runs"], p["generations"], "Searching")
            try:
                cfg = DatasetConfig(target=target, features=feats, val_fraction=float(valf), seed=int(seed),
                                    n_runs=p["n_runs"], evolution=EvolutionConfig(pop_size=p["pop_size"], generations=p["generations"]))
                res = run_dataset_experiment(df, cfg, progress_callback=cb, prep_report=report)
                bar.empty()
                store.append(res.to_record())
                st.session_state[f"{ns}_result"] = (name, res)
            except ValueError as e:
                bar.empty()
                st.error(f"Could not run the search: {e}")
    stored = st.session_state.get(f"{ns}_result")
    if stored is not None:
        st.divider()
        st.caption(f"Showing the latest result, from dataset: {stored[0]}")
        render_dataset_result(stored[1], ns)


# --------------------------------------------------------------------------- Scientific Discovery Lab
def render_lab_result(res) -> None:
    s, b = res.spec, res.best
    pred_va = res.predict_best(res.X_val)

    section("Best discovery")
    st.latex(best_latex(res))
    alg = algebraic_latex(res)
    if alg:
        st.caption("Expanded form (SymPy)")
        st.latex(alg)
    m = st.columns(5)
    m[0].metric("Held-out R²", f"{b.validation['r2']:.4f}")
    m[1].metric("Held-out RMSE", f"{b.validation['rmse']:.4g}")
    m[2].metric("Held-out MAE", f"{b.validation['mae']:.4g}")
    m[3].metric("Complexity", f"{b.size} nodes")
    m[4].metric("Noise ceiling R²", f"{res.noise_ceiling_r2:.4f}")

    tabs = st.tabs(["Observations", "Validation", "Evolution", "Model comparison", "Post-experiment verification"])
    with tabs[0]:
        st.write(f"{res.X_train.shape[0]} training and {res.X_val.shape[0]} held-out validation observations "
                 f"({res.config.noise_level:.0%} noise). The engine only ever saw the training rows.")
        show(charts.raw_observations(s.input_names, res.X_train, res.y_train, s.output_name,
                                     res.X_val, res.y_val, s.units))
        preview = pd.DataFrame(res.X_train, columns=list(s.input_names))
        preview[s.output_name] = res.y_train
        st.caption("Dataset preview (training rows)")
        st.dataframe(preview.head(12), hide_index=True)
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            show(charts.actual_vs_predicted(res.y_val, pred_va))
        with c2:
            show(charts.residuals(res.y_val, pred_va))
        if len(s.input_names) == 1:
            show(charts.curve_overlay(res.X_train, res.y_train, res.X_val, res.y_val, res.predict_best,
                                      s.input_names[0], s.output_name))
        st.dataframe(pd.DataFrame({"": ["Training", "Validation (held out)"],
                                   "R²": [b.train["r2"], b.validation["r2"]],
                                   "RMSE": [b.train["rmse"], b.validation["rmse"]],
                                   "MAE": [b.train["mae"], b.validation["mae"]]}), hide_index=True)
    with tabs[2]:
        show(charts.fitness_history(b.history))
        st.caption(f"{b.generations_run} generations run"
                   f"{' (stopped early: no further improvement)' if b.stopped_early else ''}. "
                   "Fitness = normalised prediction error + complexity penalty.")
        st.latex(FORMULAS["fitness"][1])
    with tabs[3]:
        rows = pd.DataFrame([{
            "search": f"run {i}{' (selected)' if i == res.best_index else ''}", "equation": r.expression,
            "nodes": r.size, "train fitness": r.fitness, "held-out R²": r.validation["r2"],
            "held-out RMSE": r.validation["rmse"]} for i, r in enumerate(res.runs)])
        st.dataframe(rows, hide_index=True)
        show(charts.run_comparison([f"run {i}" for i in range(len(res.runs))],
                                   [r.validation["rmse"] for r in res.runs], [r.size for r in res.runs], res.best_index))
        st.caption("The selected search is chosen by TRAINING fitness only; validation data is never used for selection.")
    with tabs[4]:
        st.markdown("**Hidden law used to generate the data (never shown to the search):**")
        if s.true_latex:
            st.latex(s.true_latex)
        else:
            st.markdown(f'<div class="hidden-box">{esc(s.true_equation)}</div>', unsafe_allow_html=True)
        st.write(f"Error of the discovered equation against the noise-free true law: "
                 f"**{b.nrmse_vs_true_law:.4f}** (NRMSE on held-out inputs).")
        with st.chat_message("assistant", avatar="🔬"):
            for line in res.interpretation:
                st.markdown(f"- {line}")
        st.caption(READOUT_NOTE)
        if len(s.input_names) == 1:
            show(charts.curve_overlay(res.X_train, res.y_train, res.X_val, res.y_val, res.predict_best,
                                      s.input_names[0], s.output_name, s.true_fn))
        with st.expander("Model assumptions and limitations"):
            for a in s.assumptions:
                st.markdown(f"- {a}")


def lab_simulated() -> None:
    systems = available_systems()
    left, right = st.columns([1, 2])
    with left:
        key = st.selectbox("Simulated system", list(systems), format_func=lambda k: systems[k].title)
        spec = systems[key]
        st.caption(spec.description)
        st.caption(f"Variables: {', '.join(f'{n} [{spec.units[n]}]' for n in spec.input_names)} → "
                   f"{spec.output_name} [{spec.units[spec.output_name]}]")
        noise = st.slider("Measurement noise (fraction of signal std)", 0.0, 0.30, 0.02, 0.01)
        n = st.slider("Observations", 60, 1000, 200, 20)
        valf = st.slider("Validation fraction", 0.2, 0.5, 0.3, 0.05)
        seed = st.number_input("Random seed", min_value=0, max_value=99999, value=0, step=1)
        preset = st.selectbox("Search effort", list(PRESETS), index=0)
        p = PRESETS[preset]
        st.caption(f"{p['n_runs']} independent searches, population {p['pop_size']}, up to {p['generations']} generations.")
        run = st.button("Run discovery experiment", type="primary")
    with right:
        st.markdown("**How it works**")
        st.markdown(
            "1. Observations are generated from a **hidden** law, with measurement noise.\n"
            "2. A share of the data is **held out** and never shown to the search.\n"
            "3. An evolutionary algorithm breeds candidate equations (selection, crossover, mutation, elitism).\n"
            "4. The best equation is tested on the held-out data.\n"
            "5. Only then is the hidden law revealed for comparison.")
        with st.expander("Model assumptions"):
            for a in spec.assumptions:
                st.markdown(f"- {a}")
        if run:
            try:
                res = run_with_progress(make_config(key, noise, n, valf, seed, preset))
                rec = store.append(res.to_record())
                st.session_state["lab_result"] = res
                st.session_state["lab_record_id"] = rec["id"]
            except ValueError as e:
                st.error(f"Could not run the experiment: {e}")
    res = st.session_state.get("lab_result")
    if res is not None:
        st.divider()
        render_lab_result(res)
        st.caption(f"Saved to Experiment History (id {st.session_state.get('lab_record_id')}). Total time {res.elapsed:.1f}s.")


def page_lab() -> None:
    page_header("Scientific Discovery Lab", "Find the equation hiding in measurements, then test it on data it has never seen.")
    demo_banner()
    mode = st.radio("Mode", ["Your data (CSV)", "Simulated systems (hidden-law experiments)"], horizontal=True, key="lab_mode")
    if mode.startswith("Your"):
        loaded = load_data_block("ds", everyday=False)
        if loaded is not None:
            df, report, name = loaded
            discovery_section(df, report, name, "ds")
    else:
        lab_simulated()


# --------------------------------------------------------------------------- Everyday Data Lab
def page_everyday() -> None:
    page_header("Everyday Data Lab", "Any ordinary table: see its shape, check how normal it is, find the pattern, then choose the best settings.")
    demo_banner()
    loaded = load_data_block("ev", everyday=True)
    if loaded is None:
        return
    df, report, name = loaded
    profiles, table = profile_table(df)
    tabs = st.tabs(["Overview", "Distributions & normality", "Relationships", "Pattern finder & optimiser"])

    with tabs[0]:
        m = st.columns(3)
        m[0].metric("Rows", len(df))
        m[1].metric("Numeric columns", df.shape[1])
        m[2].metric("Columns that look normal", sum(p.verdict in ("normal", "nearly_normal") for p in profiles))
        readout_panel(readout_table(profiles, strongest_pairs(df.corr(), 3), len(df)))
        st.dataframe(table, hide_index=True)
        show(charts.distribution_overview(df, {p.name: p.label for p in profiles}))
        st.caption("Shape labels combine the Shapiro-Wilk test with skewness and kurtosis: a test alone is too strict with "
                   "many rows and too lenient with few.")

    with tabs[1]:
        col = st.selectbox("Column", list(df.columns), key=f"ev_norm_{name}")
        p = next(x for x in profiles if x.name == col)
        m = st.columns(4)
        m[0].metric("Mean", f"{p.mean:.4g}")
        m[1].metric("Standard deviation", f"{p.std:.4g}")
        m[2].metric("Skewness", f"{p.skew:+.2f}")
        m[3].metric("Shape", p.label)
        readout_panel(Readout(paragraphs=[readout_column(p)]))
        show(charts.distribution_panel(df[col], col, p.mean, p.std))
        st.dataframe(pd.DataFrame({
            "within": ["1 standard deviation", "2 standard deviations", "3 standard deviations"],
            "expected if normal": [0.6827, 0.9545, 0.9973],
            "your data": [p.within_1sd, p.within_2sd, p.within_3sd]}), hide_index=True)
        with st.expander("The maths of normality"):
            formulas_panel(["normal_pdf", "zscore", "empirical_rule", "skewness", "kurtosis", "shapiro"])

    with tabs[2]:
        corr = df.corr()
        show(charts.correlation_heatmap(corr))
        pairs = strongest_pairs(corr, 8)
        st.dataframe(pd.DataFrame(pairs, columns=["variable A", "variable B", "Pearson r"]), hide_index=True)
        c1, c2 = st.columns(2)
        a = c1.selectbox("Variable on the x-axis", list(df.columns), index=0, key=f"ev_pa_{name}")
        b = c2.selectbox("Variable on the y-axis", list(df.columns), index=min(1, df.shape[1] - 1), key=f"ev_pb_{name}")
        if a != b:
            show(charts.pair_scatter(df[a], df[b], a, b, float(df[a].corr(df[b]))))
        st.caption("Association only. " + CAUSATION_NOTICE)

    with tabs[3]:
        discovery_section(df, report, name, "ev")


# --------------------------------------------------------------------------- Business Pattern Lab
def page_business() -> None:
    page_header("Business Pattern Lab", "Association, not causation: patterns between activities, costs and outcomes.")
    demo_banner()
    st.warning(CAUSATION_NOTICE)
    source = st.radio("Data source", ["Demonstration dataset (synthetic)", "Upload CSV"], horizontal=True)
    raw = None
    if source.startswith("Demo"):
        raw = demo_dataset()
        st.info("Using a SYNTHETIC demonstration dataset: advertising, units sold, operating expenses, revenue and "
                "profit. It is generated by code, is not real company data, and contains a few deliberately unusual rows.")
    else:
        up = st.file_uploader("CSV file with numeric columns (at least 30 complete rows)", type=["csv"])
        if up is not None:
            if up.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"File is larger than {MAX_UPLOAD_MB} MB.")
            else:
                try:
                    raw = pd.read_csv(up)
                except Exception as e:  # unreadable or malformed CSV
                    st.error(f"Could not read the CSV file: {e}")
    if raw is None:
        return
    try:
        df, report = prepare_dataframe(raw)
    except ValueError as e:
        st.error(str(e))
        return
    if report["dropped_rows"] or report["ignored_columns"] or report["constant_columns"]:
        st.caption(f"Data cleaning: {report['dropped_rows']} row(s) with missing/invalid values dropped; "
                   f"ignored non-numeric columns: {report['ignored_columns'] or 'none'}; "
                   f"constant columns removed: {report['constant_columns'] or 'none'}.")
    section("Dataset preview")
    st.dataframe(df.head(10), hide_index=True)
    st.caption(f"{len(df)} usable rows, {df.shape[1]} numeric columns.")

    cols = list(df.columns)
    default_target = "profit" if "profit" in cols else cols[-1]
    c1, c2, c3 = st.columns([1, 2, 1])
    target = c1.selectbox("Outcome to explain", cols, index=cols.index(default_target))
    options = [c for c in cols if c != target]
    default_feats = [c for c in ["advertising", "units_sold", "operating_expenses"] if c in options] or options[:3]
    feats = c2.multiselect("Predictor variables (max 6)", options, default=default_feats)
    effort = c3.selectbox("Search effort", ["Quick demo", "Standard"], index=0)
    if st.button("Analyse patterns", type="primary"):
        if not feats:
            st.error("Select at least one predictor variable.")
            return
        p = PRESETS[effort]
        n_runs = 2 if effort == "Quick demo" else 3
        bar, cb = make_progress(n_runs, p["generations"], "Searching for candidate equations")
        try:
            res = analyze(df, target, feats, seed=0, n_runs=n_runs,
                          evolution=EvolutionConfig(pop_size=min(p["pop_size"], 120), generations=p["generations"]),
                          progress_callback=cb, prep_report=report)
            bar.empty()
            store.append(res.to_record())
            st.session_state["biz_result"] = res
        except ValueError as e:
            bar.empty()
            st.error(f"Analysis failed: {e}")
    res = st.session_state.get("biz_result")
    if res is None:
        return
    st.divider()
    m = res.models.set_index("model")
    k = st.columns(4)
    k[0].metric("Linear model held-out R²", f"{m.loc['Linear regression', 'val_r2']:.3f}")
    k[1].metric("Best expression held-out R²", f"{m.loc['Evolved equation', 'val_r2']:.3f}")
    k[2].metric("Unusual observations", int(res.anomaly_mask.sum()))
    k[3].metric("Train / validation rows", f"{res.n_train} / {res.n_val}")
    for note in res.notices[1:]:
        st.info(note)
    render_candidates(res.candidates, res.y_val, res.target, "business", history=res.history, ctx=ctx_from_business(res))
    section("Further analysis")
    t = st.tabs(["Correlations", "Linear regression", "Model comparison", "Prediction errors", "Unusual observations"])
    with t[0]:
        show(charts.correlation_heatmap(res.corr_matrix))
        st.dataframe(res.target_corr, hide_index=True)
        st.caption("Association only. " + CAUSATION_NOTICE)
    with t[1]:
        st.markdown("**Linear regression (original units)**")
        st.latex(res.linear_latex)
        st.code(res.linear_equation, language=None)
    with t[2]:
        st.dataframe(res.models, hide_index=True)
        show(charts.model_comparison(list(res.models["model"]), list(res.models["val_rmse"]),
                                     list(res.models["val_r2"])))
    with t[3]:
        c1, c2 = st.columns(2)
        with c1:
            show(charts.actual_vs_predicted(res.y_val, res.pred_val_equation, "Best expression: actual vs predicted"))
        with c2:
            show(charts.actual_vs_predicted(res.y_val, res.pred_val_linear, "Linear model: actual vs predicted"))
        st.dataframe(res.val_table.head(25), hide_index=True)
    with t[4]:
        show(charts.anomaly_plot(res.y_all, res.pred_linear_all, res.anomaly_mask, res.target))
        if len(res.anomalies):
            st.dataframe(res.anomalies, hide_index=True)
        else:
            st.write("No observations exceeded the unusual-residual threshold.")
        st.caption("Flagged by a robust (median/MAD) residual test on the linear model. "
                   "A flag means 'unlike the rest', not 'error' or 'fraud'.")
        with st.expander("The maths"):
            formulas_panel(["robust_z"])


# --------------------------------------------------------------------------- History
def history_row(r: dict) -> dict:
    kind = r.get("kind")
    if kind == "physics":
        return {"id": r["id"], "time (UTC)": r["timestamp"], "type": "simulated", "system/target": r["system"],
                "equation": r["expression"], "held-out R²": r["validation"]["r2"], "nodes": r["complexity"]}
    if kind == "dataset":
        return {"id": r["id"], "time (UTC)": r["timestamp"], "type": "your data", "system/target": r["target"],
                "equation": r["expression"], "held-out R²": r["validation"]["r2"], "nodes": r["complexity"]}
    return {"id": r["id"], "time (UTC)": r["timestamp"], "type": "business", "system/target": r.get("target"),
            "equation": r.get("equation") or r.get("linear_equation"),
            "held-out R²": (r.get("val_r2") or {}).get("Evolved equation"), "nodes": r.get("complexity")}


def page_history() -> None:
    page_header("Experiment History", "Every run is stored locally and can be inspected, downloaded or deleted.")
    records = store.load()
    if not records:
        st.info("No experiments stored yet. Results appear here after you run one.")
        return
    st.dataframe(pd.DataFrame([history_row(r) for r in reversed(records)]), hide_index=True)
    inspectable = [r for r in reversed(records) if r.get("kind") in ("physics", "dataset") and r.get("fitness_history")]
    if inspectable:
        section("INSPECT AN EXPERIMENT")
        pick = st.selectbox("Experiment", [r["id"] for r in inspectable],
                            format_func=lambda i: next(f"{r['id']} - {r.get('system') or r.get('target')} - {r['expression']}"
                                                       for r in inspectable if r["id"] == i))
        rec = next(r for r in inspectable if r["id"] == pick)
        c = st.columns(4)
        c[0].metric("Held-out R²", f"{rec['validation']['r2']:.4f}")
        c[1].metric("Complexity", f"{rec['complexity']} nodes")
        if rec["kind"] == "physics":
            c[2].metric("Noise ceiling R²", f"{rec['noise_ceiling_r2']:.4f}")
            c[3].metric("Runs near ceiling", f"{rec['runs_near_ceiling']}/{rec['n_runs']}")
            st.dataframe(pd.DataFrame(rec["run_summaries"]), hide_index=True)
        else:
            c[2].metric("Searches in agreement", f"{rec['runs_agreeing']}/{rec['n_runs']}")
            c[3].metric("Rows", rec["n_rows"])
            st.dataframe(pd.DataFrame(rec["candidates"]), hide_index=True)
        show(charts.fitness_history(rec["fitness_history"]))
    st.download_button("Download history (JSON Lines)", store.path.read_text(encoding="utf-8"), file_name="lawshift_history.jsonl")
    with st.expander("Danger zone"):
        if st.checkbox("I understand this permanently deletes all stored experiments"):
            if st.button("Delete history"):
                store.clear()
                for k in ("lab_result", "biz_result", "ds_result"):
                    st.session_state.pop(k, None)
                st.rerun()


# --------------------------------------------------------------------------- About
def page_about() -> None:
    page_header("About LAWSHIFT", "What it does, how it works, and what it cannot tell you.")
    demo_banner()
    st.markdown("""
**LAWSHIFT** is a computational discovery engine. Given a table of observations and a variable of interest, it searches
for mathematical expressions that explain that variable, checks them against data the search never saw, explains them in
plain English, and can use them to suggest settings that push the variable of interest where you want it.

### The discovery cycle
Observed data → candidate equations → evolutionary search → best expressions → independent validation → plain-English read-out → optional optimisation.

### The three labs
- **Everyday Data Lab**: any ordinary table. Overview, distribution and normality checks, relationships, then the pattern finder and optimiser.
- **Scientific Discovery Lab**: your own measurements, or simulated systems with a hidden law revealed only after validation.
- **Business Pattern Lab**: advertising, costs and outcomes, with correlations, linear regression and unusual-observation flags.

### Using your own data
- Upload a CSV. **Column headers become the variable names** in the equations (`Length (m)` becomes `Length_m`).
- **You choose the variable of interest** (the dependent variable); the other numeric columns are the independent variables.
  The engine cannot infer which variable depends on which, so that decision is yours.
- The app shows the **top 3 distinct expressions** (typeset in LaTeX): the best training fit plus alternatives that trade accuracy against simplicity.
- Each expression comes with a **plain-English read-out** and the **technical insights** behind it: which variables drive it, in which
  direction, whether returns diminish, where a sweet spot sits, whether variables interact, and the typical error on unseen data.

### Choosing optimal values
In the **Optimise** tab you say how much of the variable of interest you want: as high as possible, as low as possible, close to a
specific value, or inside a range. You can keep some variables fixed (the weather, say) and narrow the range of others. The optimiser
searches the fitted expression for the best settings, never leaves the range your data cover, reports how touchy each setting is,
compares the recommendation with real rows, and checks whether the top expressions agree with each other.
""")
    st.latex(OBJECTIVES["maximize"][1] + r"\qquad " + OBJECTIVES["target"][1])
    st.markdown("""
### How the engine works (symbolic regression)
- Candidate equations are **expression trees** built from `+ - * /`, `sin`, `cos`, `exp`, square and square root.
- A genetic algorithm evolves a population using **tournament selection, subtree crossover, several mutation operators and elitism**.
- **Fitness** = normalised prediction error + a penalty for complexity, so simple equations are preferred.
""")
    st.latex(FORMULAS["fitness"][1])
    st.markdown("""
- Numeric constants are tuned with SciPy. Columns are rescaled for the search, and the winning equations are converted
  **exactly** back into your original units.
- Expressions are objects, never strings: there is no `eval()`. Division by zero, overflow and invalid values are neutralised.
- Experiments are **reproducible** from a random seed.

### About the plain-English read-out
The conversational notes are written on your machine by rules from the computed numbers (held-out R², permutation importance,
partial-effect curves, residual diagnostics, optimiser output). They are not produced by a language model, nothing leaves your
computer, and every statement can be checked against the technical tabs beside it.

### Scientific honesty
- In simulated experiments the hidden law is **never** given to the search; it is shown only after validation.
- Validation rows are never used to choose or rank expressions.
- Several independent searches are run, and the app reports how many agree.
- Insights and optimiser recommendations describe the fitted equation inside the observed range. They are **not proof of cause and effect**.
- Good agreement with a *simulated* law is not proof of a physical law.

### Limitations
- Physics data are **simulated** from simplified models (one dimension, no air resistance, no damping, Gaussian noise only).
  Not a substitute for laboratory measurement or certified engineering analysis.
- Uploaded data has no ground truth; held-out accuracy and search agreement are the only evidence, and neither is proof.
- The operator set is deliberately small; very complex laws may not be found. Results vary with the random seed.
- Validation covers the range of your data only (no extrapolation tests); the optimiser is therefore confined to that range.
- Up to 6 independent variables and 50,000 rows; files up to 5 MB.
- Business analysis shows **association, not causation**.

### Technology
Python, Streamlit, NumPy, Pandas, SciPy, SymPy (display only) and Matplotlib. LaTeX is rendered by Streamlit's built-in KaTeX.
Colours follow the 60-30-10 rule: warm paper neutrals (60%), sage greens (30%) and a terracotta accent (10%).
No paid APIs, no cloud services, no external AI. Everything runs locally.
""")


# --------------------------------------------------------------------------- main
def main() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand" style="display: flex; align-items: center; gap: 10px; font-weight: bold; font-size: 22px;">
                <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://w3.org">
                    <!-- Cyan Hexagon Outline -->
                    <polygon points="16,2 28,9 28,23 16,30 4,23 4,9" stroke="#2BE4F8" stroke-width="2.4" stroke-linejoin="round" fill="#0B132B"/>
                    <!-- Purple L Shape -->
                    <path d="M11.5 10.5V21.5H20.5" stroke="#A675F5" stroke-width="2.8" stroke-linecap="square"/>
                    <!-- Pink/Magenta Dot -->
                    <circle cx="19" cy="12.5" r="2.2" fill="#E838D5"/>
                </svg>
                <span>LAWSHIFT</span>
            </div>
            <div class="brand-sub">Find the pattern in your data</div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("All figures come from real computations run on this machine.")

    {
        "Home": page_home,
        "Scientific Discovery Lab": page_lab,
        "Everyday Data Lab": page_everyday,
        "Business Pattern Lab": page_business,
        "Experiment History": page_history,
        "About LAWSHIFT": page_about,
    }[page]()



main()
