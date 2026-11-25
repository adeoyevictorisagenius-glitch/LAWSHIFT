"""A plain-English "analyst's note" written from the numbers LAWSHIFT has already computed.

The voice is conversational ("here's what I see"), but nothing here calls a language model: every sentence is
assembled by rules from computed quantities (held-out R², permutation importance, partial-effect curves,
residual diagnostics, optimiser output). The same inputs always give the same words, and every claim can be
checked against the technical tables shown next to it. Technical terms are kept in brackets so the plain
version and the precise version sit side by side.

Variable names are wrapped in backticks, so Markdown never mistakes underscores in a column header for italics.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..data.normality import ColumnProfile
from .diagnostics import ResidualDiagnostics, residual_diagnostics
from .insights import MEANINGFUL_SHARE, VariableInsight
from .optimize import OptimizationComparison, OptimizationResult


@dataclass
class Readout:
    paragraphs: List[str] = field(default_factory=list)     # Markdown
    next_steps: List[str] = field(default_factory=list)
    cautions: List[str] = field(default_factory=list)


@dataclass
class ReadoutContext:
    """Extra facts the read-out can use when the caller has them."""
    n_rows: Optional[int] = None
    n_runs: Optional[int] = None
    runs_agreeing: Optional[int] = None          # searches that reached an equivalent best equation (rank 1 only)
    linear_val_r2: Optional[float] = None        # held-out R² of plain linear regression on the same rows


# --------------------------------------------------------------------------- small helpers
def _n(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    a = abs(x)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 100:
        return f"{x:.0f}"
    if a >= 10:
        return f"{x:.1f}"
    if a >= 1:
        return f"{x:.2f}"
    return f"{x:.3g}"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _v(name: str) -> str:
    return f"`{name}`"


def _join(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _pick(options: Sequence[str], key: str) -> str:
    """Deterministic variety: the same key always gives the same phrasing."""
    return options[zlib.crc32(key.encode("utf-8")) % len(options)]


def _fit_word(r2: float) -> str:
    if r2 >= 0.95:
        return "very closely"
    if r2 >= 0.8:
        return "well"
    if r2 >= 0.5:
        return "only partly"
    return "poorly"


# --------------------------------------------------------------------------- one variable
def _driver_sentence(v: VariableInsight, target: str) -> str:
    name = _v(v.name)
    if v.direction in ("increasing", "decreasing"):
        verb = "rises" if v.delta > 0 else "falls"
        s = (f"Going from a low {name} to a high one ({_n(v.low)} → {_n(v.high)}, the middle half of your data), "
             f"{_v(target)} {verb} by about **{_n(abs(v.delta))}**")
        if v.elasticity is not None and abs(v.elasticity) >= 0.5:
            s += (f"; a 10% nudge to {name} from a typical value moves {_v(target)} by roughly "
                  f"**{v.elasticity:+.1f}%** (elasticity)")
        s += "."
        if v.shape == "flattens out" and v.gain80_x is not None:
            s += (f" The effect levels off, though: about 80% of it is already reached by {name} ≈ {_n(v.gain80_x)}, "
                  f"so pushing further changes little (diminishing returns).")
        elif v.shape == "accelerates":
            s += f" The higher {name} goes, the more each extra unit matters (the effect accelerates)."
        elif v.shape == "roughly linear":
            s += " The effect is roughly proportional across the range."
        return s
    if v.direction == "rises then falls":
        where = f" near {_n(v.turning_x)}" if v.turning_x is not None else ""
        return (f"{name} has a **sweet spot**{where}: {_v(target)} climbs up to that point and then drops again, "
                f"so more is not better here (a peak).")
    if v.direction == "falls then rises":
        where = f" near {_n(v.turning_x)}" if v.turning_x is not None else ""
        return (f"{name} has a **low point**{where}: {_v(target)} dips to that point and then recovers, "
                f"so the worst setting sits in the middle of its range (a dip).")
    return (f"{_v(target)} goes up and down repeatedly as {name} changes, a **cyclical pattern** "
            f"(oscillation) rather than a steady trend.")


# --------------------------------------------------------------------------- one expression
def readout_candidate(c, target: str, y_val: np.ndarray, ctx: Optional[ReadoutContext] = None) -> Readout:
    """Conversational read-out for one expression. `c` is a Candidate with insights."""
    ctx = ctx or ReadoutContext()
    out = Readout()
    ins = c.insights
    r2v, r2t = c.validation["r2"], c.train["r2"]
    t = _v(target)

    opener = _pick(["Here is how I read this expression.", "Here is what this equation is telling me.",
                    "Here is my read on this one."], c.expression)
    s = (f"{opener} On rows it never saw, it explains **{_pct(max(r2v, 0.0))}** of the ups and downs in {t} "
         f"(held-out R² = {r2v:.3f}), so it tracks your data **{_fit_word(r2v)}**.")
    if ins is not None and ins.mean_abs_target > 0:
        s += (f" A typical miss is about **{_n(ins.rmse)}**, roughly {100 * ins.rmse / ins.mean_abs_target:.1f}% of the "
              f"average value (RMSE).")
    out.paragraphs.append(s)
    if r2v < 0.5:
        out.cautions.append("The fit on unseen rows is weak, so treat everything below as a faint hint rather than a finding. "
                            "The chosen variables may not explain this outcome well, or the relationship may be too noisy "
                            "or too complex for the building blocks available.")

    if ins is not None:
        ranked = sorted(ins.variables, key=lambda v: -v.importance)
        drivers = [v for v in ranked if v.importance >= MEANINGFUL_SHARE and v.used]
        if not drivers:
            out.paragraphs.append("None of the variables clearly moves the prediction, so there is no real driver to point to "
                                  "(permutation importance is negligible for all of them).")
        elif len(drivers) == 1:
            out.paragraphs.append(f"**{_v(drivers[0].name)}** is the only thing that really matters here: shuffling it alone "
                                  f"wrecks the predictions (permutation importance, {_pct(drivers[0].importance)}).")
        else:
            lead = _pick(["does most of the work", "is the biggest lever", "matters most"], c.expression + "lead")
            rest = ", then ".join(f"{_v(v.name)} ({_pct(v.importance)})" for v in drivers[1:3])
            out.paragraphs.append(f"{_v(drivers[0].name)} {lead} ({_pct(drivers[0].importance)} of the equation's sensitivity), "
                                  f"followed by {rest}. These shares come from shuffling each variable and measuring how "
                                  f"much worse the predictions get (permutation importance).")
        for v in drivers[:3]:
            out.paragraphs.append("▸ " + _driver_sentence(v, target))
        if ins.interaction:
            text = ins.interaction
            for nm in sorted((v.name for v in ins.variables), key=len, reverse=True):   # code-format names, longest first
                text = re.sub(rf"(?<![`\w]){re.escape(nm)}(?![`\w])", f"`{nm}`", text)
            out.paragraphs.append("One more subtlety: " + text[0].lower() + text[1:] +
                                  " In other words, the variables **interact**: you cannot judge one without the other.")
        unused = [v.name for v in ranked if not v.used]
        if unused:
            out.paragraphs.append(f"I did not need {_join([_v(u) for u in unused])}. Within this data it adds nothing beyond "
                                  f"what the other variables already say about {t}. That is not proof it is unrelated, only "
                                  f"that the search found no use for it.")

    # how far to trust it
    trust: List[str] = []
    gap = r2t - r2v
    if gap > 0.1:
        trust.append(f"it fits the rows it learned from noticeably better (R² {r2t:.3f}) than unseen rows ({r2v:.3f}), "
                     f"which can mean it has memorised some noise (over-fitting)")
    else:
        trust.append(f"it does about as well on unseen rows (R² {r2v:.3f}) as on the rows it learned from ({r2t:.3f}), "
                     f"which is reassuring")
    if ctx.linear_val_r2 is not None:
        d = r2v - ctx.linear_val_r2
        if d > 0.02:
            trust.append(f"it beats a plain straight-line model (R² {ctx.linear_val_r2:.3f}), so the extra curvature earns its keep")
        elif d >= -0.02:
            trust.append(f"it is no better than a plain straight line (R² {ctx.linear_val_r2:.3f}); the simpler line may be "
                         f"the wiser choice")
        else:
            trust.append(f"a plain straight line actually does better (R² {ctx.linear_val_r2:.3f}); prefer the line")
    if c.rank == 1 and ctx.runs_agreeing is not None and ctx.n_runs and ctx.n_runs > 1:
        trust.append(f"{ctx.runs_agreeing} of {ctx.n_runs} independent searches landed on an equivalent equation")
    if c.size <= 9:
        trust.append(f"it is short ({c.size} building blocks), which makes a lucky coincidence less likely")
    elif c.size > 15:
        trust.append(f"it is long ({c.size} building blocks), and long formulas can match noise by accident")
    out.paragraphs.append("**How far I would trust it:** " + "; ".join(trust) + ".")

    # the mistakes it makes
    try:
        d = residual_diagnostics(y_val, c.pred_val)
    except ValueError:
        d = None
    if d is not None:
        out.paragraphs.append(_error_paragraph(d, ins.rmse if ins else d.rmse))
        if d.biased:
            out.cautions.append(f"The equation is systematically {'low' if d.mean_error > 0 else 'high'} on unseen rows "
                                f"(average error {_n(d.mean_error)}), so a constant shift may be missing.")
        if d.errors_grow:
            out.cautions.append("Its errors change size as the prediction changes (heteroscedasticity), so a single "
                                "'typical error' understates the uncertainty in some regions.")

    top = [v.name for v in sorted(ins.variables, key=lambda v: -v.importance) if v.used][:2] if ins else []
    if top:
        out.next_steps.append(f"Open the **Optimise** tab to find settings of {_join([_v(x) for x in top])} that push {t} "
                              f"where you want it: higher, lower, or at a specific number.")
    if ins is not None:
        out.cautions.append(ins.caveat)
    return out


def _error_paragraph(d: ResidualDiagnostics, rmse: float) -> str:
    s = (f"**What its mistakes look like:** for a single new case I would expect the real value to land within about "
         f"**±{_n(d.interval95)}** of the prediction (a rough 95% range)")
    if d.errors_look_normal is True:
        s += ", and the errors are bell-shaped, so that range is reasonable."
    elif d.errors_look_normal is False:
        s += ", though the errors are not bell-shaped, so treat that range as rough."
    else:
        s += "."
    if not d.biased:
        s += " The equation is not leaning high or low on average."
    if d.errors_grow:
        s += (f" The size of the errors changes with the prediction (Spearman ρ = {d.hetero_rho:+.2f}), so the range is "
              f"wider in some regions than others.")
    return s


# --------------------------------------------------------------------------- the set of expressions
def readout_run(cands, target: str) -> Readout:
    """How the top expressions relate to each other and which to pick."""
    out = Readout()
    if not cands:
        return out
    best = cands[0]
    if len(cands) == 1:
        out.paragraphs.append(f"Only one distinct equation met the quality bar, so there is no simpler or sturdier alternative "
                              f"to compare. It has {best.size} building blocks and held-out R² {best.validation['r2']:.3f}.")
        return out
    bits = []
    for c in cands[1:]:
        d_size = c.size - best.size
        d_r2 = c.validation["r2"] - best.validation["r2"]
        shape = (f"simpler ({c.size} vs {best.size} blocks)" if d_size < 0 else
                 f"more complex ({c.size} vs {best.size} blocks)" if d_size > 0 else "the same size")
        acc = ("with no real loss of accuracy" if d_r2 > -0.005 else
               f"giving up {abs(d_r2):.3f} of R²")
        bits.append(f"**#{c.rank}** is {shape}, {acc}")
    out.paragraphs.append(f"I found **{len(cands)} genuinely different equations** for {_v(target)}. #1 is the best fit on the "
                          f"training rows; " + "; ".join(bits) + ".")
    simpler = [c for c in cands[1:] if c.size < best.size and c.validation["r2"] >= best.validation["r2"] - 0.01]
    if simpler:
        s = min(simpler, key=lambda c: c.size)
        out.next_steps.append(f"If you need a formula you can explain in one line, **#{s.rank}** costs almost nothing in accuracy.")
    return out


# --------------------------------------------------------------------------- ordinary data: distributions
def readout_column(p: ColumnProfile) -> str:
    """Plain-language description of one column's distribution."""
    s = (f"{_v(p.name)} averages **{_n(p.mean)}** with a spread of **{_n(p.std)}** (standard deviation); half of the values "
         f"sit between {_n(p.q1)} and {_n(p.q3)}, and the full range is {_n(p.minimum)} to {_n(p.maximum)}. "
         f"{p.verdict_text}")
    if p.shapiro_p is not None:
        s += (f" (Shapiro–Wilk p = {p.shapiro_p:.3f}, skewness {p.skew:+.2f}, excess kurtosis {p.excess_kurtosis:+.2f}.)")
    s += (f" For a normal variable, about 68%, 95% and 99.7% of values fall within 1, 2 and 3 standard deviations; here it is "
          f"{_pct(p.within_1sd)}, {_pct(p.within_2sd)} and {_pct(p.within_3sd)}.")
    if p.outliers_z3:
        s += (f" {p.outliers_z3} value{'s' if p.outliers_z3 != 1 else ''} lie more than 3 standard deviations from the mean "
              f"(worth a second look).")
    elif p.outliers_iqr:
        s += (f" {p.outliers_iqr} value{'s' if p.outliers_iqr != 1 else ''} fall outside the usual box-plot fences "
              f"(1.5 × IQR), which is not unusual for real data.")
    else:
        s += " No values stand out as extreme."
    if p.suggestion:
        s += " " + p.suggestion
    return s


def readout_table(profiles: Sequence[ColumnProfile], pairs: Sequence[Tuple[str, str, float]], n_rows: int) -> Readout:
    """Overview of a whole table."""
    out = Readout()
    normal = [p.name for p in profiles if p.verdict in ("normal", "nearly_normal")]
    skewed = [p.name for p in profiles if p.verdict.startswith("skewed")]
    other = [p.name for p in profiles if p.verdict not in ("normal", "nearly_normal") and not p.verdict.startswith("skewed")]
    s = f"I looked at **{len(profiles)} numeric columns** across **{n_rows} rows**. "
    if normal:
        s += f"{_join([_v(x) for x in normal])} {'look' if len(normal) > 1 else 'looks'} normal or close to it. "
    if skewed:
        s += (f"{_join([_v(x) for x in skewed])} {'are' if len(skewed) > 1 else 'is'} skewed, with a long tail on one side "
              f"(means can mislead there; medians are safer). ")
    if other:
        s += f"{_join([_v(x) for x in other])} depart{'' if len(other) > 1 else 's'} from a bell curve in other ways. "
    out.paragraphs.append(s.strip())
    if pairs:
        bits = []
        for a, b, r in pairs:
            word = "strongly" if abs(r) >= 0.7 else "moderately" if abs(r) >= 0.4 else "weakly"
            bits.append(f"{_v(a)} and {_v(b)} move {word} {'together' if r > 0 else 'in opposite directions'} (r = {r:+.2f})")
        out.paragraphs.append("**Relationships that stand out:** " + "; ".join(bits) + ". Correlation shows that variables "
                              "move together, not that one causes the other.")
        out.next_steps.append("Pick one variable of interest in **Pattern finder & optimiser** and I will search for an equation "
                              "that predicts it from the others.")
    return out


# --------------------------------------------------------------------------- the optimiser
def readout_optimization(res: OptimizationResult, comparison: Optional[OptimizationComparison] = None) -> Readout:
    """Explain an optimisation result in plain words."""
    out = Readout()
    t = _v(res.target_name)

    movers = [k for k in res.free if k not in res.inert]
    setting = ", ".join(f"{_v(k)} ≈ **{_n(res.values[k])}**" for k in movers)
    goal_txt = res.goal.describe(_v(res.target_name))
    if movers:
        s = f"To get {goal_txt}, I would set {setting}"
    else:
        s = f"There is nothing to adjust to get {goal_txt}"
    if res.locked:
        s += f", with {_join([f'{_v(k)} held at {_n(v)}' for k, v in res.locked.items()])}"
    if res.inert:
        s += f". {_join([_v(k) for k in res.inert])} barely changes the prediction, so set {'it' if len(res.inert) == 1 else 'them'} wherever is convenient"
    out.paragraphs.append(s + ".")

    p = (f"With those settings the equation predicts {t} ≈ **{_n(res.predicted)}**, give or take about {_n(res.rmse)} "
         f"(its typical held-out error), against {_n(res.typical_prediction)} at typical settings. ")
    mode = res.goal.mode
    if mode in ("maximize", "minimize"):
        change = res.improvement
        better = (change > 0) if mode == "maximize" else (change < 0)
        p += (f"That is {'an improvement' if better else 'no improvement'} of {_n(abs(change))}"
              f"{' (' + _pct(abs(change) / abs(res.typical_prediction)) + ' of the typical value)' if abs(res.typical_prediction) > 1e-9 else ''}.")
    elif res.reached:
        p += "According to the equation, that hits your goal."
    else:
        p += (f"The closest the equation can get inside your data's range misses the goal by {_n(res.gap)}, "
              f"so this goal may not be reachable with the allowed ranges.")
    out.paragraphs.append(p)

    if res.edge:
        edges = _join([f"{_v(k)} (at the {side} of your data, {_n(res.values[k])})" for k, side in res.edge.items()])
        out.paragraphs.append(f"**Heads-up:** {edges}. The equation would like to go further, but your data stop there, so I "
                              f"cannot say what happens beyond. Treat it as a direction to test, not a number to trust.")
        out.cautions.append("A recommendation pinned at the edge of the data is an extrapolation in spirit: test it before relying on it.")

    if res.nearest_value is not None:
        r = (f"**Reality check:** the closest real row in your data has {t} = {_n(res.nearest_value)}")
        if res.best_observed is not None:
            r += f", and the best you have actually observed is {_n(res.best_observed)}"
        if res.observed_in_range is not None:
            r += f"; {_pct(res.observed_in_range)} of your rows are already inside the range"
        r += "."
        if res.far_from_data:
            r += (" This exact combination of values has not really been seen: each number is within its own range, but "
                  "together they sit far from any real row, so the prediction is less certain than the error above suggests.")
            out.cautions.append("The recommended combination lies in a sparsely observed region; try a small pilot before committing.")
        out.paragraphs.append(r)

    if res.sensitivity:
        touchy = [k for k in res.touchy]
        calm = [s["variable"] for s in res.sensitivity if s["variable"] not in touchy and s["variable"] not in res.inert]
        parts = []
        if touchy:
            parts.append(f"{_join([_v(k) for k in touchy])} {'are' if len(touchy) > 1 else 'is'} **touchy**: a 10% slip of the "
                         f"allowed range moves {t} a lot, so set {'them' if len(touchy) > 1 else 'it'} carefully")
        if calm:
            parts.append(f"{_join([_v(k) for k in calm])} {'are' if len(calm) > 1 else 'is'} forgiving")
        if parts:
            out.paragraphs.append("**How precise you need to be:** " + "; ".join(parts) + ".")

    if comparison is not None and len(comparison.results) > 1:
        k = len(comparison.results)
        if comparison.agree:
            out.paragraphs.append(f"All **{k}** top expressions point to similar settings (they differ by at most "
                                  f"{_pct(comparison.disagreement)} of any variable's range), which makes me more comfortable.")
        else:
            diffs = [n for n, vals in comparison.per_variable.items()
                     if (max(vals) - min(vals)) / max(1e-12, res.bounds[n][1] - res.bounds[n][0]) >= 0.15]
            out.paragraphs.append(f"The top **{k}** expressions do **not** agree on the settings"
                                  f"{' for ' + _join([_v(x) for x in diffs]) if diffs else ''}; the data cannot pin the best "
                                  f"setting down, so treat the answer as a region rather than a point.")
            out.cautions.append("The candidate expressions disagree about the optimum: treat the answer as approximate.")

    out.cautions.append("This describes the fitted equation. Changing a variable in real life may not move the outcome the "
                        "way the data suggest (association, not cause and effect).")
    for n in res.notes:
        out.cautions.append(n)
    return out
