"""Matplotlib charts in the LAWSHIFT visual style: warm paper, sage greens, one terracotta highlight.

The colours come from `src.theme` (60-30-10: paper neutrals, sage data colours, a rare terracotta accent
reserved for whatever deserves the eye - held-out points, the optimum, flagged observations).

Every function takes real computed arrays/records and returns a Figure. Nothing here
generates or fabricates data. Callers are responsible for closing figures.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from .. import theme

BG = theme.CREAM
PANEL = theme.PAPER
GRID = theme.LINEN
TEXT = theme.INK
MUTED = theme.MUTED
SAGE = theme.SAGE
SAGE_DEEP = theme.SAGE_DEEP
SAGE_MID = theme.SAGE_MID
SAGE_SOFT = theme.SAGE_SOFT
TAUPE = theme.TAUPE
ACCENT = theme.TERRACOTTA
RANK_COLORS = [SAGE, SAGE_MID, TAUPE, SAGE_DEEP]
DIVERGING = LinearSegmentedColormap.from_list("lawshift_div", [ACCENT, PANEL, SAGE_DEEP])


def _line(ax, x, y, color, lw: float = 2.0, **kw):
    (line,) = ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", **kw)
    return line


def _dots(ax, x, y, color, s: float = 16, alpha: float = 0.8, **kw):
    return ax.scatter(x, y, s=s, color=color, alpha=alpha, edgecolors="none", **kw)


def _fig(w: float = 6.4, h: float = 4.0, ncols: int = 1, nrows: int = 1):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "font.sans-serif": ["Inter", "DejaVu Sans", "Arial", "sans-serif"]})
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), facecolor=BG, squeeze=False)
    for ax in axes.ravel():
        _style(ax)
    return fig, axes


def _style(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for name, sp in ax.spines.items():
        sp.set_color(GRID)
        sp.set_visible(name in ("left", "bottom"))
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(theme.SAGE_DEEP)


def _legend(ax, **kw) -> None:
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8, framealpha=0.95, **kw)
    if leg:
        leg.get_frame().set_linewidth(0.8)


def raw_observations(
    input_names: Sequence[str], X_train, y_train, output_name: str,
    X_val=None, y_val=None, units: Optional[Dict[str, str]] = None,
) -> Figure:
    """One scatter panel per input variable: observed output vs that input."""
    X_train = np.asarray(X_train)
    n = X_train.shape[1]
    fig, axes = _fig(5.2, 3.8, ncols=n)
    u = units or {}
    for j, ax in enumerate(axes.ravel()):
        _dots(ax, X_train[:, j], y_train, SAGE, s=16, alpha=0.7, label="training")
        if X_val is not None and y_val is not None:
            _dots(ax, np.asarray(X_val)[:, j], y_val, ACCENT, s=18, alpha=0.85, label="validation (held out)")
        nm = input_names[j]
        ax.set_xlabel(f"{nm} [{u[nm]}]" if nm in u else nm)
        ax.set_ylabel(f"{output_name} [{u[output_name]}]" if output_name in u else output_name)
        ax.set_title(f"{output_name} vs {nm}", fontsize=10)
        if j == 0:
            _legend(ax)
    fig.tight_layout()
    return fig


def actual_vs_predicted(y_true, y_pred, title: str = "Actual vs predicted (held-out data)") -> Figure:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    fig, axes = _fig(5.2, 4.2)
    ax = axes[0, 0]
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1.3, ls="--", label="perfect prediction")
    _dots(ax, y_true, y_pred, SAGE, s=22, alpha=0.8, label="observations")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title, fontsize=10)
    _legend(ax)
    fig.tight_layout()
    return fig


def residuals(y_true, y_pred, title: str = "Residuals (actual - predicted)") -> Figure:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    res = y_true - y_pred
    fig, axes = _fig(4.6, 4.0, ncols=2)
    a1, a2 = axes[0, 0], axes[0, 1]
    a1.axhline(0, color=MUTED, lw=1.2, ls="--")
    _dots(a1, y_pred, res, SAGE, s=18, alpha=0.8)
    a1.set_xlabel("Predicted")
    a1.set_ylabel("Residual")
    a1.set_title(title, fontsize=10)
    a2.hist(res, bins=min(20, max(5, len(res) // 4)), color=SAGE_MID, edgecolor=PANEL, alpha=0.95)
    a2.axvline(0, color=MUTED, lw=1.2, ls="--")
    a2.set_xlabel("Residual")
    a2.set_ylabel("Count")
    a2.set_title(f"Mean {res.mean():.3g}, std {res.std():.3g}", fontsize=10)
    fig.tight_layout()
    return fig


def fitness_history(history: List[Dict]) -> Figure:
    """Best and mean fitness per generation plus the best candidate's complexity."""
    if not history:
        raise ValueError("history is empty")
    g = [h["generation"] for h in history]
    best = np.array([h["best_fitness"] for h in history], float)
    mean = np.array([np.nan if h.get("mean_fitness") is None else h["mean_fitness"] for h in history], float)
    sizes = [h["best_size"] for h in history]
    fig, axes = _fig(6.4, 4.0)
    ax = axes[0, 0]
    _line(ax, g, best, SAGE_DEEP, 2.2, label="best fitness")
    ax.plot(g, mean, color=TAUPE, lw=1.4, ls="--", label="population mean")
    pos = np.concatenate([best[best > 0], mean[np.isfinite(mean) & (mean > 0)]])
    if pos.size and pos.max() / pos.min() > 20:
        ax.set_yscale("log")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (lower = better)")
    ax.set_title("Evolution of the best candidate", fontsize=10)
    ax2 = ax.twinx()
    ax2.step(g, sizes, color=ACCENT, lw=1.2, where="post", alpha=0.9, label="best complexity (nodes)")
    ax2.set_ylabel("Complexity (nodes)", color=ACCENT)
    ax2.tick_params(colors=ACCENT, labelsize=9, length=0)
    ax2.grid(False)
    for name, sp in ax2.spines.items():
        sp.set_color(GRID)
        sp.set_visible(name == "right")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax.legend(h1 + h2, l1 + l2, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    return fig


def run_comparison(labels: Sequence[str], val_rmse: Sequence[float], sizes: Sequence[int],
                   best_index: Optional[int] = None) -> Figure:
    """Compare independent searches on held-out RMSE and complexity; the selected search is highlighted."""
    fig, axes = _fig(4.6, 3.8, ncols=2)
    a1, a2 = axes[0, 0], axes[0, 1]
    colors = [ACCENT if i == best_index else SAGE_MID for i in range(len(labels))]
    a1.bar(labels, val_rmse, color=colors, edgecolor=PANEL)
    a1.set_ylabel("Held-out RMSE")
    a1.set_title("Prediction error by search", fontsize=10)
    a2.bar(labels, sizes, color=colors, edgecolor=PANEL)
    a2.set_ylabel("Complexity (nodes)")
    a2.set_title("Equation complexity by search", fontsize=10)
    fig.tight_layout()
    return fig


def curve_overlay(
    x_train, y_train, x_val, y_val, predict: Callable[[np.ndarray], np.ndarray],
    xlabel: str, ylabel: str, true_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Figure:
    """1-D systems: data, the discovered curve and (optionally) the hidden true curve."""
    x_train, x_val = np.ravel(x_train), np.ravel(x_val)
    lo, hi = float(min(x_train.min(), x_val.min())), float(max(x_train.max(), x_val.max()))
    grid = np.linspace(lo, hi, 400).reshape(-1, 1)
    fig, axes = _fig(7.0, 4.2)
    ax = axes[0, 0]
    _dots(ax, x_train, y_train, TAUPE, s=14, alpha=0.6, label="training")
    _dots(ax, x_val, y_val, SAGE_MID, s=18, alpha=0.9, label="validation")
    _line(ax, grid[:, 0], predict(grid), SAGE_DEEP, 2.2, label="discovered equation")
    if true_fn is not None:
        ax.plot(grid[:, 0], true_fn(grid), color=ACCENT, lw=1.6, ls="--", label="hidden true law")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Discovered curve", fontsize=10)
    _legend(ax)
    fig.tight_layout()
    return fig


def correlation_heatmap(corr) -> Figure:
    """corr: pandas DataFrame correlation matrix."""
    labels = list(corr.columns)
    n = len(labels)
    fig, axes = _fig(max(5.0, 0.9 * n + 2), max(4.2, 0.8 * n + 1.5))
    ax = axes[0, 0]
    ax.grid(False)
    im = ax.imshow(corr.to_numpy(float), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.7 else TEXT)
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    cb.outline.set_edgecolor(GRID)
    ax.set_title("Pearson correlation (association, not causation)", fontsize=10)
    fig.tight_layout()
    return fig


def anomaly_plot(y_actual, y_pred, flagged_mask, target: str) -> Figure:
    y_actual, y_pred = np.asarray(y_actual, float), np.asarray(y_pred, float)
    mask = np.asarray(flagged_mask, bool)
    fig, axes = _fig(5.6, 4.2)
    ax = axes[0, 0]
    lo, hi = float(min(y_actual.min(), y_pred.min())), float(max(y_actual.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls="--")
    _dots(ax, y_pred[~mask], y_actual[~mask], SAGE, s=16, alpha=0.7, label="typical")
    if mask.any():
        ax.scatter(y_pred[mask], y_actual[mask], s=52, color=ACCENT, edgecolors=PANEL, label=f"unusual ({int(mask.sum())})")
    ax.set_xlabel(f"Predicted {target} (linear model)")
    ax.set_ylabel(f"Actual {target}")
    ax.set_title("Unusual observations", fontsize=10)
    _legend(ax)
    fig.tight_layout()
    return fig


def model_comparison(models: Sequence[str], val_rmse: Sequence[float], val_r2: Sequence[float]) -> Figure:
    fig, axes = _fig(4.6, 3.8, ncols=2)
    a1, a2 = axes[0, 0], axes[0, 1]
    cols = [SAGE_DEEP if i == int(np.argmin(val_rmse)) else SAGE_MID for i in range(len(models))]
    a1.bar(models, val_rmse, color=cols, edgecolor=PANEL)
    a1.set_ylabel("Held-out RMSE (lower = better)")
    a1.set_title("Prediction error", fontsize=10)
    a2.bar(models, val_r2, color=cols, edgecolor=PANEL)
    a2.set_ylabel("Held-out R²")
    a2.set_title("Variance explained", fontsize=10)
    for a in (a1, a2):
        a.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    return fig


def history_scatter(records: List[Dict]) -> Figure:
    """Held-out R² of every stored physics experiment against its noise level."""
    phys = [r for r in records if r.get("kind") == "physics" and r.get("validation", {}).get("r2") is not None]
    fig, axes = _fig(6.0, 3.8)
    ax = axes[0, 0]
    palette = {"newtonian": SAGE_DEEP, "harmonic": ACCENT, "projectile": TAUPE}
    for key in sorted({r["system"] for r in phys}):
        pts = [r for r in phys if r["system"] == key]
        ax.scatter([100 * r["noise_level"] for r in pts], [r["validation"]["r2"] for r in pts],
                   s=50, color=palette.get(key, MUTED), edgecolors=PANEL, label=key)
    ax.set_xlabel("Measurement noise (% of signal std)")
    ax.set_ylabel("Held-out R²")
    ax.set_title("Stored experiments: accuracy vs noise", fontsize=10)
    if phys:
        _legend(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- insights & candidates
def variable_importance(names: Sequence[str], shares: Sequence[float]) -> Figure:
    """Horizontal bars: each variable's share of the equation's total sensitivity (held-out permutation test)."""
    order = np.argsort(shares)
    fig, axes = _fig(5.6, max(2.2, 0.55 * len(names) + 1.2))
    ax = axes[0, 0]
    vals = np.asarray(shares, float)[order] * 100
    labs = [names[i] for i in order]
    colors = [SAGE_DEEP if i == len(vals) - 1 else SAGE for i in range(len(vals))]
    ax.barh(labs, vals, color=colors, height=0.55, edgecolor=PANEL)
    for y_, v in enumerate(vals):
        ax.text(v + 1, y_, f"{v:.0f}%", va="center", color=TEXT, fontsize=9)
    ax.set_xlim(0, max(100, vals.max() + 12) if len(vals) else 100)
    ax.set_xlabel("Share of influence on predictions (%)")
    ax.set_title("Which variables drive the equation", fontsize=10)
    ax.grid(False, axis="y")
    fig.tight_layout()
    return fig


def partial_effects(variables, target: str) -> Figure:
    """One panel per variable: predicted target vs that variable, others held at their medians."""
    n = len(variables)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = _fig(4.2, 3.2, ncols=cols, nrows=rows)
    flat = axes.ravel()
    lo = min(float(np.min(v.curve)) for v in variables)
    hi = max(float(np.max(v.curve)) for v in variables)
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    for ax, v in zip(flat, variables):
        ax.set_ylim(lo - pad, hi + pad)          # one shared scale: a flat effect must look flat
        col = SAGE_DEEP if v.importance >= 0.05 else TAUPE
        ax.axvspan(v.low, v.high, color=SAGE_SOFT, alpha=0.9, lw=0)
        _line(ax, v.grid, v.curve, col, 2.0)
        if v.turning_x is not None:
            ax.axvline(v.turning_x, color=ACCENT, lw=1.1, ls=":")
        ax.set_xlabel(v.name)
        ax.set_ylabel(f"predicted {target}")
        ax.set_title(f"{v.name}: {v.direction}" + (f", {v.shape}" if v.shape else ""), fontsize=9)
    for ax in flat[n:]:
        ax.axis("off")
    fig.suptitle("Effect of each variable (others held at median; shaded = 25th-75th percentile)",
                 color=MUTED, fontsize=9, y=1.0)
    fig.tight_layout()
    return fig


def candidates_comparison(labels: Sequence[str], val_r2: Sequence[float], sizes: Sequence[int]) -> Figure:
    """Top expressions side by side: held-out accuracy against complexity (the accuracy/simplicity trade-off)."""
    fig, axes = _fig(4.8, 3.6, ncols=2)
    a1, a2 = axes[0, 0], axes[0, 1]
    cols = [RANK_COLORS[i % len(RANK_COLORS)] for i in range(len(labels))]
    a1.bar(labels, val_r2, color=cols, edgecolor=PANEL)
    a1.set_ylabel("Held-out R²")
    a1.set_title("Accuracy on unseen data", fontsize=10)
    a1.set_ylim(min(0.0, min(val_r2) - 0.05), 1.02)
    a2.bar(labels, sizes, color=cols, edgecolor=PANEL)
    a2.set_ylabel("Complexity (nodes)")
    a2.set_title("Equation size", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- optimiser
def optimizer_curves(res) -> Figure:
    """One panel per free variable: the equation's response with the other variables at their recommended values.

    Terracotta marks the recommended value; the hollow marker is the typical value; the dashed line / band is the goal.
    """
    names = [k for k in res.free]
    if not names:
        raise ValueError("Nothing to plot: no free variables.")
    cols = min(3, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = _fig(4.3, 3.2, ncols=cols, nrows=rows)
    flat = axes.ravel()
    all_y = np.concatenate([res.curves[k][1] for k in names])
    g = res.goal
    extra = [res.predicted]
    if g.mode == "target":
        extra.append(g.target)
    if g.mode == "range":
        extra += [g.low, g.high]
    lo, hi = float(min(all_y.min(), *extra)), float(max(all_y.max(), *extra))
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    for ax, k in zip(flat, names):
        grid, curve = res.curves[k]
        ax.set_ylim(lo - pad, hi + pad)
        if g.mode == "range":
            ax.axhspan(g.low, g.high, color=SAGE_SOFT, alpha=0.9, lw=0)
        elif g.mode == "target":
            ax.axhline(g.target, color=MUTED, lw=1.1, ls="--")
        _line(ax, grid, curve, SAGE_DEEP if k not in res.inert else TAUPE, 2.0)
        ax.scatter([res.typical[k]], [float(np.interp(res.typical[k], grid, curve))], s=46, facecolors=PANEL,
                   edgecolors=MUTED, linewidths=1.4, zorder=4, label="typical")
        ax.scatter([res.values[k]], [res.predicted], s=70, color=ACCENT, edgecolors=PANEL, zorder=5, label="recommended")
        ax.set_xlabel(k)
        ax.set_ylabel(f"predicted {res.target_name}")
        ax.set_title(f"{k} ≈ {res.values[k]:.3g}", fontsize=9.5)
        if k == names[0]:
            _legend(ax, loc="best")
    for ax in flat[len(names):]:
        ax.axis("off")
    fig.suptitle("How the equation responds around the recommended settings (others held at their recommended values)",
                 color=MUTED, fontsize=9, y=1.0)
    fig.tight_layout()
    return fig


def optimum_agreement(per_variable: Dict[str, List[float]], bounds: Dict[str, tuple]) -> Figure:
    """Where each expression places its recommended value inside the allowed range (0 = lowest, 1 = highest)."""
    names = list(per_variable)
    if not names:
        raise ValueError("Nothing to plot: no free variables.")
    fig, axes = _fig(5.6, max(2.0, 0.55 * len(names) + 1.1))
    ax = axes[0, 0]
    for y_, k in enumerate(names):
        lo, hi = bounds[k]
        pos = [(v - lo) / (hi - lo) if hi > lo else 0.5 for v in per_variable[k]]
        ax.plot([min(pos), max(pos)], [y_, y_], color=GRID, lw=5, solid_capstyle="round", zorder=1)
        for i, p_ in enumerate(pos):
            ax.scatter([p_], [y_], s=70, color=RANK_COLORS[i % len(RANK_COLORS)], edgecolors=PANEL, zorder=3,
                       label=f"expression #{i + 1}" if y_ == 0 else None)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Position inside the allowed range (0 = lowest, 1 = highest)")
    ax.set_title("Do the top expressions agree on the best settings?", fontsize=10)
    ax.grid(False, axis="y")
    _legend(ax, loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- ordinary data
def distribution_panel(values, name: str, mean: float, std: float) -> Figure:
    """Histogram with the fitted normal curve (left) and a normal Q-Q plot (right)."""
    from scipy import stats

    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    fig, axes = _fig(4.8, 3.8, ncols=2)
    a1, a2 = axes[0, 0], axes[0, 1]
    a1.hist(x, bins=min(30, max(8, int(np.sqrt(len(x))) + 2)), density=True, color=SAGE_MID, edgecolor=PANEL, alpha=0.95,
            label="your data")
    if std > 0:
        grid = np.linspace(x.min() - 0.5 * std, x.max() + 0.5 * std, 300)
        _line(a1, grid, stats.norm.pdf(grid, mean, std), ACCENT, 2.0, label="normal curve with same mean and spread")
    a1.axvline(mean, color=SAGE_DEEP, lw=1.2, ls="--")
    a1.set_xlabel(name)
    a1.set_ylabel("Density")
    a1.set_title("Shape of the data", fontsize=10)
    _legend(a1)
    (osm, osr), (slope, icpt, _r) = stats.probplot(x, dist="norm")
    _dots(a2, osm, osr, SAGE, s=16, alpha=0.75)
    xs = np.array([osm.min(), osm.max()])
    a2.plot(xs, slope * xs + icpt, color=ACCENT, lw=1.6)
    a2.set_xlabel("Expected value if normal (standard deviations)")
    a2.set_ylabel(f"Observed {name}")
    a2.set_title("Q-Q plot: points on the line = normal", fontsize=10)
    fig.tight_layout()
    return fig


def distribution_overview(df, labels: Optional[Dict[str, str]] = None, max_cols: int = 12) -> Figure:
    """Small-multiple histograms for every numeric column."""
    cols = list(df.columns)[:max_cols]
    n = len(cols)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = _fig(3.0, 2.4, ncols=ncols, nrows=nrows)
    flat = axes.ravel()
    for ax, c in zip(flat, cols):
        x = df[c].to_numpy(float)
        ax.hist(x, bins=min(24, max(6, int(np.sqrt(len(x))))), color=SAGE_MID, edgecolor=PANEL)
        ax.set_title(c, fontsize=9)
        ax.set_yticks([])
        if labels and c in labels:
            ax.set_xlabel(labels[c], fontsize=8, color=MUTED)
    for ax in flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def pair_scatter(x, y, xname: str, yname: str, r: float) -> Figure:
    """Two variables with a least-squares line, for the relationships view."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    fig, axes = _fig(5.2, 3.8)
    ax = axes[0, 0]
    _dots(ax, x, y, SAGE, s=20, alpha=0.65)
    if np.std(x) > 0:
        slope, icpt = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + icpt, color=ACCENT, lw=1.8)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_title(f"{yname} vs {xname} (r = {r:+.2f})", fontsize=10)
    fig.tight_layout()
    return fig
