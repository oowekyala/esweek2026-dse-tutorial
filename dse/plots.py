"""The figures the notebooks draw, in one consistent style.

Costs are in milliseconds and span more than an order of magnitude in every
space here, so cost axes are logarithmic throughout. Categorical colours are
used in a fixed order; magnitude is always one blue ramp.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from .cinm import param_columns, visited

# Categorical slots, in the order they are assigned.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2dd"
# One-hue ramp for magnitude, light to dark.
BLUES = LinearSegmentedColormap.from_list("blues", ["#e7f0fb", "#9dc1ee", BLUE, "#123d73"])

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "font.size": 10,
    }
)


def _ms_log_axis(axis):
    """A log axis in milliseconds that reads 2, 3, 5, 10, 20 rather than 10^n."""
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _legend_below(ax, ncol: int = 2):
    """The legend under the plot, where it can never sit on a line."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol, fontsize=8)


def _log_bins(costs: np.ndarray, n: int = 40) -> np.ndarray:
    lo, hi = np.nanmin(costs), np.nanmax(costs)
    return np.logspace(np.log10(lo) - 0.02, np.log10(hi) + 0.02, n)


def _running_best(pool: pd.DataFrame) -> np.ndarray:
    """Best cost seen after each evaluation that produced one, in order.

    Evaluations that failed are left out entirely, so the x axis counts the
    evaluations that make up the search's budget (``max-evals`` does not count
    failures).
    """
    cost = visited(pool)["cost"].to_numpy(dtype=float)
    return np.minimum.accumulate(cost[~np.isnan(cost)])


def cost_histogram(
    oracle: pd.DataFrame,
    marks: Mapping[str, float] | None = None,
    title: str = "Cost of every feasible configuration",
    ax=None,
):
    """Distribution of simulated cost over the whole feasible set.

    ``marks`` are named costs drawn as vertical lines: the best, the median,
    the attendee's guess.
    """
    costs = oracle["cost"].dropna().to_numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(costs, bins=_log_bins(costs), color=BLUE, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    _ms_log_axis(ax.xaxis)
    ax.set_xlabel("estimated latency [ms], log scale")
    ax.set_ylabel("configurations")
    ax.set_title(title, loc="left", color=INK)
    if marks:
        # Marks close together on a log axis would overprint each other's
        # label, so the labels step down the plot in the order given.
        top = ax.get_ylim()[1]
        for i, ((label, x), colour) in enumerate(zip(marks.items(), SERIES[1:])):
            ax.axvline(x, color=colour, linewidth=1.6)
            ax.annotate(
                f"{label}: {x:.2f} ms",
                (x, top * (0.97 - 0.13 * i)),
                xytext=(4, 0),
                textcoords="offset points",
                color=colour,
                fontsize=9,
                va="top",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
            )
    return ax


def best_so_far(
    runs: Mapping[str, pd.DataFrame],
    oracle_best: float | None = None,
    baseline: np.ndarray | None = None,
    n_init: int | None = None,
    title: str = "Best latency found vs. simulations spent",
    ax=None,
):
    """Best cost seen after each simulation, one line per search.

    ``baseline`` is the array ``random_search_baseline`` returns: its median
    is drawn as the random-search reference, with a band for the middle 50 %.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.6))
    if baseline is not None:
        xs = np.arange(1, baseline.shape[1] + 1)
        lo, mid, hi = np.percentile(baseline, [25, 50, 75], axis=0)
        ax.fill_between(xs, lo, hi, color=INK_2, alpha=0.12, linewidth=0)
        ax.plot(xs, mid, color=INK_2, linewidth=1.4, linestyle="--", label="random search (median, IQR)")
    for (label, pool), colour in zip(runs.items(), SERIES):
        best = _running_best(pool)
        ax.plot(np.arange(1, len(best) + 1), best, color=colour, linewidth=2, label=label)
    if oracle_best is not None:
        ax.axhline(oracle_best, color=INK, linewidth=1, linestyle=":")
        ax.annotate("optimum (exhaustive)", (ax.get_xlim()[1], oracle_best), xytext=(-4, 4),
                    textcoords="offset points", ha="right", fontsize=8, color=INK)
    if n_init:
        ax.axvline(n_init + 0.5, color=GRID, linewidth=1.2)
        ax.annotate("surrogate takes over →", (n_init + 0.5, ax.get_ylim()[1]), xytext=(4, -12),
                    textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_yscale("log")
    _ms_log_axis(ax.yaxis)
    ax.set_xlabel("simulations")
    ax.set_ylabel("best latency so far [ms]")
    ax.set_title(title, loc="left", color=INK)
    _legend_below(ax, ncol=3)
    return ax


def seeds_spread(
    pools: Mapping[int, pd.DataFrame],
    oracle_best: float | None = None,
    baseline: np.ndarray | None = None,
    ax=None,
):
    """Best-so-far of several seeds of the same search: the spread is the point."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.6))
    if baseline is not None:
        xs = np.arange(1, baseline.shape[1] + 1)
        ax.plot(xs, np.median(baseline, axis=0), color=INK_2, linewidth=1.4, linestyle="--",
                label="random search (median)")
    curves = []
    for seed, pool in sorted(pools.items()):
        c = _running_best(pool)
        curves.append(c)
        ax.plot(np.arange(1, len(c) + 1), c, color=BLUE, linewidth=1, alpha=0.45)
    n = min(len(c) for c in curves)
    med = np.median(np.vstack([c[:n] for c in curves]), axis=0)
    ax.plot(np.arange(1, n + 1), med, color=BLUE, linewidth=2.4, label=f"BO, {len(curves)} seeds (median)")
    if oracle_best is not None:
        ax.axhline(oracle_best, color=INK, linewidth=1, linestyle=":", label="optimum (exhaustive)")
    ax.set_yscale("log")
    _ms_log_axis(ax.yaxis)
    ax.set_xlabel("simulations")
    ax.set_ylabel("best latency so far [ms]")
    ax.set_title("The same search, different random seeds", loc="left", color=INK)
    _legend_below(ax, ncol=3)
    return ax


def surrogate_check(pool: pd.DataFrame, ax=None):
    """Predicted vs. simulated cost for the points the surrogate chose."""
    v = visited(pool)
    v = v[v["mu"].notna() & (v["mu"] != 0)]
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4))
    ax.errorbar(v["cost"], v["mu"], yerr=v["sigma"], fmt="o", color=BLUE, ecolor="#9dc1ee",
                markersize=5, elinewidth=1, capsize=0)
    lo = min(v["cost"].min(), v["mu"].min()) * 0.8
    hi = max(v["cost"].max(), v["mu"].max()) * 1.2
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _ms_log_axis(ax.xaxis)
    _ms_log_axis(ax.yaxis)
    ax.set_xlabel("simulated latency [ms]")
    ax.set_ylabel("surrogate prediction µ ± σ")
    ax.set_title("What the surrogate believed when it chose", loc="left", fontsize=10, color=INK)
    return ax
