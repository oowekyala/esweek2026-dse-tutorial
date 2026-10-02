#!/usr/bin/env python3
"""Draws the figures the slides include, from the data in ../../data.

The notebooks draw the same kinds of figure from whatever their live runs
produce; the slides show runs made beforehand -- the oracles and the seed
sweeps under data/sweeps -- so that what is on the board does not depend on
which seed a Binder kernel happened to draw. Run from anywhere:

    python slides/figures/_make.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import dse  # noqa: E402
import dse.plots as plots  # noqa: E402
from dse import DATA  # noqa: E402


def sweep(name: str) -> dict[int, "dse.pd.DataFrame"]:
    out = {}
    for f in sorted((DATA / "sweeps" / name).glob("seed_*/pool.csv")):
        out[int(f.parent.name.split("_")[1])] = dse.load_pool(f)
    return out


def save(fig_or_ax, name: str):
    fig = fig_or_ax if isinstance(fig_or_ax, plt.Figure) else fig_or_ax.figure
    fig.savefig(HERE / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def main():
    small = dse.load_pool(DATA / "small" / "pool.csv")
    big = dse.load_pool(DATA / "big" / "pool.csv")

    save(
        plots.cost_histogram(
            small, {"best": small["cost"].min(), "median": small["cost"].median()}
        ),
        "histogram.pdf",
    )
    base_small = dse.random_search_baseline(small, budget=100)
    ax = plots.seeds_spread(sweep("small_40x10"), oracle_best=small["cost"].min(), baseline=base_small[:, :40])
    ax.set_title("3234 configurations, budget 40 — ten seeds", loc="left")
    save(ax, "seeds_small_40.pdf")

    ax = plots.seeds_spread(sweep("small_100x5"), oracle_best=small["cost"].min(), baseline=base_small)
    ax.set_title("3234 configurations, budget 100 — five seeds", loc="left")
    save(ax, "seeds_small_100.pdf")

    base_big = dse.random_search_baseline(big, budget=100)
    ax = plots.seeds_spread(sweep("big_100x5"), oracle_best=big["cost"].min(), baseline=base_big)
    ax.set_title("26,238 configurations, budget 100 — five seeds", loc="left")
    save(ax, "seeds_big_100.pdf")


if __name__ == "__main__":
    main()
