#!/usr/bin/env python3
"""Writes the three tutorial notebooks.

Kept as a script so the notebooks can be regenerated after editing the prose or
the cells here in one place; the .ipynb files are what attendees open.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

HEADER = """\
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dse
import dse.plots as plots
from dse import DATA, WORK

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)
"""


def nb(cells: list[tuple[str, str]], title: str) -> nbf.NotebookNode:
    out = nbf.v4.new_notebook()
    out.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out.metadata["language_info"] = {"name": "python"}
    for kind, src in cells:
        src = src.rstrip("\n")
        out.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Notebook 1 — the design space
# ═══════════════════════════════════════════════════════════════════════════

NB1 = [
    ("md", """\
# 1 · The design space

A matrix–vector product on a compute-near-memory system: one rank of **64 DPUs**,
each with up to **16 tasklets** (hardware threads), each DPU owning a slice of
memory (MRAM) and a small working memory (WRAM).

To run `y = A·x` there, the loop nest has to be **tiled** so that every DPU gets a
piece of `A` that fits its MRAM, every tasklet a piece that fits WRAM, and the
pieces are laid out so the result can be gathered back. Which tile sizes, how
many tasklets, in which order the dimensions are distributed — those are the
*design parameters*. This notebook is about how many there are, and how few
combinations of them are even legal.
"""),
    ("code", HEADER),
    ("md", """\
## The program

Cinnamon's input is MLIR. The kernel is a single `cinm.op.gemv` on a
1024 × 1024 matrix; the `#upmem.platform` attribute says what hardware is
available.
"""),
    ("code", """\
mlir = DATA / "small" / "gemv.mlir"
dse.show_mlir(mlir)
"""),
    ("md", """\
## Your guess

Before the tools tell us anything: pick a configuration by hand.

The picker only offers choices that can still complete a **feasible**
configuration — as you fix parameters, the other dropdowns shrink to what remains
legal. Fix all of them, then press **Simulate**: `eval-solution` runs the whole
lowering for that one configuration and reports what the simulator estimates —
a fraction of a second.

- `tasklets` — threads per DPU
- `gemv.M.mram`, `gemv.K.mram` — the tile of `A` each *(DPU, tasklet)* worker
  holds in its MRAM, along the two dimensions of the loop nest
  (M = rows of A / entries of y, K = columns of A / entries of x)
- `gemv.M.wram`, `gemv.K.wram` — the inner tile a tasklet streams through WRAM
- `gemv.order` — which dimension is spread across DPUs first
"""),
    ("code", """\
import json

space = dse.load_space(DATA / "small" / "space.json")
feasible = dse.load_pool(DATA / "small" / "pool.csv")[lambda p: dse.param_columns(p)]


def simulate(config):
    report = dse.eval_solution(mlir, config)
    # Notebook 2 reads this back to show where the guess landed.
    (WORK / "my_guess.json").write_text(
        json.dumps({"config": config, "cost_ms": report.total_ms}, indent=1)
    )
    return report


picker = dse.ConfigPicker(feasible, space, on_pick=simulate).show()
"""),
    ("md", """\
The breakdown says where the time goes: the kernel itself, launching it,
moving the operands in and the result back. Keep the total in mind.

Skipped the picker? The cell below falls back to a plausible first guess —
all 16 tasklets, middling tiles — so the rest of the tutorial has something to
compare against.
"""),
    ("code", """\
if picker.result is None:
    report = simulate(picker.default_guess())
    print("no configuration simulated in the widget; used a default guess:", report.config)
else:
    report = picker.result
report
"""),
    ("md", """\
In the next notebook you will see where that number lands among *all*
feasible configurations.

## Where the space comes from

The picker knew which combinations were legal. That knowledge is derived, not
listed: from the loop nest of the operation and the memory hierarchy of the
platform, Cinnamon builds a **constraint system** and solves it. `dump-space-only`
does exactly that and stops.
"""),
    ("code", """\
space = dse.dump_space(mlir)
print(dse.cinm.describe_space(space))
"""),
    ("md", """\
Four billion combinations of parameter values, of which about three thousand
are configurations that can run at all. The parameters:
"""),
    ("code", """\
dse.cinm.params_table(space)
"""),
    ("md", """\
And the constraints that cut the Cartesian product down. Some are pure
arithmetic on the problem shape and were applied as filters; the rest went to
the solver. Read them as the hardware talking: *a tile has to divide the
dimension*, *what the tasklets hold has to fit MRAM*, *a DMA moves 8 bytes at a
time*, *there is exactly one tile per worker*.
"""),
    ("code", """\
pd.set_option("display.max_colwidth", 120)
dse.cinm.constraints_table(space)
"""),
    ("md", """\
**What to take away.** The design space is not something we wrote down; it
falls out of the program's structure and the hardware's structure. The same
derivation, with a different platform description, gives the space of a
different CNM system.

→ Notebook 2: what the 3234 configurations look like.
"""),
]

# ═══════════════════════════════════════════════════════════════════════════
# Notebook 2 — the landscape
# ═══════════════════════════════════════════════════════════════════════════

NB2 = [
    ("md", """\
# 2 · The landscape

We simulated **every** feasible configuration of the small space ahead of time
— 3234 runs of the simulator, a little over three minutes on one core. That
oracle is loaded here. With it, three questions have exact answers:

1. How is cost distributed — and where did *your* guess land?
2. Which parameters matter?
3. What do the good configurations have in common?
"""),
    ("code", HEADER),
    ("code", """\
import json

mlir = DATA / "small" / "gemv.mlir"
oracle = dse.load_pool(DATA / "small" / "pool.csv")
space = dse.load_space(DATA / "small" / "space.json")

best = oracle["cost"].min()
median = oracle["cost"].median()

guess_file = WORK / "my_guess.json"
if guess_file.exists():
    my = json.loads(guess_file.read_text())
    my_cost = my["cost_ms"]
else:
    # No guess from notebook 1: take the median configuration as a stand-in.
    my_cost = median
    print("no guess saved by notebook 1; using the median as yours")

print(f"{len(oracle)} feasible configurations")
print(f"best   {best:7.2f} ms")
print(f"median {median:7.2f} ms")
print(f"yours  {my_cost:7.2f} ms  →  rank {dse.rank_of(my_cost, oracle)} of {len(oracle)}, "
      f"better than {100 - dse.percentile_of(my_cost, oracle):.0f} % of the space")
"""),
    ("md", """\
## How cost is distributed
"""),
    ("code", """\
ax = plots.cost_histogram(oracle, {"best": best, "median": median, "your guess": my_cost})
plt.show()

for f in (1.1, 1.5, 2, 3):
    print(f"within {f}× of the best: {(oracle['cost'] <= f * best).sum():4d} configurations")
"""),
    ("md", """\
The bulk sits between 8 and 12 ms. The best configurations are **five times**
faster than the median and there are only a handful of them: a needle, not a
hill. Any search that samples uniformly is unlikely to land on one.

## Which parameters matter

For each parameter, the best and the median cost at each of its values. A
parameter whose *best* line moves a lot is one the search has to get right.
"""),
    ("code", """\
fig = plots.marginals(oracle, ["tasklets", "gemv.M.mram", "gemv.K.mram",
                               "gemv.M.wram", "gemv.K.wram", "gemv.order[0]"])
plt.show()
"""),
    ("md", """\
Two things to notice:

- **`tasklets`** — the best configuration uses 8 of the 16 threads. Using all
  of them is *worse* than using half: the tiles each tasklet can hold shrink,
  and the per-launch overhead grows.
- **`gemv.K.mram`** — only the full extent, 1024, reaches the fast region. That
  is the reduction dimension: splitting it across workers means partial sums
  that have to be combined afterwards.

The WRAM tiles, by contrast, barely move the best line: once the MRAM tiles are
right, the inner tiling is forgiving.
"""),
    ("code", """\
plots.heatmap(oracle, x="gemv.K.mram", y="tasklets")
plt.show()
"""),
    ("md", """\
## What the good configurations have in common
"""),
    ("code", """\
cols = dse.param_columns(oracle) + ["cost"]
oracle.sort_values("cost").head(10)[cols].style.format({"cost": "{:.3f}"}).hide(axis="index")
"""),
    ("md", """\
The ten best configurations differ in **one** parameter. Everything else —
tasklets, both MRAM tiles, the loop order — is the same. The optimum is not a
point but a short line through a seven-dimensional space, and the whole line
is the same *shape*: distribute M finely across the workers, keep K whole.

## Pinning a hardware knob — live

The hardware parameters and the tiling parameters are not independent. Pin the
number of tasklets and exhaustively search what is left; it is a fifth of the
space and takes about half a minute on one core.
"""),
    ("code", """\
r = dse.exhaustive(mlir, "exhaustive_t8", fixed_tasklets=8)
sub = r.pool()

print(f"{len(sub)} configurations with tasklets = 8, simulated in {r.seconds:.0f} s")
print(f"best in the subspace: {sub['cost'].min():.3f} ms   (whole space: {best:.3f} ms)")
"""),
    ("code", """\
# The same comparison for tasklets = 16, read off the oracle rather than re-run.
for t in (1, 4, 8, 16):
    part = oracle[oracle["tasklets"] == t]
    print(f"tasklets = {t:2d}: {len(part):4d} configurations, best {part['cost'].min():6.2f} ms")
"""),
    ("md", """\
Fixing a hardware knob to the wrong value caps what any tiling can reach —
with 16 tasklets the best possible configuration is five times slower than
with 8. Whoever decides the hardware parameters has to see the tiling space,
and the other way round.

→ Notebook 3: finding the needle without simulating the haystack.
"""),
]

# ═══════════════════════════════════════════════════════════════════════════
# Notebook 3 — the search
# ═══════════════════════════════════════════════════════════════════════════

NB3 = [
    ("md", """\
# 3 · The search

The small space could be enumerated: 3234 simulations. A realistic one cannot.
The question of this notebook is how close a **budget** of simulations gets to
the optimum — and, because we have the oracle, we can actually measure it.

## Bayesian optimisation in one paragraph

Spend a few simulations on a spread-out initial sample. Fit a **surrogate** to
what you have seen: a model that, for any configuration, predicts a cost µ and
how unsure it is, σ (in the literature these are often Gaussian processes;
here it is an ensemble of small networks). Pick the next configuration by an
**acquisition** rule that trades a low µ against a high σ — exploit what looks
good, explore what is unknown. Simulate it, refit, repeat until the budget is
spent.
"""),
    ("code", HEADER),
    ("code", """\
mlir = DATA / "small" / "gemv.mlir"
oracle = dse.load_pool(DATA / "small" / "pool.csv")
best = oracle["cost"].min()

# Random search needs no simulator once the oracle exists; 2000 trials of it
# give the reference every search below is measured against.
baseline = dse.random_search_baseline(oracle, budget=100)
"""),
    ("md", """\
## Forty simulations
"""),
    ("code", """\
r40 = dse.bo(mlir, "bo40", max_evals=40, n_init=10)
pool40 = r40.pool()
v = dse.visited(pool40)

print(f"{len(v)} simulations in {r40.seconds:.1f} s")
print(f"best found {v['cost'].min():.3f} ms — rank {dse.rank_of(v['cost'].min(), oracle)} of {len(oracle)}")
"""),
    ("md", """\
Every row the search visited, in order. The first ten are the initial sample —
no surrogate yet. From then on each row carries what the surrogate believed
about it when it was chosen: its prediction `mu`, its uncertainty `sigma`, and
the acquisition value `acq` that made it the pick.
"""),
    ("code", """\
cols = dse.param_columns(pool40) + ["cost", "mu", "sigma", "acq"]
v[cols].style.format({"cost": "{:.2f}", "mu": "{:.2f}", "sigma": "{:.2f}", "acq": "{:.2f}"}).hide(axis="index")
"""),
    ("code", """\
plots.surrogate_check(pool40)
plt.show()
"""),
    ("md", """\
Points on the dotted line were predicted right; error bars show how sure the
surrogate was. Early picks are far off with wide bars — it was exploring. The
later ones cluster near the line, low: it had learned where the fast region is.
"""),
    ("code", """\
ax = plots.best_so_far({"Bayesian, 40 evals": pool40}, oracle_best=best,
                       baseline=baseline[:, :40], n_init=10)
plt.show()
"""),
    ("md", """\
## Is that luck?

One run tells you one thing. Run the same search from five random seeds.
"""),
    ("code", """\
r5 = dse.bo(mlir, "bo40x5", max_evals=40, n_init=10, seeds=5)
pools5 = r5.pools()

print(f"5 × 40 simulations in {r5.seconds:.0f} s")
for seed, p in sorted(pools5.items()):
    m = dse.visited(p)["cost"].min()
    print(f"seed {seed:4d}: best {m:6.3f} ms  rank {dse.rank_of(m, oracle):4d}  ({(m / best - 1) * 100:4.0f} % above optimum)")
"""),
    ("code", """\
ax = plots.seeds_spread(pools5, oracle_best=best, baseline=baseline[:, :40])
plt.show()
"""),
    ("md", """\
With a target that is 0.3 % of the space, forty simulations is a coin flip:
some seeds find the fast line, some never leave the bulk. That spread *is* a
result — a single run of a stochastic search is an anecdote.

## More budget
"""),
    ("code", """\
r100 = dse.bo(mlir, "bo100", max_evals=100, n_init=20)
pool100 = r100.pool()
m = dse.visited(pool100)["cost"].min()
print(f"100 simulations in {r100.seconds:.1f} s — best {m:.3f} ms, rank {dse.rank_of(m, oracle)}")

ax = plots.best_so_far({"Bayesian, 40 evals": pool40, "Bayesian, 100 evals": pool100},
                       oracle_best=best, baseline=baseline, n_init=20)
plt.show()
"""),
    ("md", """\
A hundred simulations is 3 % of the space. Over the seeds we tried beforehand,
four of five land exactly on the optimum with that budget; the fifth never
leaves the bulk. Random search with the same budget is still four to five
times off in the median.

## A space that cannot be enumerated live

4096 × 4096 on up to 2048 DPUs (32 ranks). The number of DPUs is now a
parameter too. We enumerated this space *once*, on 14 cores, in seven minutes,
so that the search can be scored; on one core it would take an hour and a half.
"""),
    ("code", """\
big_mlir = DATA / "big" / "gemv.mlir"
big_oracle = dse.load_pool(DATA / "big" / "pool.csv")
big_space = dse.load_space(DATA / "big" / "space.json")
big_best = big_oracle["cost"].min()

print(dse.cinm.describe_space(big_space))
print()
print(f"best {big_best:.2f} ms, median {big_oracle['cost'].median():.1f} ms; "
      f"within 10 % of the best: {(big_oracle['cost'] <= 1.1 * big_best).sum()} configurations")
"""),
    ("code", """\
rb = dse.bo(big_mlir, "big_bo", max_evals=100, n_init=20)
poolb = rb.pool()
mb = dse.visited(poolb)["cost"].min()
big_baseline = dse.random_search_baseline(big_oracle, budget=100)

print(f"100 simulations in {rb.seconds:.1f} s")
print(f"best found {mb:.2f} ms — rank {dse.rank_of(mb, big_oracle)} of {len(big_oracle)}, "
      f"{(mb / big_best - 1) * 100:.0f} % above the optimum")
print(f"random search, same budget: median {np.median(big_baseline[:, -1]):.1f} ms "
      f"({(np.median(big_baseline[:, -1]) / big_best - 1) * 100:.0f} % above)")
"""),
    ("code", """\
ax = plots.best_so_far({"Bayesian, 100 evals": poolb}, oracle_best=big_best,
                       baseline=big_baseline, n_init=20,
                       title="26,238 feasible configurations — best latency found vs. simulations")
plt.show()
"""),
    ("code", """\
cols = dse.param_columns(big_oracle) + ["cost"]
big_oracle.sort_values("cost").head(5)[cols].style.format({"cost": "{:.3f}"}).hide(axis="index")
"""),
    ("md", """\
Compare with the small space: there the optimum ran 8 tasklets on 64 DPUs;
here it runs **1** tasklet on all **2048**. The right hardware parameters flip
with the problem size — one more reason they cannot be fixed before the tiling
is chosen.

## Extra · a space nobody enumerates

A matrix–matrix product on the same 32 ranks. Three loop dimensions instead of
two: nine parameters.
"""),
    ("code", """\
gemm_mlir = DATA / "large" / "gemm.mlir"
gemm_space = dse.dump_space(gemm_mlir, "gemm_space")
print(dse.cinm.describe_space(gemm_space))
print(f"\\nAt 60 ms per simulation, enumerating it would take "
      f"{gemm_space['feasible_size'] * 0.06 / 3600:.0f} hours on one core.")
"""),
    ("code", """\
rl = dse.bo(gemm_mlir, "gemm_bo", max_evals=60, n_init=15)
pooll = rl.pool()
vl = dse.visited(pooll)

print(f"60 simulations in {rl.seconds:.1f} s")
ax = plots.best_so_far({"Bayesian, 60 evals": pooll}, n_init=15,
                       title="2.8 million feasible configurations — no oracle to compare with")
plt.show()

vl.sort_values("cost").head(3)[dse.param_columns(pooll) + ["cost"]].style.format({"cost": "{:.2f}"}).hide(axis="index")
"""),
    ("md", """\
No oracle here, and there never will be one. What the previous two spaces
bought us is *calibration*: on spaces where the answer was known, this budget
and this search landed within a few percent of it. That is the evidence one
has when applying the same procedure where the answer is not known — which is
every real case.
"""),
]


def main():
    (HERE.parent / "slides" / "figures").mkdir(parents=True, exist_ok=True)
    for name, cells, title in [
        ("01_space.ipynb", NB1, "The design space"),
        ("02_landscape.ipynb", NB2, "The landscape"),
        ("03_search.ipynb", NB3, "The search"),
    ]:
        nbf.write(nb(cells, title), HERE / name)
        print("wrote", name)


if __name__ == "__main__":
    main()
