# Design-space exploration for compute-near-memory systems

A 45-minute tutorial: slides for the board, three notebooks for the attendees,
using Cinnamon's accelerator-inference pass as the running example.

```
slides/        Beamer deck; `make` in there builds slides/dse-tutorial.pdf
               figures/_make.py redraws its figures from data/ (oracles and seed sweeps)
notebooks/     01_space  02_landscape  03_search  (+ _generate.py, their source)
dse/           what the notebooks import: running cinm-opt, reading its dumps, plots, the picker
data/          the kernels and the precomputed oracles
  small/       1024² gemv, 64 DPUs — 3234 feasible configurations, exhaustively simulated
  big/         4096² gemv, ≤2048 DPUs — 26,238 feasible, exhaustively simulated (14 cores, 7 min)
  large/       2048² gemm, ≤2048 DPUs — 2.8 million feasible; space only, no oracle
  sweeps/      the BO seed sweeps the slides quote: 40×10 and 100×5 on small, 100×5 on big
binder/        the Dockerfile mybinder.org builds
work/          everything the notebooks write; not tracked
```

## Running the notebooks

They need `cinm-opt` and Cinnamon's Python environment. From a Cinnamon checkout
that has been built:

```sh
cd /path/to/cinm-mlir
pixi run -e host jupyter notebook --notebook-dir=/path/to/esweek2026-tutorial/notebooks
```

`pixi run` puts `build/bin` on `PATH`, which is where the notebooks find
`cinm-opt`; `CINM_OPT=/path/to/cinm-opt` names one explicitly. Use the
environment the compiler was built in (`host` for a system-compiler build,
default for the pixi toolchain).

Every search the notebooks run is pinned to one core and the fast simulator,
the conditions of a Binder session. Timings on a laptop:

| Cell | Wall |
|---|---|
| picker + one `eval-solution` | < 1 s |
| `dump-space-only`, small space | 0.4 s |
| exhaustive with `tasklets` pinned (616 configs) | 34 s |
| BO, 40 evaluations | 3 s |
| BO, 40 evaluations × 5 seeds | 19 s |
| BO, 100 evaluations | 8 s |
| BO, 100 evaluations, big space | 14 s |
| `dump-space-only` + BO 60 evaluations, 2.8 M space | 4 s + 14 s |

Binder's cores are slower; budget two to three times these.

## Regenerating the oracles

The pools in `data/` came from

```sh
cinm-opt data/small/gemv.mlir --cinm-isolate-compute-blocks \
  --upmem-infer-accelerator="exhaustive-search=1 dump-full-pool=1 simulator=fast dump-dir=out"
```

and the same for `data/big/gemv.mlir` (with `n-workers` at the machine's core
count: it is 26k simulations). `dump-space-only=1` writes `space.json` alone.

## Binder

`binder/Dockerfile` starts from the tutorial image Cinnamon's CI publishes and
adds the user and layout Binder requires. Point mybinder.org at this
repository once it is public; the launch link goes on the first slide.
