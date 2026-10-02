# Design-space exploration for compute-near-memory systems

A 45-minute tutorial: slides for the board, three notebooks for the attendees,
using Cinnamon's accelerator-inference pass as the running example.

```
slides/        Beamer deck; `make` in there builds slides/dse-tutorial.pdf
               figures/_make.py redraws its figures from data/ (oracles and seed sweeps)
notebooks/     01_space  02_landscape  03_search
dse/           what the notebooks import: running cinm-opt, reading its dumps, plots, the picker
data/          the kernels and the precomputed oracles
  small/       1024² gemv, 64 DPUs — 3234 feasible configurations, exhaustively simulated
  big/         4096² gemv, ≤2048 DPUs — 26,238 feasible, exhaustively simulated (14 cores, 7 min)
  large/       2048² gemm, ≤2048 DPUs — 2.8 million feasible; space only, no oracle
  sweeps/      the BO seed sweeps the slides quote: 40×10 and 100×5 on small, 100×5 on big
binder/        the Dockerfile mybinder.org builds
scripts/       check-notebooks.sh, behind `pixi run check`
work/          everything the notebooks write; not tracked
```

## Working on the tutorial

The repository has a pixi environment of its own: Jupyter, the notebooks'
dependencies, and the `dse` package installed editable.

```sh
pixi install
pixi run notebook     # Jupyter, rooted at notebooks/
pixi run check        # executes the three notebooks in order on one core, timing each
pixi run clean-notebooks  # strips outputs before a commit
pixi run figures      # redraws slides/figures from data/
pixi run slides       # builds slides/dse-tutorial.pdf (system TeX Live)
```

In an editor, choose `.pixi/envs/default/bin/python` as the notebook kernel;
after editing anything under `dse/`, restart the kernel.

`cinm-opt` is not part of the environment. The notebooks take it from
`CINM_OPT`, then from `PATH`, then from a Cinnamon checkout beside this
repository (`../cinm-mlir`, `../Cinnamon`, or `../../MLIR/cinm-mlir`), and only
accept a binary that answers `--version` -- an old Cinnamon wheel can leave a
`cinm-opt` on `PATH` whose libraries are gone. The tutorial image has it on
`PATH`.

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
