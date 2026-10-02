"""Everything precomputed for the tutorial, and what depends on it.

    doit list                 # the tasks
    doit                      # bring everything up to date
    doit sweep                # one group: space, oracle, sweep, figures, slides
    doit oracle:big           # one task
    doit forget oracle:big    # make doit consider it stale

The chain, each step depending on exactly what it reads:

    data/<k>/<kernel>.mlir, cinm-opt
        -> space:<k>    data/<k>/space.json          the configuration space
        -> oracle:<k>   data/<k>/pool.csv            every feasible config, simulated
        -> sweep:<name> data/sweeps/<name>/seed_*/   BO from several seeds
    oracles, sweeps, dse/plots.py
        -> figures      slides/figures/*.pdf
    figures, slides/dse-tutorial.tex
        -> slides       slides/dse-tutorial.pdf

cinm-opt is a dependency of every run, so a rebuilt compiler makes all of them
stale; `doit forget` and `doit ignore` are the way around that when it should
not. It is the one the notebooks use (CINM_OPT, PATH, or a Cinnamon checkout
beside this repository; see dse/paths.py). Runs use the fast simulator, as the
notebooks do, and every core: a search visits the same configurations whatever
the thread count, so the sweeps are what a one-core Binder kernel produces
with the same seeds. JOBS sets the thread count.

The seed tables on the slides are typed into slides/dse-tutorial.tex; update
them by hand when the sweeps change.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SLIDES = ROOT / "slides"
FIGURES = SLIDES / "figures"
JOBS = int(os.environ.get("JOBS") or os.cpu_count() or 1)

DOIT_CONFIG = {"default_verbosity": 2}

# The three problems: data/<key>/<kernel>
KERNELS = {
    "small": "gemv.mlir",  # 1024^2 gemv, one rank
    "big": "gemv.mlir",  # 4096^2 gemv, 32 ranks
    "large": "gemm.mlir",  # 2048^2 gemm, 32 ranks: no oracle, far too big
}
ORACLES = ["small", "big"]
# name -> (problem, max-evals, n-init, seeds). The pass numbers its seeds
# 42, 73, 104, ...: 42 plus 31 per seed.
SWEEPS = {
    "small_40x10": ("small", 40, 10, 10),
    "small_100x5": ("small", 100, 20, 5),
    "big_100x5": ("big", 100, 20, 5),
}
SEED_BASE, SEED_STEP = 42, 31


def _cinm_opt() -> Path | None:
    try:
        import dse

        return dse.cinm_opt()
    except Exception:
        # Resolved again, with its error, by the first task that needs it;
        # `doit list` and the figure and slide tasks work without one.
        return None


CINM_OPT = _cinm_opt()


def _kernel(key: str) -> Path:
    return DATA / key / KERNELS[key]


def _compiler_dep() -> list[str]:
    return [str(CINM_OPT)] if CINM_OPT else []


def _infer(kernel: Path, **options: object):
    """A python-action running --upmem-infer-accelerator on ``kernel``.

    It dumps into a scratch directory and hands it to ``collect``, which moves
    what it needs into place: an interrupted run leaves the old files as they
    were, and doit does not record a target that was never written.
    """

    def action(collect, targets):
        if CINM_OPT is None:
            import dse

            dse.cinm_opt()  # raises, saying what it tried
        opts = {"simulator": "fast", **options}
        with tempfile.TemporaryDirectory() as tmp:
            opts["dump-dir"] = tmp
            spec = " ".join(f"{k.replace('_', '-')}={v}" for k, v in opts.items())
            cmd = [
                str(CINM_OPT),
                str(kernel),
                "--cinm-isolate-compute-blocks",
                f"--upmem-infer-accelerator={spec}",
                "-o",
                os.devnull,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(" ".join(cmd))
                print(proc.stderr[-4000:])
                return False
            collect(Path(tmp))

    return action


def _only(dump: Path, name: str) -> Path:
    """The one ``name`` in a dump: the pass puts it under infer_<op>/."""
    found = sorted(dump.rglob(name))
    if len(found) != 1:
        raise RuntimeError(f"expected one {name} under {dump}, found {len(found)}")
    return found[0]


def _move_one(name: str, target: Path):
    def collect(dump: Path):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(_only(dump, name), target)

    return collect


def task_space():
    """The configuration space of each problem (dump-space-only)."""
    for key in KERNELS:
        target = DATA / key / "space.json"
        yield {
            "name": key,
            "file_dep": [str(_kernel(key))] + _compiler_dep(),
            "targets": [str(target)],
            "actions": [
                (
                    _infer(_kernel(key), dump_space_only=1, n_solve_workers=JOBS),
                    [_move_one("space.json", target)],
                )
            ],
        }


def task_oracle():
    """Every feasible configuration simulated: what searches are scored against."""
    for key in ORACLES:
        target = DATA / key / "pool.csv"
        yield {
            "name": key,
            "file_dep": [str(_kernel(key))] + _compiler_dep(),
            "targets": [str(target)],
            "actions": [
                (
                    _infer(
                        _kernel(key),
                        exhaustive_search=1,
                        dump_full_pool=1,
                        n_workers=JOBS,
                        n_solve_workers=JOBS,
                    ),
                    [_move_one("pool.csv", target)],
                )
            ],
        }


def _seed_pools(name: str) -> dict[int, Path]:
    _, _, _, seeds = SWEEPS[name]
    return {
        SEED_BASE + SEED_STEP * i: DATA / "sweeps" / name / f"seed_{SEED_BASE + SEED_STEP * i}" / "pool.csv"
        for i in range(seeds)
    }


def task_sweep():
    """Bayesian optimisation of one problem from several seeds."""
    for name, (key, evals, init, seeds) in SWEEPS.items():
        pools = _seed_pools(name)

        def collect(dump: Path, name=name, pools=pools):
            for seed, target in pools.items():
                found = sorted(dump.rglob(f"seed_{seed}/pool.csv"))
                if len(found) != 1:
                    raise RuntimeError(f"{name}: no pool for seed {seed} in the dump")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(found[0], target)

        yield {
            "name": name,
            "file_dep": [str(_kernel(key))] + _compiler_dep(),
            "targets": [str(p) for p in pools.values()],
            "actions": [
                (
                    _infer(
                        _kernel(key),
                        max_evals=evals,
                        n_init=init,
                        n_seeds=seeds,
                        n_workers=JOBS,
                        n_solve_workers=JOBS,
                    ),
                    [collect],
                )
            ],
        }


FIGURE_FILES = [
    "histogram.pdf",
    "seeds_small_40.pdf",
    "seeds_small_100.pdf",
    "seeds_big_100.pdf",
]


def task_figures():
    """The slides' figures, drawn from the oracles and sweeps."""
    sweep_pools = [str(p) for name in SWEEPS for p in _seed_pools(name).values()]
    return {
        "file_dep": [
            str(FIGURES / "_make.py"),
            str(ROOT / "dse" / "plots.py"),
            str(ROOT / "dse" / "cinm.py"),
        ]
        + [str(DATA / key / "pool.csv") for key in ORACLES]
        + sweep_pools,
        "targets": [str(FIGURES / f) for f in FIGURE_FILES],
        "actions": [["python", str(FIGURES / "_make.py")]],
    }


def task_slides():
    """The deck. Needs the system's TeX Live (metropolis, tikz, fontawesome5)."""
    return {
        "file_dep": [str(SLIDES / "dse-tutorial.tex")]
        + [str(FIGURES / f) for f in FIGURE_FILES],
        "targets": [str(SLIDES / "dse-tutorial.pdf")],
        "actions": [
            f"cd {SLIDES} && latexmk -pdf -interaction=nonstopmode -halt-on-error dse-tutorial.tex"
        ],
    }
