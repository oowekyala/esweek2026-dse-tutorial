"""Running ``--upmem-infer-accelerator`` and reading what it dumps.

Every search writes a *pool*: one row per configuration, with the parameter
values as columns, whether the configuration was visited, its simulated cost,
and -- for a Bayesian run -- the surrogate's mean, uncertainty and acquisition
value at the time it was chosen. ``space.json`` describes the parameters and
the constraints that carved the feasible set out of the Cartesian product.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .paths import WORK, cinm_opt

# Columns every pool has besides the parameters.
META_COLUMNS = {
    "visited",
    "valid",
    "cost",
    "mu",
    "sigma",
    "acq",
    "eval_iter",
    "eval_time_ms",
    "cpu_time_ms",
}

# The notebooks run on one core in the tutorial image. Everything the search
# does in parallel is pinned down here, and the fast simulator is the one the
# whole tutorial is calibrated on.
BASE_OPTIONS: dict[str, object] = {
    "simulator": "fast",
    "n-workers": 1,
    "n-solve-workers": 1,
}


@dataclass
class Run:
    """One invocation of the inference pass and where it wrote."""

    out_dir: Path
    stdout: str
    stderr: str
    seconds: float
    options: dict[str, object] = field(default_factory=dict)

    def _find(self, name: str) -> list[Path]:
        return sorted(self.out_dir.rglob(name))

    def pool(self, seed: int | None = None) -> pd.DataFrame:
        """The pool of the run, or of one seed of a multi-seed run."""
        files = self._find("pool.csv")
        if seed is not None:
            files = [f for f in files if f.parent.name == f"seed_{seed}"]
        if not files:
            raise FileNotFoundError(f"no pool.csv under {self.out_dir}")
        if len(files) > 1 and seed is None:
            raise ValueError(
                f"{len(files)} pools under {self.out_dir}; pick a seed or use pools()"
            )
        return load_pool(files[0])

    def pools(self) -> dict[int, pd.DataFrame]:
        """Every seed's pool, keyed by seed."""
        out = {}
        for f in self._find("pool.csv"):
            m = re.match(r"seed_(\d+)", f.parent.name)
            out[int(m.group(1)) if m else 0] = load_pool(f)
        return out

    def space(self) -> dict:
        files = self._find("space.json")
        if not files:
            raise FileNotFoundError(f"no space.json under {self.out_dir}")
        return load_space(files[0])

    def rounds(self, seed: int | None = None) -> pd.DataFrame:
        """Per-round surrogate diagnostics of a Bayesian run."""
        files = self._find("rounds.csv")
        if seed is not None:
            files = [f for f in files if f.parent.name == f"seed_{seed}"]
        return pd.read_csv(files[0])


def _format_options(options: Mapping[str, object]) -> str:
    parts = []
    for key, value in options.items():
        if isinstance(value, bool):
            value = int(value)
        parts.append(f"{key}={value}")
    return " ".join(parts)


def run(
    mlir: Path,
    out_dir: Path,
    *,
    show_progress: bool = True,
    **options: object,
) -> Run:
    """Run cinm-opt with --upmem-infer-accelerator on ``mlir``, dumping to ``out_dir``.

    ``options`` are the pass's own options, e.g. ``max_evals=40`` becomes
    ``max-evals=40``. The dump directory is cleared first, so a re-run of a
    cell never mixes with the previous run's files.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    opts = dict(BASE_OPTIONS)
    opts.update({k.replace("_", "-"): v for k, v in options.items()})
    opts["dump-dir"] = str(out_dir)

    cmd = [
        str(cinm_opt()),
        str(mlir),
        "--cinm-isolate-compute-blocks",
        f"--upmem-infer-accelerator={_format_options(opts)}",
        "-o",
        "/dev/null",
    ]

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    )
    # The pass reports progress on stderr as "[progress mm:ss] ..." lines.
    # Those are shown in place; everything else is kept.
    stderr_lines: list[str] = []
    assert proc.stderr is not None
    for line in proc.stderr:
        if line.startswith("[progress") and show_progress:
            sys.stdout.write("\r" + line.rstrip()[:100].ljust(100))
            sys.stdout.flush()
        else:
            stderr_lines.append(line)
    stdout = proc.stdout.read() if proc.stdout else ""
    proc.wait()
    seconds = time.perf_counter() - start
    if show_progress:
        sys.stdout.write("\r" + " " * 100 + "\r")
        sys.stdout.flush()

    stderr = "".join(stderr_lines)
    if proc.returncode != 0:
        raise RuntimeError(
            f"cinm-opt failed ({proc.returncode}):\n{' '.join(cmd)}\n{stderr}"
        )
    return Run(out_dir, stdout, stderr, seconds, opts)


# ---------------------------------------------------------------------------
# The three ways the notebooks use the pass
# ---------------------------------------------------------------------------


def dump_space(mlir: Path, name: str = "space") -> dict:
    """Build the configuration space and stop: no evaluation."""
    r = run(mlir, WORK / name, show_progress=False, dump_space_only=True)
    return r.space()


def exhaustive(mlir: Path, name: str = "exhaustive", **options: object) -> Run:
    """Evaluate every feasible configuration."""
    return run(
        mlir, WORK / name, exhaustive_search=True, dump_full_pool=True, **options
    )


def bo(
    mlir: Path,
    name: str = "bo",
    *,
    max_evals: int = 40,
    n_init: int = 10,
    seeds: int = 1,
    **options: object,
) -> Run:
    """Bayesian optimisation with a budget of ``max_evals`` simulations."""
    # The pass's own defaults otherwise, n-random-candidates included: scoring
    # its 16k candidates per round costs ~20 ms on one core, and cutting them
    # to 1024 halved the seeds that found the optimum on both spaces here.
    opts: dict[str, object] = {
        "max_evals": max_evals,
        "n_init": n_init,
    }
    if seeds > 1:
        opts["n_seeds"] = seeds
    opts.update(options)
    return run(mlir, WORK / name, **opts)


@dataclass
class CostReport:
    total_ms: float
    breakdown: dict[str, float]
    config: dict[str, object]

    def _repr_html_(self) -> str:
        rows = "".join(
            f"<tr><td style='padding:0 1em'>{k}</td><td style='text-align:right'>{v:.3f}</td></tr>"
            for k, v in self.breakdown.items()
        )
        return (
            f"<b>Estimated cost: {self.total_ms:.3f} ms</b>"
            f"<table style='margin-top:.3em'>{rows}</table>"
        )


def eval_solution(mlir: Path, config: Mapping[str, object]) -> CostReport:
    """Simulate one named configuration and report its cost breakdown."""
    solution = ",".join(f"{k}={v}" for k, v in config.items())
    r = run(
        mlir, WORK / "eval", show_progress=False, eval_solution=solution
    )
    text = r.stdout + r.stderr
    m = re.search(r"Estimated cost:\s*([0-9.]+)\s*ms", text)
    if not m:
        raise RuntimeError(f"no cost in cinm-opt's output:\n{text}")
    breakdown = {
        k: float(v)
        for k, v in re.findall(r"^\s+([\w.]+):\s*([0-9.]+)\s*ms", text, re.M)
    }
    return CostReport(float(m.group(1)), breakdown, dict(config))


# ---------------------------------------------------------------------------
# Reading dumps
# ---------------------------------------------------------------------------


def load_pool(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Costs are only meaningful on rows that were simulated and feasible.
    if "cost" in df:
        df.loc[(df.get("valid", 1) != 1) | (df["cost"] <= 0), "cost"] = np.nan
    return df


def load_space(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def param_columns(pool: pd.DataFrame) -> list[str]:
    """The parameter columns of a pool, in the pool's order."""
    return [c for c in pool.columns if c not in META_COLUMNS]


def visited(pool: pd.DataFrame) -> pd.DataFrame:
    """The configurations a search actually simulated, in evaluation order."""
    v = pool[pool["visited"] == 1].copy()
    if "eval_iter" in v:
        v = v.sort_values("eval_iter")
    return v.reset_index(drop=True)


def rank_of(cost: float, oracle: pd.DataFrame) -> int:
    """1-based rank of a cost among the oracle's configurations."""
    return int((oracle["cost"] < cost).sum()) + 1


def percentile_of(cost: float, oracle: pd.DataFrame) -> float:
    """Share of the oracle that is at least as good as ``cost``, in percent."""
    return 100.0 * (oracle["cost"] <= cost).mean()


def random_search_baseline(
    oracle: pd.DataFrame,
    budget: int,
    trials: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Best-so-far curves of ``trials`` random searches drawn from the oracle.

    Returns an array of shape (trials, budget): entry [t, i] is the best cost
    trial t had seen after i+1 samples. Random search needs no simulator once
    the oracle exists, which is what makes it the baseline of choice.
    """
    rng = np.random.default_rng(seed)
    costs = oracle["cost"].dropna().to_numpy()
    out = np.empty((trials, budget))
    for t in range(trials):
        draw = costs[rng.choice(len(costs), budget, replace=False)]
        out[t] = np.minimum.accumulate(draw)
    return out


def constraints_table(space: dict) -> pd.DataFrame:
    return pd.DataFrame(space["constraints"])


def params_table(space: dict) -> pd.DataFrame:
    rows = []
    for p in space["params"]:
        if p["type"] == "range":
            values = f"{p['lo']} … {p['hi']}"
        else:
            values = ", ".join(str(v) for v in p.get("values", []))
        rows.append(
            {
                "parameter": p["name"],
                "kind": p["kind"],
                "values": values,
                "count": p["cardinality"],
                "meaning": p["doc"].split(";")[0].split("(")[0].strip(),
            }
        )
    return pd.DataFrame(rows)


def describe_space(space: dict) -> str:
    cart = space["cartesian_size"]
    feas = space["feasible_size"]
    return (
        f"Cartesian product: {cart:,.0f} combinations\n"
        f"Feasible:          {feas:,} configurations "
        f"({100 * feas / cart:.2g} %)\n"
        f"Built in {space['space_build_seconds']:.2f} s "
        f"by the constraint solver ({space['solver']['nodes']:,} search nodes)"
    )


def ordering_labels(space: dict, name: str) -> dict[tuple[int, ...], str]:
    """For a permutation parameter, map each encoded assignment to its label."""
    p = next(p for p in space["params"] if p["name"] == name)
    out = {}
    for o in p["orderings"]:
        key = tuple(o["assignment"][d] for d in p["dims"])
        out[key] = o["order"]
    return out


def iter_param_groups(space: dict) -> Iterable[tuple[str, list[str]]]:
    """Parameter name -> the pool columns that encode it (one, or arity many)."""
    for p in space["params"]:
        yield p["name"], p.get("dims", [p["name"]])
