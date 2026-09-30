"""Where things are: the repository, its data, and the cinm-opt binary."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# Everything the notebooks produce. Not tracked.
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)


def _runs(path: Path) -> str | None:
    """None if ``path`` starts and answers --version; otherwise why it did not."""
    try:
        proc = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True, timeout=30
        )
    except OSError as e:
        return str(e)
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout).strip().splitlines()[-1] if (proc.stderr or proc.stdout).strip() else f"exit {proc.returncode}"
    return None


def cinm_opt() -> Path:
    """The cinm-opt to run: the first candidate that actually starts.

    ``CINM_OPT`` names one explicitly. Otherwise PATH is tried -- where the
    tutorial image and `pixi run` put it -- then a Cinnamon checkout beside
    this repository, for a kernel an editor started without that PATH. Each
    candidate has to answer ``--version``: a Python environment may carry a
    cinm-opt from an old Cinnamon wheel whose libraries are long gone, and
    that one must not win by being first on PATH.
    """
    explicit = os.environ.get("CINM_OPT")
    if explicit:
        path = Path(explicit).expanduser()
        why = _runs(path) if path.is_file() else "not a file"
        if why is None:
            return path.resolve()
        raise FileNotFoundError(f"CINM_OPT={explicit}: {why}")

    candidates: list[Path] = []
    found = shutil.which("cinm-opt")
    if found:
        candidates.append(Path(found))
    candidates += [
        ROOT.parent / "cinm-mlir" / "build" / "bin" / "cinm-opt",
        ROOT.parent / "Cinnamon" / "build" / "bin" / "cinm-opt",
        ROOT.parent.parent / "MLIR" / "cinm-mlir" / "build" / "bin" / "cinm-opt",
    ]
    tried = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        why = _runs(candidate)
        if why is None:
            return candidate.resolve()
        tried.append(f"  {candidate}: {why}")
    raise FileNotFoundError(
        "No working cinm-opt. Set CINM_OPT to Cinnamon's build/bin/cinm-opt, run the "
        "notebooks with `pixi run notebook` from the Cinnamon checkout, or keep that "
        "checkout next to this repository."
        + ("\nTried:\n" + "\n".join(tried) if tried else "")
    )
