"""Support code for the DSE tutorial notebooks.

The notebooks stay about design-space exploration; how ``cinm-opt`` is invoked,
how its dumps are read and how they are drawn lives here.
"""

from .cinm import (  # noqa: F401
    CostReport,
    Run,
    bo,
    dump_space,
    Lowering,
    eval_solution,
    lower,
    exhaustive,
    load_pool,
    load_space,
    param_columns,
    percentile_of,
    random_search_baseline,
    rank_of,
    visited,
)
from .display import show_mlir  # noqa: F401
from .paths import DATA, ROOT, WORK, cinm_opt  # noqa: F401
from .picker import ConfigPicker  # noqa: F401
