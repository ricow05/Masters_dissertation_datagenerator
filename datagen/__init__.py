"""Parent-child data generator that satisfies constant selection rules, and an error injector.

The two halves are separate subpackages that do not import each other:

* ``datagen.generator``: ``generate(DatasetSpec) -> CleanData`` (and ``fit``);
* ``datagen.injector``: ``inject_errors(schema, rules, clean, ErrorSpec, rng)`` for any clean data.

``inject(clean_data, ErrorSpec) -> DirtyData`` connects them. Standalone: depends on numpy
and pandas only. Output folders follow ``FORMAT.md``.
"""
__version__ = "0.3.0"

from .rules import Rule  # noqa: E402
from .schema import Schema, Table, two_tables  # noqa: E402
from .generator import (  # noqa: E402
    CPT, CleanData, DatasetSpec, Depends, Empirical, FitResult, Fixed, Frequencies, Geometric, Normal, Poisson,
    Skewed, Uniform, UniformRange, fit, generate,
)
from .injector import ErrorSpec, RowWeight, inject_errors  # noqa: E402
from .pipeline import DirtyData, inject  # noqa: E402
from .io import (  # noqa: E402
    load_error_spec, load_spec, read, read_clean, recompute_stats, save_error_spec, save_spec, write,
)
from .sweep import load_sweep, run_sweep  # noqa: E402

__all__ = [
    "Rule", "Schema", "Table", "two_tables",
    "DatasetSpec", "CleanData", "generate", "fit", "FitResult",
    "CPT", "Depends", "Frequencies", "Normal", "Skewed", "Uniform",
    "Empirical", "Fixed", "Geometric", "Poisson", "UniformRange",
    "ErrorSpec", "RowWeight", "inject_errors",
    "DirtyData", "inject",
    "load_spec", "save_spec", "load_error_spec", "save_error_spec", "read", "read_clean", "write", "recompute_stats",
    "load_sweep", "run_sweep",
]
