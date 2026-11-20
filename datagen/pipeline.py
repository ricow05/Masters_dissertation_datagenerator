"""Generator output plus injector: ``inject(generate(spec), ErrorSpec(...))``.

The generator and the injector do not import each other. This module connects them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from .data import Coded
from .generator.generate import CleanData, streams
from .generator.spec import DatasetSpec
from .injector import stats as error_stats
from .injector.inject import eligible_attributes, inject_errors
from .injector.spec import ErrorSpec


@dataclass
class DirtyData:
    spec: DatasetSpec
    error_spec: ErrorSpec
    clean: Coded
    dirty: Coded
    errors: pd.DataFrame          # the error log (errors.csv)
    stats: dict
    timings: dict = field(default_factory=dict)

    def frames(self) -> dict[str, pd.DataFrame]:
        """``<Parent>_clean``, ``<Parent>_dirty``, ``<Child>_clean``, ``<Child>_dirty``."""
        parent, child = self.spec.schema.parent.name, self.spec.schema.child.name
        pc, cc = self.clean.to_frames()
        pd_, cd = self.dirty.to_frames()
        return {f"{parent}_clean": pc, f"{parent}_dirty": pd_, f"{child}_clean": cc, f"{child}_dirty": cd}


def error_rng(spec: DatasetSpec, error_spec: ErrorSpec):
    """``error_spec.seed`` if set, else the error stream of the clean data's seed."""
    if error_spec.seed is not None:
        return error_spec.seed
    return streams(spec.seed)[2]


def merge_stats(clean_stats: dict, injected: dict) -> dict:
    """One statistics dictionary for a folder with clean and dirty data."""
    return {
        **clean_stats,
        "violations": {**clean_stats["violations"], **injected["violations"]},
        "errors": injected["errors"],
    }


def inject(clean: CleanData, error_spec: ErrorSpec) -> DirtyData:
    """Add the errors of ``error_spec`` to generated (or read) clean data."""
    spec = clean.spec
    t = time.perf_counter()
    injection = inject_errors(spec.schema, spec.rules, clean.clean, error_spec, error_rng(spec, error_spec))
    errors_s = time.perf_counter() - t

    t = time.perf_counter()
    eligible = eligible_attributes(spec.schema, spec.rules, error_spec)
    injected = error_stats.compute(spec.schema, spec.rules, clean.clean, injection.dirty, injection.log, eligible,
                                   injection.shortfalls)
    timings = {**clean.timings, "errors_s": round(errors_s, 4), "error_stats_s": round(time.perf_counter() - t, 4)}
    return DirtyData(spec, error_spec, clean.clean, injection.dirty, injection.log,
                     merge_stats(clean.stats, injected), timings)
