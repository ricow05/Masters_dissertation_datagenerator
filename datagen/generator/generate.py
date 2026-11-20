"""Generate clean data from a spec. Errors are added afterwards by ``datagen.injector``."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..data import Coded
from ..rules import check_rules
from . import stats as stats_module
from .network import Network
from .sampler import Sampler
from .spec import DatasetSpec


@dataclass
class CleanData:
    spec: DatasetSpec
    clean: Coded
    stats: dict
    timings: dict = field(default_factory=dict)

    def frames(self) -> dict[str, pd.DataFrame]:
        """``<Parent>_clean``, ``<Child>_clean``."""
        parent, child = self.spec.schema.parent.name, self.spec.schema.child.name
        pc, cc = self.clean.to_frames()
        return {f"{parent}_clean": pc, f"{child}_clean": cc}


def streams(seed):
    """Independent random streams for children counts, clean values and errors.

    The generator reads the first two. The third is the injector's default stream for data
    generated with this seed (``datagen.inject``), so the error settings never change the
    clean data.
    """
    fanout, clean, errors = np.random.SeedSequence(seed).spawn(3)
    return np.random.default_rng(fanout), np.random.default_rng(clean), np.random.default_rng(errors)


def network_and_sampler(spec: DatasetSpec):
    """The network of the spec and the sampler that conditions it on the rules."""
    network = Network(spec.schema, spec.marginals, spec.dependencies)
    return network, Sampler(spec.schema, spec.rules, network, spec.max_combinations)


def clean_stats(spec: DatasetSpec, clean: Coded, network=None, sampler=None) -> dict:
    if network is None or sampler is None:
        network, sampler = network_and_sampler(spec)
    return stats_module.compute(spec.schema, spec.rules, network, sampler.target_marginals(), clean)


def generate(spec: DatasetSpec, seed=None) -> CleanData:
    if seed is not None:
        spec = replace(spec, seed=seed)
    if spec.parents < 0:
        raise ValueError("parents must be non-negative")
    timings = {}
    start = time.perf_counter()
    check_rules(spec.schema, spec.rules)
    network, sampler = network_and_sampler(spec)
    timings["setup_s"] = time.perf_counter() - start

    rng_fanout, rng_clean, _ = streams(spec.seed)
    t = time.perf_counter()
    counts = spec.fanout.sample(rng_fanout, spec.parents)
    clean = sampler.sample(counts, rng_clean, spec.conditioning)
    timings["clean_s"] = time.perf_counter() - t

    t = time.perf_counter()
    stats = clean_stats(spec, clean, network, sampler)
    timings["stats_s"] = time.perf_counter() - t
    return CleanData(spec, clean, stats, {k: round(v, 4) for k, v in timings.items()})
