"""Number of children per parent row."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Fixed:
    """Exactly ``k`` children per parent (the GouDa setting)."""

    k: int = 5

    def sample(self, rng, n):
        if self.k < 0:
            raise ValueError("Fixed: k must be non-negative")
        return np.full(n, self.k, dtype=np.int64)

    def to_dict(self):
        return {"type": "fixed", "k": self.k}


@dataclass(frozen=True)
class UniformRange:
    """Uniform on low..high, both included."""

    low: int = 0
    high: int = 10

    def sample(self, rng, n):
        if not 0 <= self.low <= self.high:
            raise ValueError("UniformRange: need 0 <= low <= high")
        return rng.integers(self.low, self.high + 1, size=n).astype(np.int64)

    def to_dict(self):
        return {"type": "uniform_range", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class Poisson:
    lam: float = 5.0

    def sample(self, rng, n):
        if self.lam < 0:
            raise ValueError("Poisson: lam must be non-negative")
        return rng.poisson(self.lam, size=n).astype(np.int64)

    def to_dict(self):
        return {"type": "poisson", "lam": self.lam}


@dataclass(frozen=True)
class Geometric:
    """P(k) = (1 - p)^k p for k = 0, 1, 2, ...; mean (1 - p) / p."""

    p: float = 0.2

    def sample(self, rng, n):
        if not 0 < self.p <= 1:
            raise ValueError("Geometric: p must be in (0, 1]")
        return (rng.geometric(self.p, size=n) - 1).astype(np.int64)

    def to_dict(self):
        return {"type": "geometric", "p": self.p}


@dataclass(frozen=True)
class Empirical:
    """Explicit weights per number of children, e.g. learned from real data."""

    weights: dict = field(default_factory=dict)   # int -> weight

    def sample(self, rng, n):
        items = sorted((int(k), float(w)) for k, w in self.weights.items())
        counts = np.array([k for k, _ in items], dtype=np.int64)
        weights = np.array([w for _, w in items])
        if len(counts) == 0 or (counts < 0).any():
            raise ValueError("Empirical: need non-negative counts")
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Empirical: weights must be non-negative and not all zero")
        return counts[rng.choice(len(counts), size=n, p=weights / weights.sum())]

    def to_dict(self):
        return {"type": "empirical", "weights": {str(k): v for k, v in sorted(self.weights.items(), key=lambda kv: int(kv[0]))}}


FANOUTS = {"fixed": Fixed, "uniform_range": UniformRange, "poisson": Poisson, "geometric": Geometric}


def fanout_from_dict(data):
    data = dict(data)
    kind = data.pop("type")
    if kind == "empirical":
        return Empirical({int(k): float(v) for k, v in data["weights"].items()})
    return FANOUTS[kind](**data)
