"""Frequency tables for one attribute, and dependencies between attributes.

Marginals are defined over the order of the domain, so ordinal domains (phase I-IV, a
Likert scale) can get a bell shape or a skew towards one end.

* ``Uniform()``
* ``Normal(mean, sd)``: normal density at the value indices 0..d-1. ``mean`` is an index
  (may be fractional) or a value of the domain; by default the middle of the domain.
* ``Skewed(s, towards)``: Zipf weights 1/rank^s; rank 1 is the first value, the last value,
  or a named value (then the rank grows with the distance from it).
* ``Frequencies({value: weight})``: explicit weights; values left out get weight 0.

Dependencies replace an attribute's marginal by a conditional distribution:

* ``Depends(attr, on, strength, mapping)``:
  P(attr | on = a) = (1 - strength) * marginal(attr) + strength * [attr = mapping(a)].
  Without a mapping the domains are aligned by rank (first to first, last to last).
* ``CPT(attr, on, rows)``: explicit conditional table, possibly on several attributes.
  Rows left out fall back to the attribute's marginal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _normalise(weights, what):
    weights = np.asarray(weights, dtype=float)
    if (weights < 0).any() or not np.isfinite(weights).all():
        raise ValueError(f"{what}: weights must be finite and non-negative")
    total = weights.sum()
    if total <= 0:
        raise ValueError(f"{what}: weights sum to zero")
    return weights / total


# -- marginals -------------------------------------------------------------------------
@dataclass(frozen=True)
class Uniform:
    def probabilities(self, domain):
        return np.full(len(domain), 1.0 / len(domain))

    def to_dict(self):
        return {"type": "uniform"}


@dataclass(frozen=True)
class Normal:
    mean: float | str | None = None
    sd: float = 1.0

    def probabilities(self, domain):
        if self.sd <= 0:
            raise ValueError("Normal: sd must be positive")
        if self.mean is None:
            centre = (len(domain) - 1) / 2
        elif isinstance(self.mean, str):
            centre = domain.index(self.mean)
        else:
            centre = float(self.mean)
        z = (np.arange(len(domain)) - centre) / self.sd
        log_density = -0.5 * z**2
        return _normalise(np.exp(log_density - log_density.max()), "Normal")

    def to_dict(self):
        return {"type": "normal", "mean": self.mean, "sd": self.sd}


@dataclass(frozen=True)
class Skewed:
    s: float = 1.0
    towards: str = "first"

    def probabilities(self, domain):
        index = np.arange(len(domain))
        if self.towards == "first":
            rank = index + 1
        elif self.towards == "last":
            rank = len(domain) - index
        else:
            rank = np.abs(index - domain.index(self.towards)) + 1
        return _normalise(1.0 / rank.astype(float) ** self.s, "Skewed")

    def to_dict(self):
        return {"type": "skewed", "s": self.s, "towards": self.towards}


@dataclass(frozen=True)
class Frequencies:
    weights: dict = field(default_factory=dict)

    def probabilities(self, domain):
        unknown = set(self.weights) - set(domain)
        if unknown:
            raise ValueError(f"Frequencies: values not in the domain: {sorted(unknown)}")
        return _normalise([self.weights.get(v, 0.0) for v in domain], "Frequencies")

    def to_dict(self):
        return {"type": "frequencies", "weights": dict(self.weights)}


MARGINALS = {"uniform": Uniform, "normal": Normal, "skewed": Skewed, "frequencies": Frequencies}


def marginal_from_dict(data):
    data = dict(data)
    kind = data.pop("type")
    return MARGINALS[kind](**data)


# -- dependencies ----------------------------------------------------------------------
def rank_alignment(source, target):
    """Map the i-th of ``source`` to the value of ``target`` at the same relative position."""
    if len(source) == 1:
        return {source[0]: target[0]}
    return {
        a: target[int(math.floor(i * (len(target) - 1) / (len(source) - 1) + 0.5))]
        for i, a in enumerate(source)
    }


@dataclass(frozen=True)
class Depends:
    attr: str
    on: str
    strength: float = 0.5
    mapping: dict | None = None

    @property
    def parents(self):
        return (self.on,)

    def table(self, domains, base):
        """Array of shape (|on|, |attr|): P(attr | on)."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Depends({self.attr}): strength must be in [0, 1]")
        source, target = domains[self.on], domains[self.attr]
        mapping = self.mapping or rank_alignment(source, target)
        out = np.empty((len(source), len(target)))
        for i, a in enumerate(source):
            if mapping.get(a) not in target:
                raise ValueError(f"Depends({self.attr}): {a!r} maps to {mapping.get(a)!r}, not a value of {self.attr}")
            onehot = np.zeros(len(target))
            onehot[target.index(mapping[a])] = 1.0
            out[i] = (1 - self.strength) * base + self.strength * onehot
        return out

    def to_dict(self):
        out = {"type": "depends", "attr": self.attr, "on": self.on, "strength": self.strength}
        if self.mapping is not None:
            out["mapping"] = dict(self.mapping)
        return out


@dataclass(frozen=True)
class CPT:
    attr: str
    on: tuple[str, ...]
    rows: dict = field(default_factory=dict)   # tuple of parent values -> {value: weight}

    def __post_init__(self):
        on = (self.on,) if isinstance(self.on, str) else tuple(self.on)
        object.__setattr__(self, "on", on)
        rows = {}
        for given, weights in self.rows.items():
            given = (given,) if isinstance(given, str) else tuple(given)
            if len(given) != len(on):
                raise ValueError(f"CPT({self.attr}): row {given} does not match on={on}")
            rows[given] = dict(weights)
        object.__setattr__(self, "rows", rows)

    @property
    def parents(self):
        return self.on

    def table(self, domains, base):
        """Array of shape (|on_1|, ..., |on_k|, |attr|)."""
        shape = tuple(len(domains[a]) for a in self.on)
        target = domains[self.attr]
        out = np.broadcast_to(base, shape + (len(target),)).copy()
        for given, weights in self.rows.items():
            index = tuple(domains[a].index(v) for a, v in zip(self.on, given))
            out[index] = Frequencies(weights).probabilities(target)
        return out

    def to_dict(self):
        return {
            "type": "cpt",
            "attr": self.attr,
            "on": list(self.on),
            "rows": [{"given": list(given), "p": weights} for given, weights in self.rows.items()],
        }


def dependency_from_dict(data):
    kind = data["type"]
    if kind == "depends":
        return Depends(data["attr"], data["on"], data.get("strength", 0.5), data.get("mapping"))
    if kind == "cpt":
        return CPT(data["attr"], tuple(data["on"]), {tuple(r["given"]): r["p"] for r in data["rows"]})
    raise ValueError(f"unknown dependency type {kind!r}")
