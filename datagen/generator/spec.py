"""Everything that defines the clean data (a ``spec.json``, and ``"spec"`` in a folder's ``dataset.json``).

The errors are not part of it: they have their own ``ErrorSpec`` (``datagen.injector``),
saved as ``error_spec.json`` (``"error_spec"`` in ``dataset.json``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..rules import Rule
from ..schema import Schema
from .distributions import dependency_from_dict, marginal_from_dict
from .fanout import Fixed, fanout_from_dict

FORMAT_VERSION = 2
READABLE_VERSIONS = (1, 2)      # version 1 also held the errors; they are ignored here


@dataclass
class DatasetSpec:
    schema: Schema
    rules: list = field(default_factory=list)
    marginals: dict = field(default_factory=dict)        # attribute -> Uniform/Normal/Skewed/Frequencies
    dependencies: list = field(default_factory=list)     # Depends / CPT
    fanout: object = field(default_factory=lambda: Fixed(5))
    parents: int = 1000
    seed: int = 0
    conditioning: str = "sequential"
    max_combinations: int = 1_000_000

    def to_dict(self):
        return {
            "format_version": FORMAT_VERSION,
            "schema": self.schema.to_dict(),
            "rules": [r.to_dict() for r in self.rules],
            "marginals": {a: m.to_dict() for a, m in self.marginals.items()},
            "dependencies": [d.to_dict() for d in self.dependencies],
            "fanout": self.fanout.to_dict(),
            "parents": self.parents,
            "seed": self.seed,
            "conditioning": self.conditioning,
            "max_combinations": self.max_combinations,
        }

    @classmethod
    def from_dict(cls, data):
        version = data.get("format_version", FORMAT_VERSION)
        if version not in READABLE_VERSIONS:
            raise ValueError(f"unsupported spec format_version {version}")
        return cls(
            schema=Schema.from_dict(data["schema"]),
            rules=[Rule.from_dict(r) for r in data.get("rules", [])],
            marginals={a: marginal_from_dict(m) for a, m in data.get("marginals", {}).items()},
            dependencies=[dependency_from_dict(d) for d in data.get("dependencies", [])],
            fanout=fanout_from_dict(data.get("fanout", {"type": "fixed", "k": 5})),
            parents=int(data.get("parents", 1000)),
            seed=int(data.get("seed", 0)),
            conditioning=data.get("conditioning", "sequential"),
            max_combinations=int(data.get("max_combinations", 1_000_000)),
        )
