"""Two tables held as integer codes: the index of each value in its attribute's domain."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Coded:
    schema: object
    parents: dict[str, np.ndarray]    # attribute -> code per parent row
    children: dict[str, np.ndarray]   # attribute -> code per child row
    fk: np.ndarray                    # index of each child's parent row
    parent_keys: np.ndarray | None = None
    child_keys: np.ndarray | None = None

    def __post_init__(self):
        if self.parent_keys is None:
            self.parent_keys = np.arange(self.n_parents).astype(str).astype(object)
        if self.child_keys is None:
            self.child_keys = np.arange(self.n_children).astype(str).astype(object)

    @property
    def n_parents(self) -> int:
        return len(self.parent_keys) if self.parent_keys is not None else len(next(iter(self.parents.values())))

    @property
    def n_children(self) -> int:
        return len(self.fk)

    def children_per_parent(self) -> np.ndarray:
        return np.bincount(self.fk, minlength=self.n_parents)

    def joined(self) -> dict[str, np.ndarray]:
        """Every child row with its parent's attributes."""
        out = {a: codes[self.fk] for a, codes in self.parents.items()}
        out.update(self.children)
        return out

    def copy(self) -> "Coded":
        return Coded(
            self.schema,
            {a: c.copy() for a, c in self.parents.items()},
            {a: c.copy() for a, c in self.children.items()},
            self.fk,
            self.parent_keys,
            self.child_keys,
        )

    def column(self, table, attribute) -> np.ndarray:
        return (self.parents if table == self.schema.parent.name else self.children)[attribute]

    # -- conversion ------------------------------------------------------------------
    def to_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        schema = self.schema
        parent, child = schema.parent, schema.child
        parents = {parent.key: self.parent_keys}
        for a in schema.parent_attrs:
            parents[a] = np.asarray(schema.domains[a], dtype=object)[self.parents[a]]
        children = {child.key: self.child_keys, child.foreign_key: self.parent_keys[self.fk]}
        for a in schema.child_attrs:
            children[a] = np.asarray(schema.domains[a], dtype=object)[self.children[a]]
        return pd.DataFrame(parents), pd.DataFrame(children)

    @classmethod
    def from_frames(cls, schema, parents: pd.DataFrame, children: pd.DataFrame) -> "Coded":
        parent, child = schema.parent, schema.child
        parent_keys = parents[parent.key].astype(str).to_numpy(dtype=object)
        position = {k: i for i, k in enumerate(parent_keys)}
        if len(position) != len(parent_keys):
            raise ValueError(f"{parent.name}: duplicate keys")
        missing = [k for k in children[child.foreign_key].astype(str) if k not in position]
        if missing:
            raise ValueError(f"{child.name}: foreign keys without a parent, e.g. {missing[:5]}")
        fk = np.array([position[k] for k in children[child.foreign_key].astype(str)], dtype=np.int64)
        return cls(
            schema,
            {a: encode(schema, a, parents[a]) for a in schema.parent_attrs},
            {a: encode(schema, a, children[a]) for a in schema.child_attrs},
            fk,
            parent_keys,
            children[child.key].astype(str).to_numpy(dtype=object),
        )


def encode(schema, attribute, values) -> np.ndarray:
    """Codes of ``values`` in the domain of ``attribute``; raises on a value outside it."""
    domain = schema.domains[attribute]
    codes = pd.Categorical(pd.Series(values).astype(str), categories=list(domain)).codes.astype(np.int64)
    if (codes < 0).any():
        bad = sorted(set(pd.Series(values).astype(str)[codes < 0]))[:5]
        raise ValueError(f"values outside the domain of {attribute}: {bad}")
    return codes
