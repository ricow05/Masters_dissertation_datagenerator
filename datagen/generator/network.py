"""Bayesian network over the attributes of both tables.

Every attribute has one conditional table P(attr | its network parents). An attribute
without a dependency uses its marginal. Edges may run inside a table or from a parent-table
attribute to a child-table attribute, never from child to parent: a child row is drawn
given its parent row, not the other way round.
"""
from __future__ import annotations

import numpy as np

from .distributions import Uniform


class Network:
    def __init__(self, schema, marginals=None, dependencies=()):
        self.schema = schema
        self.marginals = dict(marginals or {})
        self.dependencies = list(dependencies)
        unknown = set(self.marginals) - set(schema.domains)
        if unknown:
            raise ValueError(f"marginals for unknown attributes: {sorted(unknown)}")

        self.parents: dict[str, tuple[str, ...]] = {a: () for a in schema.attrs}
        self.tables: dict[str, np.ndarray] = {}
        seen = set()
        for dep in self.dependencies:
            if dep.attr not in schema.domains:
                raise ValueError(f"dependency on unknown attribute {dep.attr!r}")
            if dep.attr in seen:
                raise ValueError(f"{dep.attr}: more than one dependency")
            seen.add(dep.attr)
            for on in dep.parents:
                if on not in schema.domains:
                    raise ValueError(f"{dep.attr}: depends on unknown attribute {on!r}")
                if on == dep.attr:
                    raise ValueError(f"{dep.attr}: depends on itself")
                if schema.is_parent_attr(dep.attr) and not schema.is_parent_attr(on):
                    raise ValueError(
                        f"{dep.attr} is in the parent table and cannot depend on child attribute {on}"
                    )
            self.parents[dep.attr] = tuple(dep.parents)

        self.order = self._topological_order()
        for attribute in schema.attrs:
            base = self.marginal(attribute)
            dep = next((d for d in self.dependencies if d.attr == attribute), None)
            table = base if dep is None else dep.table(schema.domains, base)
            if not np.allclose(table.sum(axis=-1), 1.0):
                raise ValueError(f"{attribute}: conditional distribution does not sum to one")
            self.tables[attribute] = table

    def marginal(self, attribute) -> np.ndarray:
        """The attribute's own frequency table (before any dependency)."""
        return self.marginals.get(attribute, Uniform()).probabilities(self.schema.domains[attribute])

    def _topological_order(self):
        remaining = {a: set(p) for a, p in self.parents.items()}
        order = []
        while remaining:
            ready = [a for a in self.schema.attrs if a in remaining and not remaining[a]]
            if not ready:
                raise ValueError(f"the dependencies contain a cycle among {sorted(remaining)}")
            for a in ready:
                order.append(a)
                del remaining[a]
            for parents in remaining.values():
                parents.difference_update(ready)
        return order

    def edges(self):
        """Declared (on, attr) pairs."""
        return [(on, a) for a in self.schema.attrs for on in self.parents[a]]

    def probability(self, attrs, columns, fixed=None):
        """P(attrs | their network parents) for every row.

        ``columns`` maps attributes to code arrays of equal length; ``fixed`` maps
        attributes to a single code. Every network parent of an attribute in ``attrs`` must
        be in one of the two.
        """
        fixed = fixed or {}
        length = len(next(iter(columns.values()))) if columns else 1
        out = np.ones(length)
        for attribute in attrs:
            index = []
            for source in self.parents[attribute] + (attribute,):
                if source in columns:
                    index.append(columns[source])
                elif source in fixed:
                    index.append(fixed[source])
                else:
                    raise KeyError(f"{attribute}: network parent {source} is not given")
            out = out * self.tables[attribute][tuple(index)]
        return out
