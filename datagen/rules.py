"""Constant selection rules: NOT(a1 in S1 AND a2 in S2 AND ...).

A rule *fails* on a row when every atom holds. A rule over attributes of one table is
checked per row of that table; a parent-child rule is checked on every joined
(parent, child) row.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rule:
    atoms: tuple[tuple[str, frozenset[str]], ...]
    name: str = ""
    meaning: str = ""

    @classmethod
    def of(cls, atoms: dict, name="", meaning=""):
        """Build a rule from ``{attribute: value or collection of values}``."""
        if not atoms:
            raise ValueError(f"{name or 'rule'}: no atoms")
        parsed = []
        for attribute, values in atoms.items():
            if isinstance(values, str):
                values = {values}
            values = frozenset(str(v) for v in values)
            if not values:
                raise ValueError(f"{name or 'rule'}: empty value set on {attribute}")
            parsed.append((str(attribute), values))
        return cls(tuple(parsed), name, meaning)

    @property
    def attrs(self) -> list[str]:
        return [a for a, _ in self.atoms]

    def values(self, attribute) -> frozenset[str]:
        for a, values in self.atoms:
            if a == attribute:
                return values
        raise KeyError(attribute)

    def involves(self, attribute) -> bool:
        return any(a == attribute for a, _ in self.atoms)

    def fails(self, row) -> bool:
        """True when ``row`` (a mapping attribute -> value) violates the rule."""
        return all(row[a] in values for a, values in self.atoms)

    def __str__(self):
        parts = []
        for a, values in self.atoms:
            if len(values) == 1:
                parts.append(f"{a} == {next(iter(values))}")
            else:
                parts.append(f"{a} in {{{', '.join(sorted(values))}}}")
        return "NOT " + " & ".join(parts)

    def to_dict(self):
        out = {"name": self.name, "atoms": {a: sorted(v) for a, v in self.atoms}}
        if self.meaning:
            out["meaning"] = self.meaning
        return out

    @classmethod
    def from_dict(cls, data):
        return cls.of(data["atoms"], data.get("name", ""), data.get("meaning", ""))


def check_rules(schema, rules):
    """Raise if a rule names an unknown attribute or value, or two rules share a name."""
    names = [r.name for r in rules if r.name]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate rule names: {sorted(n for n in set(names) if names.count(n) > 1)}")
    for rule in rules:
        if len(set(rule.attrs)) != len(rule.attrs):
            raise ValueError(f"{rule.name}: an attribute appears twice")
        for attribute, values in rule.atoms:
            if attribute not in schema.domains:
                raise ValueError(f"{rule.name}: unknown attribute {attribute!r}")
            for value in values:
                if value not in schema.domains[attribute]:
                    raise ValueError(f"{rule.name}: {value!r} is not in the domain of {attribute}")


def masks(schema, rule):
    """For every atom, a boolean array over the attribute's domain indices."""
    out = {}
    for attribute, values in rule.atoms:
        domain = schema.domains[attribute]
        out[attribute] = np.array([v in values for v in domain], dtype=bool)
    return out


def holds(schema, rule, columns, skip=()):
    """Rows (of ``columns``: attribute -> code array) on which every atom outside ``skip`` holds.

    Attributes missing from ``columns`` must be in ``skip``. Returns ``True`` (a scalar)
    when every atom is skipped.
    """
    result = True
    for attribute, mask in masks(schema, rule).items():
        if attribute in skip:
            continue
        result = result & mask[columns[attribute]]
    return result


def rule_fails(schema, rule, data):
    """Where ``rule`` fails on coded data.

    Returns a boolean array over the parent rows for a parent rule, and over the child
    rows (joined with their parent) for a child or parent-child rule.
    """
    if schema.kind(rule) == "parent":
        return np.broadcast_to(holds(schema, rule, data.parents), (data.n_parents,)).copy()
    joined = data.joined()
    return np.broadcast_to(holds(schema, rule, joined), (data.n_children,)).copy()


def count_violations(schema, rules, data):
    """(tuple, rule) violations as counted by ``algo1_bench/gendb/validate.violations``.

    Returns ``(total, {rule name: count})``.
    """
    per_rule = {}
    for rule in rules:
        per_rule[rule.name or str(rule)] = int(rule_fails(schema, rule, data).sum())
    return sum(per_rule.values()), per_rule
