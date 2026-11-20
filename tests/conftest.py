"""Shared helpers. The checks here are written in plain Python on the string tables, so they
do not reuse the vectorised code they test."""
from __future__ import annotations

import pytest

from datagen import Rule, two_tables


def rows(frame):
    return frame.to_dict("records")


def slow_violations(schema, rules, parents, children):
    """(tuple, rule) violations, counted like algo1_bench gendb/validate.violations."""
    parent_rules = [r for r in rules if schema.kind(r) == "parent"]
    joint_rules = [r for r in rules if schema.kind(r) != "parent"]
    by_key = {p[schema.parent.key]: p for p in rows(parents)}
    count = sum(r.fails(p) for p in by_key.values() for r in parent_rules)
    for child in rows(children):
        joined = {**by_key[child[schema.child.foreign_key]], **child}
        count += sum(r.fails(joined) for r in joint_rules)
    return count


def slow_detectable(schema, rules, frames, errors):
    """detectable_recall's notion of a detectable error, per row of the error log."""
    p, c = schema.parent, schema.child
    parents = {r[p.key]: r for r in rows(frames[f"{p.name}_dirty"])}
    children = {r[c.key]: r for r in rows(frames[f"{c.name}_dirty"])}
    kids = {}
    for child in children.values():
        kids.setdefault(child[c.foreign_key], []).append(child)
    joint = [r for r in rules if schema.kind(r) != "parent"]
    parent_rules = [r for r in rules if schema.kind(r) == "parent"]
    out = []
    for e in rows(errors):
        if e["table"] == p.name:
            parent = parents[e["key"]]
            violated = [r for r in parent_rules if r.fails(parent)]
            for child in kids.get(e["key"], []):
                violated += [r for r in joint if r.fails({**parent, **child})]
        else:
            child = children[e["key"]]
            parent = parents[child[c.foreign_key]]
            violated = [r for r in joint if r.fails({**parent, **child})]
        out.append(any(r.involves(e["attribute"]) for r in violated))
    return out


@pytest.fixture
def tiny_schema():
    """Parent x (3 values), y (2); child u (3), w (2)."""
    return two_tables(
        "P", "pid", {"x": ("a", "b", "c"), "y": ("0", "1")},
        "C", "cid", {"u": ("0", "1", "2"), "w": ("lo", "hi")},
    )


@pytest.fixture
def tiny_rules():
    """x=c is allowed for a parent but leaves no valid child (an implied parent rule)."""
    return [
        Rule.of({"x": "a", "y": "1"}, "P1"),
        Rule.of({"u": "2", "w": "hi"}, "C1"),
        Rule.of({"x": "b", "u": "0"}, "PC1"),
        Rule.of({"x": "c", "u": "0"}, "PC2"),
        Rule.of({"x": "c", "u": ["1", "2"]}, "PC3"),
        Rule.of({"y": "0", "w": "lo", "u": "1"}, "PC4"),
    ]
