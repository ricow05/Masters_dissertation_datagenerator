"""Learn a spec from real parent and child tables.

* Domains come from the schema, or are inferred from the data (sorted distinct values;
  columns with more than ``max_domain`` values are left out and reported).
* Frequency tables and conditional tables are counts plus ``alpha`` (Laplace smoothing).
  Parent configurations never seen in the data fall back to the attribute's marginal.
* Structure: ``"chow-liu"`` (default) builds the maximum mutual-information tree. Parent
  attributes form a tree of their own (mutual information on the parent rows); child
  attributes are then attached to any attribute already in the tree (mutual information on
  the joined rows). Edges point from the root outwards, so they never run child -> parent.
  An edge is kept only when a G-test (G = 2 N MI, chi-squared with (|A|-1)(|B|-1) degrees of
  freedom) rejects independence at ``significance``. ``"none"`` gives independent
  attributes; a list of ``(on, attr)`` pairs or ``{"attr": ..., "on": [...]}`` fixes the
  structure.
* Children per parent becomes an ``Empirical`` fan-out, including parents without children.
* Rules are not learned. Given rules are checked on the input and the violations reported;
  the generator conditions on them anyway.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import Coded
from .distributions import CPT, Frequencies
from .fanout import Empirical
from ..rules import check_rules, rule_fails
from ..schema import Schema, Table
from .spec import DatasetSpec

MISSING = "<missing>"


@dataclass
class FitResult:
    spec: DatasetSpec
    report: dict


def chi2_sf(x, df) -> float:
    """P(X > x) for X ~ chi-squared(df), by the Wilson-Hilferty approximation."""
    if df <= 0 or x <= 0:
        return 1.0
    h = 2.0 / (9.0 * df)
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - h)) / math.sqrt(h)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def mutual_information(x, y, dx, dy) -> float:
    """Plug-in mutual information (nats) of two code arrays."""
    n = len(x)
    if n == 0:
        return 0.0
    joint = np.bincount(x * dy + y, minlength=dx * dy).reshape(dx, dy) / n
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / np.outer(px, py)[nz])).sum())


# -- schema -----------------------------------------------------------------------------
def _clean_column(series):
    return series.astype(object).where(series.notna(), MISSING).astype(str)


def infer_schema(parents, children, fk, parent_key=None, child_key=None,
                 parent_name="Parent", child_name="Child", max_domain=50):
    """A schema from raw tables; returns ``(schema, renamed_child_columns, report)``."""
    parent_key = parent_key or fk
    if parent_key not in parents:
        raise ValueError(f"parent key column {parent_key!r} not found")
    if fk not in children:
        raise ValueError(f"foreign key column {fk!r} not found in the child table")
    if child_key is None:
        child_key = next(
            (c for c in children.columns if c != fk and children[c].is_unique and children[c].notna().all()), None
        )
    report = {"dropped_columns": {}, "renamed_columns": {}, "missing_values": {}}

    def domains(frame, skip):
        out = {}
        for column in frame.columns:
            if column in skip:
                continue
            values = _clean_column(frame[column])
            distinct = sorted(values.unique())
            if len(distinct) > max_domain:
                report["dropped_columns"][column] = f"{len(distinct)} distinct values > max_domain={max_domain}"
                continue
            if (values == MISSING).any():
                report["missing_values"][column] = int((values == MISSING).sum())
            out[column] = tuple(distinct)
        return out

    parent_domains = domains(parents, {parent_key})
    child_domains = domains(children, {fk, child_key})
    renamed = {}
    reserved = {parent_key, fk, child_key or f"{child_name.lower()}_id"}
    for column in list(child_domains):
        if column in parent_domains or column in reserved:
            new = f"{column}_{child_name.lower()}"
            renamed[column] = new
            child_domains[new] = child_domains.pop(column)
    report["renamed_columns"] = renamed
    schema = Schema(
        [
            Table(parent_name, parent_key, parent_domains),
            Table(child_name, child_key or f"{child_name.lower()}_id", child_domains, parent=parent_name, foreign_key=fk),
        ]
    )
    return schema, renamed, report


def _coded(schema, parents, children, renamed):
    p, c = schema.parent, schema.child
    parents = parents.copy()
    children = children.rename(columns=renamed).copy()
    keys = parents[p.key].astype(str)
    if not keys.is_unique:
        raise ValueError(f"{p.name}: duplicate keys")
    fk = children[c.foreign_key].astype(str)
    orphans = ~fk.isin(set(keys))
    children = children[~orphans.to_numpy()]
    for frame, attrs in ((parents, schema.parent_attrs), (children, schema.child_attrs)):
        for a in attrs:
            frame[a] = _clean_column(frame[a])
    if c.key not in children:
        children[c.key] = np.arange(len(children)).astype(str)
    return Coded.from_frames(schema, parents, children), int(orphans.sum())


# -- structure ----------------------------------------------------------------------------
def chow_liu(schema, data):
    """Maximum mutual-information tree, parent attributes first; returns ``[(on, attr, mi, n)]``."""
    joined = data.joined()
    sizes = {a: len(schema.domains[a]) for a in schema.attrs}
    cache = {}

    def mi(u, v):
        if (u, v) not in cache:
            both_parent = schema.is_parent_attr(u) and schema.is_parent_attr(v)
            columns = data.parents if both_parent else joined
            cache[(u, v)] = cache[(v, u)] = (
                mutual_information(columns[u], columns[v], sizes[u], sizes[v]),
                data.n_parents if both_parent else data.n_children,
            )
        return cache[(u, v)]

    edges = []
    tree = [schema.parent_attrs[0]]
    for group in (schema.parent_attrs, schema.child_attrs):
        todo = [a for a in group if a not in tree]
        while todo:
            value, u, v = max(((mi(u, v)[0], u, v) for u in tree for v in todo), key=lambda t: t[0])
            edges.append((u, v, value, mi(u, v)[1]))
            tree.append(v)
            todo.remove(v)
    return edges


def _parse_structure(schema, structure):
    if structure in (None, "none"):
        return {}
    parents = {}
    for item in structure:
        if isinstance(item, dict):
            attr, on = item["attr"], item["on"]
            on = [on] if isinstance(on, str) else list(on)
        else:
            on, attr = item
            on = [on]
        parents.setdefault(attr, [])
        parents[attr] += [o for o in on if o not in parents[attr]]
    for attr, on in parents.items():
        for o in on:
            if o not in schema.domains or attr not in schema.domains:
                raise ValueError(f"structure: unknown attribute in edge {o} -> {attr}")
    return parents


# -- main --------------------------------------------------------------------------------
def fit(parents: pd.DataFrame, children: pd.DataFrame, fk: str, schema: Schema | None = None, rules=(),
        structure="chow-liu", alpha=1.0, significance=1e-3, parent_key=None, child_key=None,
        parent_name="Parent", child_name="Child", max_domain=50, rows=None, seed=0) -> FitResult:
    """Learn a ``DatasetSpec`` from a parent and a child table linked by ``fk``."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if schema is None:
        schema, renamed, report = infer_schema(parents, children, fk, parent_key, child_key, parent_name, child_name, max_domain)
    else:
        renamed, report = {}, {"dropped_columns": {}, "renamed_columns": {}, "missing_values": {}}
    rules = list(rules)
    check_rules(schema, rules)
    data, orphans = _coded(schema, parents, children, renamed)
    report["rows"] = {schema.parent.name: data.n_parents, schema.child.name: data.n_children}
    report["orphan_children_dropped"] = orphans

    marginals = {}
    for a in schema.attrs:
        codes = data.parents[a] if schema.is_parent_attr(a) else data.children[a]
        counts = np.bincount(codes, minlength=len(schema.domains[a])) + alpha
        if counts.sum() == 0:
            counts = np.ones_like(counts)
        marginals[a] = Frequencies({v: float(n) for v, n in zip(schema.domains[a], counts)})

    edge_report = []
    if structure == "chow-liu":
        network_parents = {}
        for on, attr, mi, n in chow_liu(schema, data):
            df = (len(schema.domains[on]) - 1) * (len(schema.domains[attr]) - 1)
            g = 2.0 * n * mi
            p = chi2_sf(g, df)
            kept = p < significance
            edge_report.append({"on": on, "attr": attr, "mi": round(mi, 6), "g": round(g, 3), "df": df,
                                "p_value": p, "kept": bool(kept)})
            if kept:
                network_parents[attr] = [on]
    else:
        network_parents = _parse_structure(schema, structure)
        edge_report = [{"on": o, "attr": a, "kept": True} for a, on in network_parents.items() for o in on]
    report["edges"] = edge_report

    joined = data.joined()
    dependencies = []
    for attr in schema.attrs:
        on = network_parents.get(attr)
        if not on:
            continue
        columns = data.parents if all(schema.is_parent_attr(x) for x in [attr, *on]) else joined
        sizes = [len(schema.domains[x]) for x in on] + [len(schema.domains[attr])]
        flat = np.ravel_multi_index([columns[x] for x in on] + [columns[attr]], sizes)
        counts = np.bincount(flat, minlength=int(np.prod(sizes))).reshape(sizes)
        rows_ = {}
        for index in np.ndindex(*sizes[:-1]):
            row = counts[index]
            if row.sum() == 0:
                continue
            given = tuple(schema.domains[x][i] for x, i in zip(on, index))
            rows_[given] = {v: float(n + alpha) for v, n in zip(schema.domains[attr], row)}
        dependencies.append(CPT(attr, tuple(on), rows_))

    fanout_counts = np.bincount(data.children_per_parent())
    fanout = Empirical({k: float(n) for k, n in enumerate(fanout_counts) if n})

    if rules:
        per_rule = {r.name: int(rule_fails(schema, r, data).sum()) for r in rules}
        parent_rows = np.zeros(data.n_parents, dtype=bool)
        joined_rows = np.zeros(data.n_children, dtype=bool)
        for r in rules:
            if schema.kind(r) == "parent":
                parent_rows |= rule_fails(schema, r, data)
            else:
                joined_rows |= rule_fails(schema, r, data)
        report["rule_violations"] = {
            "total": sum(per_rule.values()),
            "per_rule": per_rule,
            "parent_rows_violating": int(parent_rows.sum()),
            "joined_rows_violating": int(joined_rows.sum()),
        }

    spec = DatasetSpec(
        schema=schema,
        rules=rules,
        marginals=marginals,
        dependencies=dependencies,
        fanout=fanout,
        parents=int(rows if rows is not None else data.n_parents),
        seed=seed,
    )
    return FitResult(spec, report)
