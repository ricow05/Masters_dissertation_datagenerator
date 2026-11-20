"""Error injection with a controllable rate, location, mechanism and detectability.

A *scope* is one attribute of one table. For each eligible scope:

1. The rate is ``rate_by_attribute[attr]``, else ``rate_by_table[table]``, else ``rate``.
2. ``count="exact"`` picks exactly round(rate * rows) cells; ``"bernoulli"`` draws that
   number from Binomial(rows, rate), which makes every cell an independent Bernoulli(rate)
   when no row weights are used.
3. Rows are picked without replacement with probability proportional to their row weight
   (``row_weights``; rows with weight 0 never get an error in that table).
4. With ``detectable_share`` set, round(share * count) cells get a *detectable* value (one that
   makes a target rule fail, judged on the clean data) and the rest a *silent* value (one that
   keeps every rule satisfied). Rows without such a value are not picked. If there are too few,
   the shortfall is reported; ``on_shortfall="fill"`` fills it with the other kind.
5. The wrong value comes from a mechanism drawn with the weights in ``mechanism``:
   ``random`` (any other value of the domain), ``neighbour`` (the next or previous value in
   the domain order) or ``confusion`` (``confusion[attr][clean][dirty]`` weights). A dirty
   value always differs from the clean one.

Keys and foreign keys never get errors. After injection every error is classified again on
the dirty data, with the definition of ``evaluate/metrics.detectable_recall`` in
``algo1_bench``: errors in the same family can mask or expose each other.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..rules import holds, masks
from .spec import ELIGIBLE, MECHANISMS, ErrorSpec, as_list


@dataclass
class Injection:
    dirty: object                 # Coded
    log: pd.DataFrame
    shortfalls: list
    scopes: list                  # (table, attribute, rows, rate, picked)


LOG_COLUMNS = ["table", "key", "attribute", "clean", "dirty", "mechanism", "intended", "detectable"]


# -- validation ----------------------------------------------------------------------------
def eligible_attributes(schema, rules, spec):
    if spec.eligible == "rule_attributes":
        used = {a for r in rules for a in r.attrs}
        return [a for a in schema.attrs if a in used]
    if spec.eligible == "all":
        return list(schema.attrs)
    if isinstance(spec.eligible, str):
        raise ValueError(f"eligible must be one of {ELIGIBLE} or a list of attributes")
    unknown = [a for a in spec.eligible if a not in schema.domains]
    if unknown:
        raise ValueError(f"eligible: unknown attributes {unknown}")
    return [a for a in schema.attrs if a in set(spec.eligible)]


def _validate(schema, rules, spec):
    def check_rate(r, what):
        if not 0.0 <= float(r) <= 1.0:
            raise ValueError(f"{what}: rate must be in [0, 1]")

    check_rate(spec.rate, "rate")
    tables = {t.name for t in schema.tables}
    for name, r in spec.rate_by_table.items():
        if name not in tables:
            raise ValueError(f"rate_by_table: unknown table {name!r}")
        check_rate(r, f"rate_by_table[{name}]")
    for a, r in spec.rate_by_attribute.items():
        if a not in schema.domains:
            raise ValueError(f"rate_by_attribute: unknown attribute {a!r}")
        check_rate(r, f"rate_by_attribute[{a}]")
    if spec.count not in ("exact", "bernoulli"):
        raise ValueError("count must be 'exact' or 'bernoulli'")
    if spec.on_shortfall not in ("skip", "fill"):
        raise ValueError("on_shortfall must be 'skip' or 'fill'")
    unknown = set(spec.mechanism) - set(MECHANISMS)
    if unknown:
        raise ValueError(f"unknown mechanisms {sorted(unknown)}; use {MECHANISMS}")
    weights = [float(w) for w in spec.mechanism.values()]
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("mechanism weights must be non-negative and not all zero")
    for a, table in spec.confusion.items():
        if a not in schema.domains:
            raise ValueError(f"confusion: unknown attribute {a!r}")
        for clean, row in table.items():
            for v in [clean, *row]:
                if v not in schema.domains[a]:
                    raise ValueError(f"confusion[{a}]: {v!r} is not in the domain")
    if spec.detectable_share is not None and not 0.0 <= spec.detectable_share <= 1.0:
        raise ValueError("detectable_share must be in [0, 1] or None")
    if spec.target_rules is not None:
        names = {r.name for r in rules}
        unknown = [n for n in spec.target_rules if n not in names]
        if unknown:
            raise ValueError(f"target_rules: unknown rules {unknown}")
    for rw in spec.row_weights:
        if rw.table not in tables:
            raise ValueError(f"row_weights: unknown table {rw.table!r}")
        if rw.factor < 0:
            raise ValueError("row_weights: factor must be non-negative")
        for a, values in rw.where.items():
            if a not in schema.domains:
                raise ValueError(f"row_weights: unknown attribute {a!r}")
            if rw.table == schema.parent.name and not schema.is_parent_attr(a):
                raise ValueError(f"row_weights on {rw.table}: {a} is not a parent attribute")
            for v in as_list(values):
                if v not in schema.domains[a]:
                    raise ValueError(f"row_weights: {v!r} is not in the domain of {a}")


def rate_of(schema, spec, attribute):
    if attribute in spec.rate_by_attribute:
        return float(spec.rate_by_attribute[attribute])
    return float(spec.rate_by_table.get(schema.table_of(attribute).name, spec.rate))


# -- candidate values ----------------------------------------------------------------------
def row_weights(schema, spec, clean, table):
    is_parent = table == schema.parent.name
    columns = clean.parents if is_parent else clean.joined()
    n = clean.n_parents if is_parent else clean.n_children
    weights = np.ones(n)
    for rw in spec.row_weights:
        if rw.table != table:
            continue
        match = np.ones(n, dtype=bool)
        for a, values in rw.where.items():
            mask = np.array([v in set(as_list(values)) for v in schema.domains[a]])
            match &= mask[columns[a]]
        weights[match] *= rw.factor
    return weights


def failing_values(schema, rules, clean, attribute):
    """Boolean matrices (rows x domain) for one scope, judged on the clean data.

    ``fails[r]`` marks the values that make rule ``r`` fail when written into the cell, with
    every other cell clean. Only rules that involve the attribute can start failing.
    """
    is_parent = schema.is_parent_attr(attribute)
    n = clean.n_parents if is_parent else clean.n_children
    out = {}
    joined = None if is_parent else clean.joined()
    for rule in rules:
        if not rule.involves(attribute):
            continue
        mask = masks(schema, rule)[attribute]
        if is_parent:
            child_side = [a for a in rule.attrs if not schema.is_parent_attr(a)]
            other = np.broadcast_to(holds(schema, rule, clean.parents, skip=[attribute, *child_side]), (n,))
            if child_side:
                parent_side = [a for a in rule.attrs if schema.is_parent_attr(a)]
                child_part = np.broadcast_to(holds(schema, rule, clean.children, skip=parent_side), (clean.n_children,))
                other = other & (np.bincount(clean.fk[child_part], minlength=n) > 0)
        else:
            other = np.broadcast_to(holds(schema, rule, joined, skip=[attribute]), (n,))
        out[rule.name or str(rule)] = other[:, None] & mask[None, :]
    return out


def mechanism_candidates(schema, spec, attribute, clean_codes):
    """For each mechanism with positive weight: allowed dirty values (rows x domain) and value weights."""
    d = len(schema.domains[attribute])
    values = np.arange(d)[None, :]
    differs = values != clean_codes[:, None]
    out = {}
    for name in MECHANISMS:
        w = float(spec.mechanism.get(name, 0.0))
        if w <= 0:
            continue
        if name == "random":
            allowed = differs
            weights = differs.astype(float)
        elif name == "neighbour":
            allowed = np.abs(values - clean_codes[:, None]) == 1
            weights = allowed.astype(float)
        else:
            table = np.zeros((d, d))
            domain = schema.domains[attribute]
            for clean, row in spec.confusion.get(attribute, {}).items():
                for dirty, weight in row.items():
                    table[domain.index(clean), domain.index(dirty)] = float(weight)
            np.fill_diagonal(table, 0.0)
            weights = table[clean_codes]
            allowed = weights > 0
        out[name] = (w, allowed, weights)
    return out


# -- sampling helpers ----------------------------------------------------------------------
def _pick(rng, rows, k, weights):
    """Up to ``k`` of ``rows`` without replacement, proportional to ``weights[rows]``."""
    if k <= 0 or len(rows) == 0:
        return rows[:0]
    if k >= len(rows):
        return rows
    p = weights[rows]
    return np.sort(rng.choice(rows, size=k, replace=False, p=p / p.sum()))


def _draw_index(rng, weights):
    """One column per row of ``weights``, proportional to the row's weights."""
    cdf = np.cumsum(weights, axis=1)
    u = rng.random(len(weights)) * cdf[:, -1]
    return np.minimum((cdf <= u[:, None]).sum(axis=1), weights.shape[1] - 1)


def _round(x):
    return int(math.floor(x + 0.5))


# -- injection -----------------------------------------------------------------------------
def inject_errors(schema, rules, clean, spec: ErrorSpec, rng) -> Injection:
    """Inject errors into a copy of ``clean`` (a ``Coded``), with ``rng`` a numpy Generator or a seed.

    Works on any clean parent-child data with its schema and rules, not only generated data.
    """
    rng = np.random.default_rng(rng)
    _validate(schema, rules, spec)
    dirty = clean.copy()
    targets = rules if spec.target_rules is None else [r for r in rules if r.name in set(spec.target_rules)]
    target_names = {r.name or str(r) for r in targets}
    weights_of = {t.name: row_weights(schema, spec, clean, t.name) for t in schema.tables}
    records, shortfalls, scopes = [], [], []

    for attribute in eligible_attributes(schema, rules, spec):
        table = schema.table_of(attribute).name
        codes = clean.column(table, attribute)
        n = len(codes)
        rate = rate_of(schema, spec, attribute)
        if spec.count == "exact":
            k = _round(rate * n)
        else:
            k = int(rng.binomial(n, rate)) if n else 0
        weights = weights_of[table]
        mechanisms = mechanism_candidates(schema, spec, attribute, codes)
        allowed = np.zeros((n, len(schema.domains[attribute])), dtype=bool)
        for _, a, _ in mechanisms.values():
            allowed |= a
        usable = weights > 0

        if spec.detectable_share is None:
            kinds = {"any": allowed}
            want = {"any": k}
        else:
            fails = failing_values(schema, rules, clean, attribute)
            any_fail = np.zeros_like(allowed)
            target_fail = np.zeros_like(allowed)
            for name, matrix in fails.items():
                any_fail |= matrix
                if name in target_names:
                    target_fail |= matrix
            kinds = {"detectable": allowed & target_fail, "silent": allowed & ~any_fail}
            k_det = _round(spec.detectable_share * k) if spec.count == "exact" else int(rng.binomial(k, spec.detectable_share))
            want = {"detectable": k_det, "silent": k - k_det}

        taken = np.zeros(n, dtype=bool)
        chosen = {}
        for kind, matrix in kinds.items():
            rows = np.flatnonzero(matrix.any(axis=1) & usable & ~taken)
            chosen[kind] = _pick(rng, rows, want[kind], weights)
            taken[chosen[kind]] = True
            if len(chosen[kind]) < want[kind]:
                shortfalls.append(
                    {"table": table, "attribute": attribute, "kind": kind, "wanted": want[kind], "got": len(chosen[kind])}
                )
        if spec.on_shortfall == "fill" and len(kinds) == 2:
            missing = sum(want[kind] - len(chosen[kind]) for kind in kinds)
            for kind, matrix in kinds.items():
                if missing <= 0:
                    break
                rows = np.flatnonzero(matrix.any(axis=1) & usable & ~taken)
                extra = _pick(rng, rows, missing, weights)
                taken[extra] = True
                chosen[kind] = np.sort(np.concatenate([chosen[kind], extra]))
                missing -= len(extra)

        picked = 0
        for kind, rows in chosen.items():
            if len(rows) == 0:
                continue
            kind_mask = kinds[kind][rows]
            names = list(mechanisms)
            options = np.stack(
                [w * (allowed_m[rows] & kind_mask).any(axis=1) for w, allowed_m, _ in mechanisms.values()], axis=1
            )
            which = _draw_index(rng, options)
            new = np.empty(len(rows), dtype=np.int64)
            for m, name in enumerate(names):
                sel = which == m
                if not sel.any():
                    continue
                _, allowed_m, value_weights = mechanisms[name]
                value_weights = value_weights[rows[sel]] * (allowed_m[rows[sel]] & kind_mask[sel])
                new[sel] = _draw_index(rng, value_weights)
            assert (new != codes[rows]).all()
            dirty.column(table, attribute)[rows] = new
            keys = (clean.parent_keys if table == schema.parent.name else clean.child_keys)[rows]
            domain = np.asarray(schema.domains[attribute], dtype=object)
            records.append(
                pd.DataFrame(
                    {
                        "table": table,
                        "key": keys,
                        "attribute": attribute,
                        "clean": domain[codes[rows]],
                        "dirty": domain[new],
                        "mechanism": np.asarray(names, dtype=object)[which],
                        "intended": kind,
                        "_row": rows,
                    }
                )
            )
            picked += len(rows)
        scopes.append({"table": table, "attribute": attribute, "rows": n, "rate": rate, "target": k, "errors": picked})

    log = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=LOG_COLUMNS + ["_row"])
    if len(log):
        order = {a: i for i, a in enumerate(schema.attrs)}
        log = log.assign(_order=log["attribute"].map(order)).sort_values(["_order", "_row"], kind="stable")
        log = log.drop(columns="_order").reset_index(drop=True)
    log["detectable"] = classify(schema, rules, dirty, log)
    log = log.drop(columns="_row")[LOG_COLUMNS]
    return Injection(dirty, log, shortfalls, scopes)


def classify(schema, rules, dirty, log) -> np.ndarray:
    """Whether each logged error is detectable, as in ``algo1_bench`` ``detectable_recall``.

    A parent error on ``a`` is detectable when a rule involving ``a`` fails on the dirty parent
    (parent rules) or on the dirty parent joined with any of its dirty children (other rules).
    A child error on ``a`` is detectable when a child or parent-child rule involving ``a``
    fails on the dirty parent joined with that dirty child.
    """
    if len(log) == 0:
        return np.zeros(0, dtype=bool)
    joined = dirty.joined()
    on_parent, on_child = {}, {}
    for rule in rules:
        if schema.kind(rule) == "parent":
            on_parent[rule] = np.broadcast_to(holds(schema, rule, dirty.parents), (dirty.n_parents,))
        else:
            fails = np.broadcast_to(holds(schema, rule, joined), (dirty.n_children,))
            on_child[rule] = fails
            on_parent[rule] = np.bincount(dirty.fk[fails], minlength=dirty.n_parents) > 0
    out = np.zeros(len(log), dtype=bool)
    rows = log["_row"].to_numpy()
    is_parent = (log["table"] == schema.parent.name).to_numpy()
    attributes = log["attribute"].to_numpy()
    for attribute in np.unique(attributes):
        for rule in rules:
            if not rule.involves(attribute):
                continue
            sel = attributes == attribute
            if schema.is_parent_attr(attribute):
                out[sel & is_parent] |= on_parent[rule][rows[sel & is_parent]]
            elif rule in on_child:
                out[sel & ~is_parent] |= on_child[rule][rows[sel & ~is_parent]]
    return out
