"""Statistics of the injected errors and of the dirty data."""
from __future__ import annotations

from ..rules import count_violations


def _rate(errors, cells):
    return {"cells": int(cells), "errors": int(errors), "rate": errors / cells if cells else 0.0}


def compute(schema, rules, clean, dirty, log, eligible, shortfalls=()) -> dict:
    """``violations`` of the dirty data and the ``errors``, as a JSON-ready dictionary."""
    total, per_rule = count_violations(schema, rules, dirty)
    out = {"violations": {"dirty": total, "dirty_per_rule": per_rule}}

    by_table, by_attribute = {}, {}
    total_errors = total_cells = 0
    for a in eligible:
        table = schema.table_of(a).name
        before = clean.parents[a] if schema.is_parent_attr(a) else clean.children[a]
        after = dirty.parents[a] if schema.is_parent_attr(a) else dirty.children[a]
        errors, cells = int((before != after).sum()), len(before)
        by_attribute[a] = _rate(errors, cells)
        t = by_table.setdefault(table, [0, 0])
        t[0] += errors
        t[1] += cells
        total_errors += errors
        total_cells += cells
    detectable = int(log["detectable"].sum()) if len(log) else 0
    out["errors"] = {
        **_rate(total_errors, total_cells),
        "eligible_attributes": list(eligible),
        "by_table": {t: _rate(e, c) for t, (e, c) in by_table.items()},
        "by_attribute": by_attribute,
        "detectable": detectable,
        "detectable_share": detectable / len(log) if len(log) else 0.0,
        "by_mechanism": {k: int(v) for k, v in log["mechanism"].value_counts().sort_index().items()} if len(log) else {},
        "by_intended": {k: int(v) for k, v in log["intended"].value_counts().sort_index().items()} if len(log) else {},
        "shortfalls": list(shortfalls),
    }
    return out
