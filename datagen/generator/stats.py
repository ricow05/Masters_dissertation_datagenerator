"""Realised statistics of the clean data, compared with what the spec asked for."""
from __future__ import annotations

import numpy as np

from ..rules import count_violations


def total_variation(p, q) -> float:
    return float(0.5 * np.abs(np.asarray(p, dtype=float) - np.asarray(q, dtype=float)).sum())


def frequencies(codes, size) -> np.ndarray:
    counts = np.bincount(codes, minlength=size).astype(float)
    return counts / counts.sum() if counts.sum() else counts


def cramers_v(x, y, dx, dy) -> float:
    """Cramér's V of two code arrays (0 = independent, 1 = one determines the other)."""
    if len(x) == 0:
        return 0.0
    table = np.bincount(x * dy + y, minlength=dx * dy).reshape(dx, dy).astype(float)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    k = min(table.shape) - 1
    if k <= 0:
        return 0.0
    n = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (n * k)))


def compute(schema, rules, network, target, clean) -> dict:
    """Statistics of the clean data as a JSON-ready dictionary.

    ``target`` holds the marginal of every attribute under the network without rules, so the
    gap to the realised frequencies shows how much the rules (and sampling noise) move them.
    """
    out = {"rows": {schema.parent.name: clean.n_parents, schema.child.name: clean.n_children}}

    fanout = clean.children_per_parent()
    histogram = np.bincount(fanout) if len(fanout) else np.zeros(0, dtype=int)
    out["children_per_parent"] = {
        "mean": float(fanout.mean()) if len(fanout) else 0.0,
        "min": int(fanout.min()) if len(fanout) else 0,
        "max": int(fanout.max()) if len(fanout) else 0,
        "share_0": float((fanout == 0).mean()) if len(fanout) else 0.0,
        "share_1": float((fanout == 1).mean()) if len(fanout) else 0.0,
        "share_2_or_more": float((fanout >= 2).mean()) if len(fanout) else 0.0,
        "histogram": {str(k): int(n) for k, n in enumerate(histogram) if n},
    }

    marginals = {}
    for a in schema.attrs:
        domain = schema.domains[a]
        codes = clean.parents[a] if schema.is_parent_attr(a) else clean.children[a]
        realised = frequencies(codes, len(domain))
        marginals[a] = {
            "target": {v: round(float(p), 6) for v, p in zip(domain, target[a])},
            "realised": {v: round(float(p), 6) for v, p in zip(domain, realised)},
            "tvd": round(total_variation(target[a], realised), 6),
        }
    out["marginals"] = marginals

    joined = clean.joined()
    edges = []
    for on, a in network.edges():
        columns = clean.parents if schema.is_parent_attr(a) else joined
        v = cramers_v(columns[on], columns[a], len(schema.domains[on]), len(schema.domains[a]))
        edges.append({"on": on, "attr": a, "cramers_v": round(v, 6)})
    out["edges"] = edges

    total, per_rule = count_violations(schema, rules, clean)
    out["violations"] = {"clean": total, "clean_per_rule": {k: v for k, v in per_rule.items() if v}}
    return out
