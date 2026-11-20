"""Clean data drawn from the Bayesian network, conditioned exactly on the rules.

The attributes are split into *blocks*: connected components of the graph that links two
attributes when they appear in the same rule or are joined by a network edge. Blocks share
no rule and no edge, so they can be drawn independently. Without network edges the blocks
are the attribute classes of Algorithm 1.

Within a block:

1. Every combination of the block's parent attributes is enumerated (``P``), and so is every
   combination of its child attributes (``C``).
2. A parent combination is *valid* when it passes the parent rules. A parent combination is
   *feasible* when at least one child combination passes the child and parent-child rules
   with it, with positive probability. Which child combinations pass depends on the parent
   only through its *context*: which parent-child rules have their parent part satisfied,
   and the values of the parent attributes that child attributes depend on in the network.
3. ``sequential`` conditioning (default) draws a parent with probability proportional to
   P(p) * valid(p), times feasible(p) when the parent gets at least one child. ``joint``
   conditioning draws it proportional to P(p) * valid(p) * Z(p)^n, where Z(p) is the
   probability that one child drawn from the network passes the rules with p and n is the
   number of children; this is the distribution that rejection sampling of whole families
   would give.
4. Each child is drawn from P(c | p) restricted to the child combinations that pass the rules
   with its parent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import Coded
from ..rules import holds

CONDITIONING = ("sequential", "joint")


class CombinationLimit(ValueError):
    pass


@dataclass
class Block:
    parent_attrs: list[str]
    child_attrs: list[str]
    rules: list

    @property
    def attrs(self):
        return self.parent_attrs + self.child_attrs

    def __str__(self):
        return "{" + ", ".join(self.attrs) + "}"


def find_blocks(schema, rules, network) -> list[Block]:
    root = {a: a for a in schema.attrs}

    def find(a):
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            root[rb] = ra

    for rule in rules:
        for a in rule.attrs[1:]:
            union(rule.attrs[0], a)
    for on, a in network.edges():
        union(on, a)

    groups: dict[str, list[str]] = {}
    for a in schema.attrs:
        groups.setdefault(find(a), []).append(a)
    blocks = []
    for members in groups.values():
        top = find(members[0])
        blocks.append(
            Block(
                [a for a in members if schema.is_parent_attr(a)],
                [a for a in members if not schema.is_parent_attr(a)],
                [r for r in rules if find(r.attrs[0]) == top],
            )
        )
    return blocks


def cartesian(sizes) -> np.ndarray:
    """All combinations of ``range(s) for s in sizes``, the first column varying slowest."""
    if not sizes:
        return np.zeros((1, 0), dtype=np.int64)
    return np.indices(sizes, dtype=np.int64).reshape(len(sizes), -1).T.copy()


def grouped(labels):
    """Yield ``(label, row indices)`` for every distinct label, in sorted order."""
    order = np.argsort(labels, kind="stable")
    values, starts = np.unique(labels[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    for value, start, end in zip(values, starts, ends):
        yield value, order[start:end]


class BlockModel:
    def __init__(self, schema, network, block, max_combinations):
        self.schema, self.network, self.block = schema, network, block
        p_sizes = [len(schema.domains[a]) for a in block.parent_attrs]
        c_sizes = [len(schema.domains[a]) for a in block.child_attrs]
        for label, sizes in (("parent", p_sizes), ("child", c_sizes)):
            total = int(np.prod(sizes, dtype=float)) if sizes else 1
            if total > max_combinations:
                raise CombinationLimit(
                    f"block {block}: {total:,} {label} combinations exceed max_combinations="
                    f"{max_combinations:,}; split the block by removing rules or dependencies"
                )
        self.P, self.C = cartesian(p_sizes), cartesian(c_sizes)
        self.pcols = {a: self.P[:, i] for i, a in enumerate(block.parent_attrs)}
        self.ccols = {a: self.C[:, j] for j, a in enumerate(block.child_attrs)}
        n_p, n_c = len(self.P), len(self.C)

        self.prob_p = network.probability(block.parent_attrs, self.pcols)
        self.valid_p = np.ones(n_p, dtype=bool)
        self.valid_c = np.ones(n_c, dtype=bool)
        hp, hc = [], []
        for rule in block.rules:
            kind = schema.kind(rule)
            if kind == "parent":
                self.valid_p &= ~np.broadcast_to(holds(schema, rule, self.pcols), (n_p,))
            elif kind == "child":
                self.valid_c &= ~np.broadcast_to(holds(schema, rule, self.ccols), (n_c,))
            else:
                parent_side = [a for a in rule.attrs if schema.is_parent_attr(a)]
                child_side = [a for a in rule.attrs if not schema.is_parent_attr(a)]
                hp.append(np.broadcast_to(holds(schema, rule, self.pcols, skip=child_side), (n_p,)))
                hc.append(np.broadcast_to(holds(schema, rule, self.ccols, skip=parent_side), (n_c,)))
        self.n_joint = len(hp)
        self.hc = np.stack(hc, axis=1) if hc else np.zeros((n_c, 0), dtype=bool)

        # Context of a parent combination: parent parts of the parent-child rules that hold,
        # and the values of the parent attributes that child attributes depend on.
        self.cross = [a for a in block.parent_attrs if any(a in network.parents[c] for c in block.child_attrs)]
        key = [np.stack(hp, axis=1).astype(np.int64)] if hp else []
        key += [self.pcols[a][:, None] for a in self.cross]
        if key:
            self.contexts, inverse = np.unique(np.concatenate(key, axis=1), axis=0, return_inverse=True)
            self.context_of = inverse.reshape(-1)
        else:
            self.contexts = np.zeros((1, 0), dtype=np.int64)
            self.context_of = np.zeros(n_p, dtype=np.int64)
        if len(self.contexts) * n_c > 10 * max_combinations:
            raise CombinationLimit(
                f"block {block}: {len(self.contexts):,} parent contexts x {n_c:,} child combinations "
                f"exceed 10 x max_combinations"
            )
        self.Z = np.array([self.child_weights(u).sum() for u in range(len(self.contexts))])

    def child_weights(self, u) -> np.ndarray:
        """P(c | parent context u), zero for child combinations that break a rule."""
        row = self.contexts[u]
        active = row[: self.n_joint].astype(bool)
        fixed = {a: int(row[self.n_joint + i]) for i, a in enumerate(self.cross)}
        weights = self.network.probability(self.block.child_attrs, self.ccols, fixed)
        valid = self.valid_c
        if active.any():
            valid = valid & ~self.hc[:, active].any(axis=1)
        return weights * valid

    def parent_weights(self, children, conditioning) -> np.ndarray:
        """Unnormalised probability of each parent combination for a parent with ``children`` children."""
        base = self.prob_p * self.valid_p
        if children == 0:
            return base
        z = self.Z[self.context_of]
        if conditioning == "sequential":
            return base * (z > 0)
        with np.errstate(divide="ignore"):
            log_w = np.log(base) + children * np.log(z)
        if np.isneginf(log_w).all():
            return np.zeros_like(base)
        return np.exp(log_w - log_w.max())

    def sample_parents(self, counts, rng, conditioning) -> np.ndarray:
        out = np.empty(len(counts), dtype=np.int64)
        if conditioning == "sequential":
            labels = np.minimum(counts, 1)   # 0 children, or at least one
        else:
            labels = counts
        for k, rows in grouped(labels):
            weights = self.parent_weights(int(k), conditioning)
            total = weights.sum()
            if not total > 0:
                extra = "" if k == 0 else " and leaves a valid child combination"
                raise ValueError(f"block {self.block}: no parent combination passes the rules{extra}")
            out[rows] = rng.choice(len(self.P), size=len(rows), p=weights / total)
        return out

    def sample_children(self, parent_combos, rng) -> np.ndarray:
        out = np.empty(len(parent_combos), dtype=np.int64)
        for u, rows in grouped(self.context_of[parent_combos]):
            weights = self.child_weights(int(u))
            out[rows] = rng.choice(len(self.C), size=len(rows), p=weights / weights.sum())
        return out

    def target_marginals(self) -> dict[str, np.ndarray]:
        """Marginal of every attribute of the block under the network alone (no rules)."""
        out = {}
        for i, a in enumerate(self.block.parent_attrs):
            out[a] = np.bincount(self.P[:, i], weights=self.prob_p, minlength=len(self.schema.domains[a]))
        if not self.block.child_attrs:
            return out
        if self.cross:
            values, inverse = np.unique(np.stack([self.pcols[a] for a in self.cross], axis=1), axis=0, return_inverse=True)
            mass = np.bincount(inverse.reshape(-1), weights=self.prob_p, minlength=len(values))
        else:
            values, mass = np.zeros((1, 0), dtype=np.int64), np.array([self.prob_p.sum()])
        for j, a in enumerate(self.block.child_attrs):
            out[a] = np.zeros(len(self.schema.domains[a]))
        for x, m in zip(values, mass):
            weights = self.network.probability(self.block.child_attrs, self.ccols, dict(zip(self.cross, map(int, x))))
            for j, a in enumerate(self.block.child_attrs):
                out[a] += m * np.bincount(self.C[:, j], weights=weights, minlength=len(out[a]))
        return out


class Sampler:
    def __init__(self, schema, rules, network, max_combinations=1_000_000):
        self.schema = schema
        self.blocks = find_blocks(schema, rules, network)
        self.models = [BlockModel(schema, network, b, max_combinations) for b in self.blocks]

    def sample(self, counts, rng, conditioning="sequential") -> Coded:
        if conditioning not in CONDITIONING:
            raise ValueError(f"conditioning must be one of {CONDITIONING}")
        counts = np.asarray(counts, dtype=np.int64)
        fk = np.repeat(np.arange(len(counts)), counts)
        parents, children = {}, {}
        for model in self.models:
            combos = model.sample_parents(counts, rng, conditioning)
            for i, a in enumerate(model.block.parent_attrs):
                parents[a] = model.P[combos, i]
            if model.block.child_attrs:
                child_combos = model.sample_children(combos[fk], rng)
                for j, a in enumerate(model.block.child_attrs):
                    children[a] = model.C[child_combos, j]
        return Coded(
            self.schema,
            {a: parents[a] for a in self.schema.parent_attrs},
            {a: children[a] for a in self.schema.child_attrs},
            fk,
        )

    def target_marginals(self) -> dict[str, np.ndarray]:
        out = {}
        for model in self.models:
            out.update(model.target_marginals())
        return {a: out[a] for a in self.schema.attrs}
