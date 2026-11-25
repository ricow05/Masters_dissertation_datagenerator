# Walkthrough: one parent with three children

This walks one small example through the generator, step by step. Every number below was
printed by the real code (see [Reproduce it](#reproduce-it) at the end). The longer
explanation, with the proofs, is in `generator.typ`. This document only follows the
example.

## 1. The example

**Schema.** One parent table `Shop` and one child table `Order`. Every attribute is stored
as a *code*: its index in the domain. The domain order is the order written here.

| table | attribute | domain (code 0, 1, 2) |
|---|---|---|
| Shop | `kind` | food, tech |
| Shop | `size` | S, M, L |
| Shop | `region` | north, south |
| Order | `pay` | cash, card |
| Order | `big` | Yes, No |
| Order | `gift` | Yes, No |

**Rules.** A rule `NOT(a ∈ S ∧ b ∈ T)` forbids that combination. It *fails* on a row when
every atom holds.

| rule | forbids | kind | meaning |
|---|---|---|---|
| R1 | `kind=tech ∧ size=S` | parent | small shops sell no tech |
| R2 | `pay=cash ∧ big=Yes` | child | big orders are not paid in cash |
| R3 | `kind=food ∧ big=Yes` | parent-child | food shops take no big orders |
| R4 | `size=L ∧ big=No` | parent-child | large shops take only big orders |

A parent-child rule is checked on every (parent, child) pair: for R3, a child of a food
shop may not have `big=Yes`.

**Everything else** uses the defaults: all frequency tables uniform, no dependencies,
`fanout=Fixed(3)` (every parent gets 3 children), `parents=1`, `seed=5`,
`conditioning="sequential"`.

```python
spec = DatasetSpec(schema, rules, fanout=Fixed(3), parents=1, seed=5)
clean = generate(spec)
```

## 2. The pipeline

`generate(spec)` ([generate.py](../datagen/generator/generate.py)) runs these steps:

1. `check_rules`: every rule names known attributes and values.
2. `Network(...)`: one probability table per attribute (section 3).
3. `Sampler(...)`: splits the attributes into blocks (section 4) and builds a `BlockModel`
   per block, which enumerates all combinations and precomputes their weights (sections 5
   and 6). **Nothing random happens up to here.**
4. `streams(seed)`: makes the random number generators (section 7).
5. `fanout.sample(...)`: how many children each parent gets (section 7).
6. `sampler.sample(counts, ...)`: draws the values, block by block (sections 8 to 10).
7. Assemble the tables and compute the statistics (section 11).

## 3. The network: probabilities without rules

Without dependencies, each attribute has only its own frequency table. With uniform tables:

- `kind`, `region`, `pay`, `big`, `gift`: [1/2, 1/2]
- `size`: [1/3, 1/3, 1/3]

The probability of a combination is the product of these numbers
(`Network.probability`). For example, P(kind=food, size=M) = 1/2 · 1/3 = 1/6. Because
everything is uniform, every combination of the same attributes has the same probability.
This is the distribution the generator would use **if there were no rules**. The rest of
the algorithm is about conditioning it on the rules.

## 4. Blocks: which attributes must be drawn together

`find_blocks` ([sampler.py:56](../datagen/generator/sampler.py#L56)) links two attributes
when they appear in the same rule (or are joined by a dependency; there are none here) and
takes the connected groups (union-find):

```
R1: kind — size        R3: kind — big
R2: pay  — big         R4: size — big

  kind ─── size          region      gift
    \      /
     big ─ pay
```

This gives three blocks:

| block | parent attributes | child attributes | rules |
|---|---|---|---|
| A | kind, size | pay, big | R1, R2, R3, R4 |
| B | region | (none) | (none) |
| C | (none) | gift | (none) |

**Why this helps.** No rule and no dependency crosses between blocks, so the blocks are
independent of each other: what `region` is has no effect on which `kind` or `gift` is
allowed. The generator draws each block separately and puts the columns together at the
end. Only block A is interesting. B and C are plain uniform draws.

## 5. Block A: list every combination

`BlockModel.__init__` ([sampler.py:109](../datagen/generator/sampler.py#L109)) enumerates
all combinations of the block's parent attributes (`P`) and of its child attributes (`C`),
with the first attribute varying slowest (`cartesian`).

**Parent combinations** (6 = 2 × 3):

| index | kind | size | P(p) | `valid_p` | why |
|---|---|---|---|---|---|
| 0 | food | S | 1/6 | 1 | |
| 1 | food | M | 1/6 | 1 | |
| 2 | food | L | 1/6 | 1 | |
| 3 | tech | S | 1/6 | **0** | R1 fails |
| 4 | tech | M | 1/6 | 1 | |
| 5 | tech | L | 1/6 | 1 | |

`valid_p` is 1 when no *parent* rule fails. R1 is the only parent rule.

**Child combinations** (4 = 2 × 2):

| index | pay | big | P(c) | `valid_c` | why |
|---|---|---|---|---|---|
| 0 | cash | Yes | 1/4 | **0** | R2 fails |
| 1 | cash | No | 1/4 | 1 | |
| 2 | card | Yes | 1/4 | 1 | |
| 3 | card | No | 1/4 | 1 | |

`valid_c` is 1 when no *child* rule fails. R2 is the only child rule.

**Parent-child rules are split in two.** R3 = `kind=food ∧ big=Yes` has a parent part
(`kind=food`) and a child part (`big=Yes`). It fails on a pair exactly when both parts hold.
So the code computes, for each rule, one boolean per parent combination (`hp`: does the
parent part hold?) and one per child combination (`hc`: does the child part hold?):

| parent | `hp` R3 (kind=food) | `hp` R4 (size=L) |
|---|---|---|
| 0 food,S | 1 | 0 |
| 1 food,M | 1 | 0 |
| 2 food,L | 1 | 1 |
| 3 tech,S | 0 | 0 |
| 4 tech,M | 0 | 0 |
| 5 tech,L | 0 | 1 |

| child | `hc` R3 (big=Yes) | `hc` R4 (big=No) |
|---|---|---|
| 0 cash,Yes | 1 | 0 |
| 1 cash,No | 0 | 1 |
| 2 card,Yes | 1 | 0 |
| 3 card,No | 0 | 1 |

A child c **passes** with a parent p when `valid_c(c) = 1` and there is no parent-child rule
with both `hp(p) = 1` and `hc(c) = 1`.

## 6. Block A: contexts, child weights and Z

Which children pass depends on the parent only through its `hp` bits: which parent-child
rules the parent "switches on". The code calls this bit pattern the parent's **context**
and computes everything about children once per context, not once per parent.

| context u | bits (R3, R4) | parents with it | rules switched on |
|---|---|---|---|
| 0 | (0, 0) | tech,S; tech,M | none |
| 1 | (0, 1) | tech,L | R4: children must have big=Yes |
| 2 | (1, 0) | food,S; food,M | R3: children must have big=No |
| 3 | (1, 1) | food,L | R3 and R4: big can be neither |

(With a dependency from a parent attribute to a child attribute, that parent attribute's
value would also be part of the context, because it changes P(c). Not the case here.)

For each context, `child_weights(u)` ([sampler.py:161](../datagen/generator/sampler.py#L161))
gives the weight of every child combination: P(c), set to 0 when the child does not pass.
**Z** is the sum of those weights: the probability that a child drawn without rules happens
to pass with this parent.

| u | cash,Yes | cash,No | card,Yes | card,No | Z |
|---|---|---|---|---|---|
| 0 | 0 (R2) | 1/4 | 1/4 | 1/4 | 3/4 |
| 1 | 0 (R2) | 0 (R4) | 1/4 | 0 (R4) | 1/4 |
| 2 | 0 (R2) | 1/4 | 0 (R3) | 1/4 | 1/2 |
| 3 | 0 (R2) | 0 (R4) | 0 (R3) | 0 (R4) | **0** |

Context 3 has **Z = 0**: a large food shop cannot have any child. Such a parent combination
is *valid* (no parent rule fails) but *infeasible* (it cannot have children). That matters
in section 8.

All of this is computed once, before anything random happens.

## 7. Seeds and the number of children

`streams(5)` ([generate.py:32](../datagen/generator/generate.py#L32)) splits the seed with
`numpy.random.SeedSequence(5).spawn(3)` into three independent generators: one for the
number of children, one for the clean values, one for the errors (used later by the
injector). Because they are separate, changing the error settings never changes the clean
data.

`Fixed(3).sample(...)` returns `counts = [3]`: parent 0 gets 3 children. (`Poisson` or
`UniformRange` would draw random counts here, from the first generator.)

`Sampler.sample` ([sampler.py:235](../datagen/generator/sampler.py#L235)) turns that into
the foreign key of every child: `fk = np.repeat([0], [3]) = [0, 0, 0]`.

## 8. Block A: draw the parent

`sample_parents` ([sampler.py:186](../datagen/generator/sampler.py#L186)). With sequential
conditioning, parents are grouped by "0 children" or "at least 1 child". Our parent has
3, so it is in the second group. The weight of each parent combination is
(`parent_weights`, [sampler.py:172](../datagen/generator/sampler.py#L172)):

```
weight(p) = P(p) · valid_p(p) · [Z(p) > 0]
```

| p | P(p) | valid_p | Z > 0 | weight | probability |
|---|---|---|---|---|---|
| food,S | 1/6 | 1 | 1 | 1/6 | **0.25** |
| food,M | 1/6 | 1 | 1 | 1/6 | **0.25** |
| food,L | 1/6 | 1 | **0** | 0 | 0 |
| tech,S | 1/6 | **0** | 1 | 0 | 0 |
| tech,M | 1/6 | 1 | 1 | 1/6 | **0.25** |
| tech,L | 1/6 | 1 | 1 | 1/6 | **0.25** |

- tech,S is excluded by R1.
- food,L is excluded because it could not have its 3 children.

`rng.choice(6, p=[.25, .25, 0, 0, .25, .25])` returns **1**: the parent is
**(kind=food, size=M)**, context 2.

## 9. Block A: draw the children

`sample_children` ([sampler.py:201](../datagen/generator/sampler.py#L201)). The parent's
context is 2 (R3 switched on), so every child is drawn from row 2 of the table in section 6,
divided by Z = 1/2:

| child | cash,Yes | cash,No | card,Yes | card,No |
|---|---|---|---|---|
| probability | 0 | 0.5 | 0 | 0.5 |

`rng.choice(4, size=3, p=[0, .5, 0, .5])` returns **[1, 3, 3]**:

| order | pay | big |
|---|---|---|
| 0 | cash | No |
| 1 | card | No |
| 2 | card | No |

None of them can break a rule: (cash, Yes) has weight 0 because of R2, and every `big=Yes`
has weight 0 because this is a food shop (R3). **The rules hold by construction. Nothing is
generated and then checked or repaired.**

## 10. Blocks B and C

**Block B = {region}** has only a parent attribute and no rules. There is one context with
Z = 1, so the weights are just P(p):
`rng.choice(2, p=[.5, .5])` returns **0** → `region=north`. It has no child attributes, so
there is no child step.

**Block C = {gift}** has only a child attribute. Its parent side has no attributes, so `P`
has exactly one (empty) combination and the parent draw `rng.choice(1, p=[1])` always returns
0. It still runs, and it still uses the generator. Then every child is drawn:
`rng.choice(2, size=3, p=[.5, .5])` returns **[0, 0, 0]** → `gift=Yes` three times.

The clean-value generator was called in this order:

| # | block | what | call | result |
|---|---|---|---|---|
| 1 | A | parent | `choice(6, p=[.25,.25,0,0,.25,.25])` | 1 = food,M |
| 2 | A | 3 children | `choice(4, size=3, p=[0,.5,0,.5])` | 1,3,3 |
| 3 | B | parent | `choice(2, p=[.5,.5])` | 0 = north |
| 4 | C | parent (trivial) | `choice(1, p=[1])` | 0 |
| 5 | C | 3 children | `choice(2, size=3, p=[.5,.5])` | 0,0,0 |

## 11. Putting it together

The codes of all blocks are put into one `Coded` object (`data.py`). The keys are numbered
0, 1, 2, …, the codes become strings again (`to_frames`), and the foreign key comes from
`fk`:

`Shop_clean.csv`

| shop_id | kind | size | region |
|---|---|---|---|
| 0 | food | M | north |

`Order_clean.csv`

| order_id | shop_id | pay | big | gift |
|---|---|---|---|---|
| 0 | 0 | cash | No | Yes |
| 1 | 0 | card | No | Yes |
| 2 | 0 | card | No | Yes |

Check: R1 (not tech) ✓, R2 (no cash with Yes) ✓, R3 (food shop, no big=Yes) ✓, R4 (not
L) ✓. `generate` then computes the statistics (rule violations: 0, frequencies compared with
the targets), and `write` saves the CSVs and `dataset.json`.

## 12. What changes in other situations

**Many parents.** The code does not loop over parents. In step 8, *all* parents with at
least one child are drawn in one `rng.choice(6, size=n, ...)` call, and all parents with 0
children in another. In step 9, all children whose parent has context 2 are drawn in one
call, then all children of context 0, and so on (`grouped`). That is why it is fast: the work
grows with the number of combinations and contexts, not with the number of parents. The
result is the same as drawing every family on its own; only the order of the random numbers
differs.

**A parent with 0 children.** It is drawn with weight P(p) · valid_p(p), *without* the
feasibility factor, because it needs no valid child. So it can be food,L, with probability
1/5 for each of the five valid combinations.

**Joint conditioning** (`conditioning="joint"`). The weight becomes P(p) · valid_p(p) · Z(p)ⁿ,
with n the number of children. This is what you would get by drawing whole families without
rules and throwing away every family that breaks one (rejection sampling). Parents that allow
many children become more likely as n grows:

| p | Z | sequential (n ≥ 1) | joint, n = 1 | joint, n = 3 |
|---|---|---|---|---|
| food,S | 1/2 | 0.25 | 0.25 | 0.18 |
| food,M | 1/2 | 0.25 | 0.25 | 0.18 |
| food,L | 0 | 0 | 0 | 0 |
| tech,S | (R1) | 0 | 0 | 0 |
| tech,M | 3/4 | 0.25 | 0.375 | 0.61 |
| tech,L | 1/4 | 0.25 | 0.125 | 0.02 |

Sequential is the default because there the parent's distribution does not depend on how
many children it gets. Under either conditioning, the children are drawn the same way.

**Non-uniform frequency tables.** Only the numbers P(p) and P(c) change: the 1/6 and 1/4
above become the products of the attributes' own frequencies. Every 0 stays a 0, so the
rules still hold.

**A dependency** such as `Depends("big", "size", 0.6)`. The edge links `size` and `big` into
the same block (they already were here). P(c) now depends on the parent's `size`, so `size`
becomes part of the context, and the context table gets more rows. Otherwise nothing changes.

## Reproduce it

```python
from datagen import DatasetSpec, Rule, Fixed, two_tables, generate

schema = two_tables(
    "Shop", "shop_id",
    {"kind": ("food", "tech"), "size": ("S", "M", "L"), "region": ("north", "south")},
    "Order", "order_id",
    {"pay": ("cash", "card"), "big": ("Yes", "No"), "gift": ("Yes", "No")},
)
rules = [
    Rule.of({"kind": "tech", "size": "S"}, "R1"),
    Rule.of({"pay": "cash", "big": "Yes"}, "R2"),
    Rule.of({"kind": "food", "big": "Yes"}, "R3"),
    Rule.of({"size": "L", "big": "No"}, "R4"),
]
spec = DatasetSpec(schema, rules, fanout=Fixed(3), parents=1, seed=5)
clean = generate(spec)
for name, frame in clean.frames().items():
    print(name); print(frame)

# The internals from sections 4 to 6:
from datagen.generator.generate import network_and_sampler
network, sampler = network_and_sampler(spec)
for model in sampler.models:
    print(model.block, [r.name for r in model.block.rules])
    print("  P:", model.P.tolist(), "valid_p:", model.valid_p.astype(int).tolist())
    print("  C:", model.C.tolist(), "valid_c:", model.valid_c.astype(int).tolist())
    print("  contexts:", model.contexts.tolist(), "context_of:", model.context_of.tolist())
    print("  Z:", model.Z.tolist())
```

Try `seed=6` (a tech,M shop: no parent-child rule switched on, so the children use three
combinations), `seed=2` (a tech,L shop: R4 leaves only card,Yes), or
`fanout=UniformRange(0, 4)` with more parents.
