#set document(title: "datagen walkthrough: one parent with three children")
#set page(paper: "a4", margin: (x: 2.3cm, y: 2.4cm), numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.")
#set table(stroke: none, inset: (x: 5pt, y: 3.5pt), align: left)
#show raw.where(block: true): block.with(fill: luma(246), inset: 8pt, radius: 3pt, width: 100%)
#show figure.where(kind: table): set figure.caption(position: top)

// A table with a rule under the header and at the bottom.
#let tab(columns, header, ..cells) = table(
  columns: columns,
  table.hline(),
  table.header(..header.map(h => strong(h))),
  table.hline(stroke: 0.5pt),
  ..cells,
  table.hline(),
)
#let no = text(fill: rgb("#b03030"), weight: "bold")[0]
#let note(body) = block(
  fill: luma(244), inset: 8pt, radius: 3pt, width: 100%, body,
)

#align(center)[
  #text(16pt, weight: "bold")[datagen walkthrough] \
  #v(2pt)
  #text(11pt)[One parent with three children, step by step]
]

#v(4pt)

This document runs one small example through the generator. Every number in it was printed by the real code (the script is at the end). The full explanation, with the proofs, is in `generator.typ`. This document only follows the example.

= The example

*Schema.* One parent table `Shop` and one child table `Order`. Every value is stored as a _code_: its index in the domain, in the order written here.

#tab(
  (auto, auto, auto),
  ([table], [attribute], [domain (code 0, 1, 2)]),
  [Shop], [`kind`], [food, tech],
  [Shop], [`size`], [S, M, L],
  [Shop], [`region`], [north, south],
  [Order], [`pay`], [cash, card],
  [Order], [`big`], [Yes, No],
  [Order], [`gift`], [Yes, No],
)

*Rules.* A rule $not(a in S and b in T)$ forbids a combination. It _fails_ on a row when every atom holds.

#tab(
  (auto, auto, auto, 1fr),
  ([rule], [forbids], [kind], [meaning]),
  [R1], [$"kind" = "tech" and "size" = "S"$], [parent], [small shops sell no tech],
  [R2], [$"pay" = "cash" and "big" = "Yes"$], [child], [big orders are not paid in cash],
  [R3], [$"kind" = "food" and "big" = "Yes"$], [parent–child], [food shops take no big orders],
  [R4], [$"size" = "L" and "big" = "No"$], [parent–child], [large shops take only big orders],
)

A parent–child rule is checked on every (parent, child) pair: under R3, a child of a food shop may not have big = Yes.

*Everything else* uses the defaults: all frequency tables uniform, no dependencies, `fanout=Fixed(3)` (every parent gets 3 children), `parents=1`, `seed=5`, `conditioning="sequential"`.

```python
spec = DatasetSpec(schema, rules, fanout=Fixed(3), parents=1, seed=5)
clean = generate(spec)
```

= The pipeline

`generate(spec)` (`generator/generate.py`) runs these steps:

+ `check_rules`: every rule names known attributes and values.
+ `Network(...)`: one probability table per attribute (@sec:network).
+ `Sampler(...)`: splits the attributes into blocks (@sec:blocks) and builds a `BlockModel` for each block. The model lists all combinations and computes their weights in advance (@sec:enum, @sec:contexts). *Nothing random happens up to here.*
+ `streams(seed)`: makes the random number generators (@sec:seeds).
+ `fanout.sample(...)`: how many children each parent gets (@sec:seeds).
+ `sampler.sample(counts, ...)`: draws the values, block by block (@sec:parent to @sec:other).
+ Assemble the tables and compute the statistics (@sec:result).

= The network: probabilities without rules <sec:network>

Without dependencies, each attribute has only its own frequency table. With uniform tables, `kind`, `region`, `pay`, `big` and `gift` get $[1/2, 1/2]$ and `size` gets $[1/3, 1/3, 1/3]$. The probability of a combination is the product (`Network.probability`), for example
$ P("kind" = "food", "size" = "M") = 1/2 dot 1/3 = 1/6. $
Because everything is uniform, all combinations of the same attributes are equally likely. This is the distribution the generator would use *if there were no rules*. The rest of the algorithm conditions it on the rules.

= Blocks: which attributes must be drawn together <sec:blocks>

`find_blocks` (`sampler.py`) links two attributes when they appear in the same rule (or are joined by a dependency; there are none here) and takes the connected groups with union–find:

#align(center, grid(
  columns: 2, column-gutter: 2.5cm, align: center + horizon,
  [
    R1: kind — size #h(1em) R3: kind — big \
    R2: pay — big #h(1.6em) R4: size — big
  ],
  [`{kind, size, big, pay}`  #h(1em) `{region}` #h(1em) `{gift}`],
))

#tab(
  (auto, auto, auto, auto),
  ([block], [parent attributes], [child attributes], [rules]),
  [A], [kind, size], [pay, big], [R1, R2, R3, R4],
  [B], [region], [—], [—],
  [C], [—], [gift], [—],
)

*Why this helps.* No rule and no dependency crosses between blocks, so the blocks are independent: the value of `region` has no effect on which `kind` or `gift` is allowed. Each block is drawn on its own, and the columns are put together at the end. Only block A is interesting. B and C are plain uniform draws.

= Block A: list every combination <sec:enum>

`BlockModel.__init__` lists every combination of the block's parent attributes ($cal(P)$) and of its child attributes ($cal(C)$), with the first attribute varying slowest (`cartesian`).

#grid(
  columns: (1fr, 1fr), column-gutter: 1em,
  figure(
    tab(
      (auto, auto, auto, auto, auto),
      ([$p$], [kind], [size], [$P(p)$], [$"valid"_p$]),
      [0], [food], [S], [1/6], [1],
      [1], [food], [M], [1/6], [1],
      [2], [food], [L], [1/6], [1],
      [3], [tech], [S], [1/6], [#no (R1)],
      [4], [tech], [M], [1/6], [1],
      [5], [tech], [L], [1/6], [1],
    ),
    caption: [Parent combinations ($2 times 3$)],
  ),
  figure(
    tab(
      (auto, auto, auto, auto, auto),
      ([$c$], [pay], [big], [$P(c)$], [$"valid"_c$]),
      [0], [cash], [Yes], [1/4], [#no (R2)],
      [1], [cash], [No], [1/4], [1],
      [2], [card], [Yes], [1/4], [1],
      [3], [card], [No], [1/4], [1],
    ),
    caption: [Child combinations ($2 times 2$)],
  ),
)

$"valid"_p (p) = 1$ when no _parent_ rule fails (only R1 is one), and $"valid"_c (c) = 1$ when no _child_ rule fails (only R2 is one).

*Parent–child rules are split in two.* R3 has a parent part ($"kind" = "food"$) and a child part ($"big" = "Yes"$), and it fails on a pair exactly when both parts hold. So for every parent–child rule $r$ the code computes one boolean per parent combination, $h^P_r (p)$ ("does the parent part hold?", `hp`), and one per child combination, $h^C_r (c)$ (`hc`):

#grid(
  columns: (1fr, 1fr), column-gutter: 1em,
  tab(
    (auto, auto, auto),
    ([parent], [$h^P_"R3"$: food], [$h^P_"R4"$: L]),
    [0 food, S], [1], [0],
    [1 food, M], [1], [0],
    [2 food, L], [1], [1],
    [3 tech, S], [0], [0],
    [4 tech, M], [0], [0],
    [5 tech, L], [0], [1],
  ),
  tab(
    (auto, auto, auto),
    ([child], [$h^C_"R3"$: Yes], [$h^C_"R4"$: No]),
    [0 cash, Yes], [1], [0],
    [1 cash, No], [0], [1],
    [2 card, Yes], [1], [0],
    [3 card, No], [0], [1],
  ),
)

A child $c$ *passes* with a parent $p$ when
$ "pass"(p, c) = "valid"_c (c) and not or.big_(r in {"R3", "R4"}) (h^P_r (p) and h^C_r (c)). $

= Block A: contexts, child weights and $Z$ <sec:contexts>

Which children pass depends on the parent only through its bits $h^P_r (p)$: which parent–child rules the parent "switches on". This bit pattern is the parent's *context*. Everything about the children is computed once per context instead of once per parent.

#tab(
  (auto, auto, auto, 1fr),
  ([context $u$], [bits (R3, R4)], [parents], [rules switched on]),
  [0], [(0, 0)], [tech S, tech M], [none],
  [1], [(0, 1)], [tech L], [R4: children must have big = Yes],
  [2], [(1, 0)], [food S, food M], [R3: children must have big = No],
  [3], [(1, 1)], [food L], [R3 and R4: big can be neither],
)

(With a dependency from a parent attribute to a child attribute, that parent attribute's value would be part of the context too, because it changes $P(c)$. That is not the case here.)

For each context, `child_weights(u)` gives every child combination the weight $w_u (c) = P(c) dot "pass"(p, c)$, i.e. $P(c)$, or 0 when the child does not pass. $Z_u = sum_c w_u (c)$ is the probability that a child drawn _without_ rules happens to pass with this parent.

#figure(
  tab(
    (auto, auto, auto, auto, auto, auto),
    ([$u$], [cash, Yes], [cash, No], [card, Yes], [card, No], [$Z_u$]),
    [0], [#no (R2)], [1/4], [1/4], [1/4], [3/4],
    [1], [#no (R2)], [#no (R4)], [1/4], [#no (R4)], [1/4],
    [2], [#no (R2)], [1/4], [#no (R3)], [1/4], [1/2],
    [3], [#no (R2)], [#no (R4)], [#no (R3)], [#no (R4)], [*0*],
  ),
  caption: [Child weights $w_u (c)$ per context],
) <tab:w>

Context 3 has $Z = 0$: a large food shop cannot have any child. That parent combination is _valid_ (no parent rule fails) but _infeasible_ (it cannot have children). This matters in @sec:parent.

All of this is computed once, before anything random happens.

= Seeds and the number of children <sec:seeds>

`streams(5)` splits the seed with `numpy.random.SeedSequence(5).spawn(3)` into three independent generators: one for the number of children, one for the clean values, and one for the errors (used later by the injector). Because they are separate, changing the error settings never changes the clean data.

`Fixed(3).sample(...)` returns `counts = [3]`, so parent 0 gets 3 children. (`Poisson` or `UniformRange` would draw random counts here, from the first generator.) `Sampler.sample` turns this into the foreign key of every child: `fk = np.repeat([0], [3]) = [0, 0, 0]`.

= Block A: draw the parent <sec:parent>

`sample_parents`: with sequential conditioning, parents are grouped into "0 children" and "at least 1 child". Our parent has 3, so it is in the second group, and `parent_weights` gives every parent combination the weight
$ "weight"(p) = P(p) dot "valid"_p (p) dot [Z(p) > 0]. $

#tab(
  (auto, auto, auto, auto, auto, auto),
  ([$p$], [$P(p)$], [$"valid"_p$], [$Z(p) > 0$], [weight], [probability]),
  [food, S], [1/6], [1], [1], [1/6], [*0.25*],
  [food, M], [1/6], [1], [1], [1/6], [*0.25*],
  [food, L], [1/6], [1], [#no], [0], [0],
  [tech, S], [1/6], [#no], [1], [0], [0],
  [tech, M], [1/6], [1], [1], [1/6], [*0.25*],
  [tech, L], [1/6], [1], [1], [1/6], [*0.25*],
)

tech S is excluded by R1. food L is excluded because it could not have its 3 children.

`rng.choice(6, p=[.25, .25, 0, 0, .25, .25])` returns *1*, so the parent is *(kind = food, size = M)*, with context 2.

= Block A: draw the children <sec:children>

`sample_children`: the parent's context is 2 (R3 switched on), so every child is drawn from row 2 of @tab:w, divided by $Z_2 = 1/2$:
$ Pr(c | p) = w_2(c) / Z_2 = [0, thick 0.5, thick 0, thick 0.5]. $

`rng.choice(4, size=3, p=[0, .5, 0, .5])` returns *[1, 3, 3]*: (cash, No), (card, No), (card, No).

#note[None of them can break a rule: (cash, Yes) has weight 0 because of R2, and every big = Yes has weight 0 because this is a food shop (R3). *The rules hold by construction. Nothing is generated and then checked or repaired.*]

= Blocks B and C <sec:other>

*Block B = {region}* has only a parent attribute and no rules. There is one context with $Z = 1$, so the weights are just $P(p)$: `rng.choice(2, p=[.5, .5])` returns *0*, so region = north. It has no child attributes, so there is no child step.

*Block C = {gift}* has only a child attribute. Its parent side has no attributes, so $cal(P)$ has exactly one (empty) combination, and the parent draw `rng.choice(1, p=[1])` always returns 0. It still runs and still uses the generator. Then every child is drawn: `rng.choice(2, size=3, p=[.5, .5])` returns *[0, 0, 0]*, so gift = Yes three times.

The clean-value generator was called in this order:

#tab(
  (auto, auto, auto, auto, auto),
  ([\#], [block], [what], [call], [result]),
  [1], [A], [parent], [`choice(6, p=[.25,.25,0,0,.25,.25])`], [1 = food, M],
  [2], [A], [3 children], [`choice(4, size=3, p=[0,.5,0,.5])`], [1, 3, 3],
  [3], [B], [parent], [`choice(2, p=[.5,.5])`], [0 = north],
  [4], [C], [parent (trivial)], [`choice(1, p=[1])`], [0],
  [5], [C], [3 children], [`choice(2, size=3, p=[.5,.5])`], [0, 0, 0],
)

= Putting it together <sec:result>

The codes of all blocks go into one `Coded` object (`data.py`). The keys are numbered 0, 1, 2, …, the codes are turned back into strings (`to_frames`), and the foreign key comes from `fk`:

#grid(
  columns: (auto, 1fr), column-gutter: 2em,
  [
    `Shop_clean.csv`
    #tab(
      (auto, auto, auto, auto),
      ([shop_id], [kind], [size], [region]),
      [0], [food], [M], [north],
    )
  ],
  [
    `Order_clean.csv`
    #tab(
      (auto, auto, auto, auto, auto),
      ([order_id], [shop_id], [pay], [big], [gift]),
      [0], [0], [cash], [No], [Yes],
      [1], [0], [card], [No], [Yes],
      [2], [0], [card], [No], [Yes],
    )
  ],
)

Check: R1 (not tech) ✓, R2 (no cash with Yes) ✓, R3 (food shop, no big = Yes) ✓, R4 (not L) ✓. `generate` then computes the statistics (0 rule violations, and the frequencies compared with the targets), and `write` saves the CSVs and `dataset.json`.

= What changes in other situations

*Many parents.* The code does not loop over parents. In @sec:parent, _all_ parents with at least one child are drawn in one `rng.choice(6, size=n, ...)` call, and all parents with 0 children in another. In @sec:children, all children whose parent has context 2 are drawn in one call, then all children of context 0, and so on (`grouped`). That is why it is fast: the work grows with the number of combinations and contexts, not with the number of parents. The result has the same distribution as drawing every family on its own. Only the order of the random numbers differs.

*A parent with 0 children* is drawn with weight $P(p) dot "valid"_p (p)$, _without_ the feasibility factor, because it needs no valid child. So it can be food L: each of the five valid combinations has probability 1/5.

*Joint conditioning* (`conditioning="joint"`) uses the weight $P(p) dot "valid"_p (p) dot Z(p)^n$, with $n$ the number of children. This is what you would get by drawing whole families without rules and throwing away every family that breaks one (rejection sampling). Parents that allow many children become more likely as $n$ grows:

#tab(
  (auto, auto, auto, auto, auto),
  ([$p$], [$Z$], [sequential ($n >= 1$)], [joint, $n = 1$], [joint, $n = 3$]),
  [food, S], [1/2], [0.25], [0.25], [0.18],
  [food, M], [1/2], [0.25], [0.25], [0.18],
  [food, L], [0], [0], [0], [0],
  [tech, S], [(R1)], [0], [0], [0],
  [tech, M], [3/4], [0.25], [0.375], [0.61],
  [tech, L], [1/4], [0.25], [0.125], [0.02],
)

Sequential is the default because there a parent's distribution does not depend on how many children it gets. The children are drawn the same way under both.

*Non-uniform frequency tables* only change the numbers $P(p)$ and $P(c)$: the 1/6 and 1/4 above become products of the attributes' own frequencies. Every 0 stays a 0, so the rules still hold.

*A dependency* such as `Depends("big", "size", 0.6)` links `size` and `big` into the same block (they already were here). $P(c)$ now depends on the parent's `size`, so `size` becomes part of the context, and the context table gets more rows. Nothing else changes.

= Reproduce it

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

# The internals of the blocks, contexts and Z:
from datagen.generator.generate import network_and_sampler
network, sampler = network_and_sampler(spec)
for model in sampler.models:
    print(model.block, [r.name for r in model.block.rules])
    print("  P:", model.P.tolist(), "valid_p:", model.valid_p.astype(int).tolist())
    print("  C:", model.C.tolist(), "valid_c:", model.valid_c.astype(int).tolist())
    print("  contexts:", model.contexts.tolist(), "context_of:", model.context_of.tolist())
    print("  Z:", model.Z.tolist())
```

Try `seed=6` (a tech M shop: no parent–child rule switched on, so the children use three combinations), `seed=2` (a tech L shop: R4 leaves only (card, Yes)), or `fanout=UniformRange(0, 4)` with more parents.
