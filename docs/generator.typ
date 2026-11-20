#set document(title: "datagen: a rule-aware generator for parent-child data with controlled errors")
#set page(paper: "a4", margin: (x: 2.3cm, y: 2.4cm), numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#set table(stroke: none, inset: (x: 5pt, y: 3.5pt), align: left)
#show raw.where(block: true): block.with(fill: luma(246), inset: 8pt, radius: 3pt, width: 100%)
#show figure.where(kind: table): set figure.caption(position: top)

#let stage(title, body) = box(
  width: 100%, height: 7.2em, stroke: 0.6pt + luma(140), inset: 6pt, radius: 3pt,
  {
    set par(justify: false)
    set align(left)
    set text(hyphenate: false)
    text(weight: "bold", size: 9pt, title)
    linebreak()
    text(size: 8pt, body)
  },
)
#let flow = align(center + horizon, text(14pt, $arrow.r$))

#align(center)[
  #text(17pt, weight: "bold")[datagen] \
  #v(2pt)
  #text(12pt)[A rule-aware generator for parent–child data with controlled errors] \
  #v(4pt)
  #text(9.5pt, fill: luma(90))[How the datagen generator works, and why it is built this way]
]

#v(6pt)

#outline(depth: 2, indent: auto)

#pagebreak()

= Why a new generator

The benchmark of Algorithm 1 (`first_algo.typ`) needs two-table databases with four properties:

+ *Valid ground truth.* The clean data satisfy every given constant selection rule and parent–child rule. A repair can only be judged against a truth that is itself consistent.
+ *Known errors.* The location and the clean value of every error are recorded, and the number of errors is controlled.
+ *Reproducibility.* The same configuration gives the same files.
+ *Variation.* Frequency tables, correlation between attributes, the number of children per parent, and the location of errors can all be varied.

The benchmark first used GouDa. GouDa does not meet these properties, as @tab:gouda shows. The numbers were measured on the trials catalogue (8 parent and 8 child attributes, 30 rules).

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    table.hline(),
    table.header([*Property*], [*GouDa (+ `make_consistent`)*], [*datagen*]),
    table.hline(stroke: 0.5pt),
    [Clean data satisfy the rules], [No. A repair step redraws 61% of the parents and 86% of the children afterwards.], [Yes, by construction. 0 violations in every test.],
    [Realised error rate at nominal 5%], [2.8%: a "wrong" value is drawn from the whole domain, so it can be the correct one. It also varies per attribute and per rule set (`placebo`: 2.3% with 6 rules, 3.6% with 18).], [Exactly 5.00% per attribute, because a dirty value always differs from the clean one.],
    [Seed], [None. A dataset cannot be regenerated.], [One seed, with separate random streams for the fan-out, the clean data and the errors.],
    [Children per parent], [Exactly $k$.], [Fixed, uniform range, Poisson, geometric or empirical. Parents with 0 or 1 children occur.],
    [Correlation], [None between attributes, and none from parent to child.], [A Bayesian network, including parent→child edges.],
    [Errors aimed at the rules], [No.], [A chosen share of detectable and silent errors, per attribute.],
    [Generation time], [17–33 minutes for 20,000 parents (grows roughly quadratically).], [0.13 s for 10,000 parents and 1.2 s for 100,000 (30 rules, grows linearly).],
    table.hline(),
  ),
  caption: [GouDa compared with datagen on the benchmark setting.],
) <tab:gouda>

datagen is a Python package in its own repository that depends only on numpy and pandas. It shares no code with the benchmark. Its output is a folder of CSV files and one `dataset.json` that records the datagen version and git commit and every parameter of the data; `FORMAT.md` is the contract for programs that read such folders. The trials catalogue in `datagen/examples/trials.py` is one example schema; the benchmark writes its own catalogue into the specs it asks for.

== Overview

datagen has two parts that never import each other. The *generator* (`datagen.generator`) turns a specification (@sec:spec) into clean data: `generate(spec)` returns a `CleanData` with the coded tables and their statistics. The *injector* (`datagen.injector`) adds errors to clean data according to an `ErrorSpec` (@sec:errors). It needs only the schema, the rules and the clean tables, so it also works on clean data that datagen did not generate. `inject(clean, error_spec)` connects the two and returns a `DirtyData` with the dirty tables, the error log and the combined statistics. @fig:pipeline shows the stages. The same clean data can be given any number of error specs.

#figure(
  grid(
    columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
    column-gutter: 3pt,
    stage[Spec][schema, rules, frequency tables, dependencies, fan-out, seed],
    flow,
    stage[Model][Bayesian network; blocks of attributes linked by rules or edges],
    flow,
    stage[Clean data][children per parent; per block an exact draw conditioned on the rules],
    flow,
    stage[Dirty data (injector)][errors per (table, attribute): rate, rows, mechanism, detectability],
    flow,
    stage[Output][statistics; CSV files, error log, `dataset.json`],
  ),
  kind: image,
  caption: [The stages of `generate` (first three) and `inject` (fourth). The children per parent, the clean values and the errors each use their own random stream.],
) <fig:pipeline>

= Data model <sec:model>

== Schema

A schema has a *parent table* $T_p$ and a *child table* $T_c$. Each has a primary key, and $T_c$ has a foreign key to $T_p$. The attributes are $A_p$ (parent) and $A_c$ (child). Their names are unique across the two tables, as in `algo1/model.py`, so a rule can name an attribute without naming its table.

Every attribute $a$ has a finite, *ordered* domain $D(a) = (v_0, dots, v_(d_a - 1))$. The order matters for two things only: the shape of a frequency table (a bell shape over `phase` I–IV, a skew towards one end) and the `neighbour` error mechanism. Keys are not attributes. They are generated (`"0"`, `"1"`, …) and never get errors.

The schema is stored as a tree of tables: every table names its parent table. Currently the generator handles exactly one parent and one child, the setting of Algorithm 1. A third table would need a deeper tree and an extension of the sampler (@sec:limits).

== Coded representation

Internally every value is an integer *code*, its index in the domain. A dataset is a set of code arrays, one per attribute. Each child also has the row index of its parent, `fk`. For a child $s$ with parent $t$, the *joined row* $t join s$ holds the parent's codes and the child's codes. Rules involving both tables are evaluated on joined rows. The conversion to and from strings only happens when files are written or read.

== Rules

A constant selection rule is a negated conjunction of atoms
$ r = not (a_1 in S_1 and dots and a_m in S_m), quad S_j subset.eq D(a_j). $
A row *fails* $r$ when every atom holds. $"attrs"(r) = {a_1, dots, a_m}$. Each rule has one of three kinds, as in `first_algo.typ`:

- *parent* ($R_p$): $"attrs"(r) subset.eq A_p$, evaluated on every parent row;
- *child* ($R_c$): $"attrs"(r) subset.eq A_c$, evaluated on every child row;
- *parent–child* ($R_"pc"$): attributes from both tables, evaluated on every joined row $t join s$.

For a parent–child rule, the *parent part* is the conjunction of its atoms on $A_p$, and the *child part* the conjunction of its atoms on $A_c$. The rule fails on $t join s$ exactly when the parent part holds on $t$ and the child part holds on $s$.

Violations are counted as in `algo1_bench/gendb/validate.violations`: one per (row, rule), on parent rows for parent rules and on joined rows for the others.

#figure(
  table(
    columns: (auto, auto, 1fr),
    table.hline(),
    table.header([*Rule*], [*Kind*], [*Meaning*]),
    table.hline(stroke: 0.5pt),
    [$not ("design" in {"single-arm"} and "control" in {"placebo", "active"})$], [parent], [A single-arm trial has no comparator arm.],
    [$not ("age_group" in {"paediatric"} and "consent" in {"self"})$], [child], [Paediatric participants cannot consent themselves.],
    [$not ("control" in {"placebo"} and "placebo" in {"No"})$], [parent–child], [A placebo-controlled trial gives placebos.],
    table.hline(),
  ),
  caption: [Three of the 30 rules of the trials catalogue (P1, C1 and PC1).],
)

== The specification <sec:spec>

`DatasetSpec` holds everything that defines a dataset (@tab:spec). It can be saved as `spec.json`, and it is stored in full in the `dataset.json` of every folder, so `python -m datagen regenerate DIR` reproduces a dataset from its folder alone.

#figure(
  table(
    columns: (auto, 1fr, auto),
    table.hline(),
    table.header([*Field*], [*Meaning*], [*Default*]),
    table.hline(stroke: 0.5pt),
    [`schema`], [The two tables, keys and ordered domains], [—],
    [`rules`], [Constant selection rules (name, atoms, meaning)], [none],
    [`marginals`], [Frequency table per attribute (@sec:marginals)], [uniform],
    [`dependencies`], [`Depends` or `CPT` per attribute (@sec:network)], [none],
    [`fanout`], [Children per parent (@sec:fanout)], [`Fixed(5)`],
    [`parents`], [Number of parent rows], [1000],
    [`seed`], [Seed of all random streams (@sec:seeds)], [0],
    [`conditioning`], [`"sequential"` or `"joint"` (@sec:targets)], [sequential],
    [`max_combinations`], [Size limit for the enumeration (@sec:cost)], [$10^6$],
    table.hline(),
  ),
  caption: [The fields of `DatasetSpec`.],
) <tab:spec>

= Frequency tables and correlation

The generator first defines a probability distribution over complete families (a parent row with its children) *without* the rules: a Bayesian network. @sec:sampler then conditions it on the rules.

== Frequency tables <sec:marginals>

A frequency table $q_a$ gives the probability of each value of $D(a)$. There are four kinds. All are defined over the value indices $i = 0, dots, d_a - 1$:

- `Uniform()`: $q_a (v_i) = 1 \/ d_a$.
- `Normal(mean, sd)`: a normal density evaluated at the indices,
  $ q_a (v_i) prop exp(-1/2 ((i - mu) / sigma)^2), $
  where $mu$ is the middle index $(d_a - 1)\/2$ by default, a (fractional) index, or the index of a named value.
- `Skewed(s, towards)`: Zipf weights
  $ q_a (v_i) prop "rank"(i)^(-s), quad "rank"(i) = |i - i^*| + 1, $
  where $i^*$ is the first index (`towards="first"`), the last one (`"last"`) or the index of a named value. Larger $s$ means a stronger skew; $s = 0$ is uniform.
- `Frequencies({value: weight})`: explicit weights, normalised. Values that are left out get probability 0.

== Dependencies <sec:network>

An attribute can depend on other attributes. Its frequency table is then replaced by a conditional table $theta_a (x | x_(pi(a)))$, where $pi(a)$ are its *network parents*. There are two ways to declare one.

*`Depends(attr, on, strength, mapping)`* makes attribute $b$ follow attribute $a$ with strength $rho in [0, 1]$:
$ theta_b (y | a = x) = (1 - rho) thin q_b (y) + rho thin [y = m(x)]. $ <eq:depends>
With probability $rho$ the value of $b$ is copied (through the mapping $m$) from $a$, otherwise it is drawn from its own frequency table. $rho = 0$ makes the two independent and $rho = 1$ makes $b$ a function of $a$. In between, the association grows with $rho$, which the notebook shows with Cramér's V. Without an explicit mapping, the two domains are aligned by rank, first value to first value and last to last:
$ m(v_i) = w_(floor(i (d_b - 1)\/(d_a - 1) + 1\/2)), $
where $(w_j)$ is the domain of $b$. An explicit mapping must give a value for every $x in D(a)$.

*`CPT(attr, on, rows)`* gives the conditional table explicitly, possibly on several attributes. Rows that are left out fall back to the frequency table of the attribute. This is also what fitting produces (@sec:fit).

== The network

The distribution of one family without rules is
$ P(t, s_1, dots, s_n) = product_(a in A_p) theta_a (t_a | t_(pi(a))) dot product_(i = 1)^n product_(b in A_c) theta_b (s_(i,b) | (t join s_i)_(pi(b))). $ <eq:network>
An attribute without a dependency has $pi(a) = emptyset$ and $theta_a = q_a$. The network is checked when the spec is built:

- every attribute has at most one dependency (a `CPT` may still have several parents);
- a parent attribute never depends on a child attribute. A child row is drawn given its parent row, never the reverse, so child→parent edges are refused. Parent→child edges are allowed, and they carry correlation from a trial to its registrations;
- the dependencies are acyclic, checked by computing a topological order;
- every conditional table sums to one.

By @eq:network the children of one parent are independent given the parent. Siblings are correlated only through their common parent.

= Children per parent <sec:fanout>

The number of children $n(t)$ of each parent is drawn first, from its own random stream, before any value. The children of parent $t$ get foreign key $t$.

#figure(
  table(
    columns: (auto, 1fr, auto),
    table.hline(),
    table.header([*Fan-out*], [*$P(n = k)$*], [*Mean*]),
    table.hline(stroke: 0.5pt),
    [`Fixed(k)`], [$1$ for $k$, else 0 (the GouDa setting)], [$k$],
    [`UniformRange(low, high)`], [$1 \/ ("high" - "low" + 1)$ on low..high, both included], [$("low" + "high") \/ 2$],
    [`Poisson(lam)`], [$e^(-lambda) lambda^k \/ k!$], [$lambda$],
    [`Geometric(p)`], [$(1 - p)^k p$, $k = 0, 1, dots$], [$(1 - p) \/ p$],
    [`Empirical(weights)`], [given weights, e.g. learned from data], [—],
    table.hline(),
  ),
  caption: [Fan-out distributions.],
)

Algorithm 1 assumes at least two children per parent. `Fixed(k)` with $k >= 2$ keeps that assumption. The other distributions produce the zero-child and one-child cases that a later version of the algorithm must handle.

= Clean data that satisfy the rules <sec:sampler>

== The problem

The clean data must be drawn from the network of @eq:network *conditioned on the event that no rule fails*. There are three ways to get there:

+ *Rejection sampling.* Draw whole families and keep those without a violation. This is exact, but the acceptance probability of a family falls geometrically with its number of children. With 30 rules almost every family is rejected.
+ *Repair afterwards.* Draw freely, then redraw the rows that violate a rule. This is what `gendb/consistent.py` does for GouDa. It redraws most rows, and its distribution is only correct if the redraws are made in exactly the right way.
+ *Exact conditioning by enumeration.* Compute the conditional distribution directly and draw from it. This is what datagen does. It is possible because the domains are finite and small, and because the rules split the attributes into independent blocks.

== Blocks <sec:blocks>

Let $G$ be the graph over all attributes with an edge between $a$ and $b$ when they appear in the same rule or are joined by a network edge. A *block* $B$ is a connected component of $G$, with parent attributes $B_p = B inter A_p$ and child attributes $B_c = B inter A_c$. Blocks are computed with union–find. Without network edges they are exactly the attribute classes of Algorithm 1. An attribute that is in no rule and has no edge is a block of its own, and is drawn from its frequency table.

*Blocks are independent.* Every conditional table of @eq:network has its attribute and its network parents in one block, so the network factorises into one factor per block. Every rule has all its attributes in one block, so the event "no rule fails" is an intersection of one event per block. A product distribution conditioned on a product event is again a product. The blocks can therefore be drawn one after the other, each from its own conditional distribution, and the result is exactly the conditional distribution of the whole.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    table.hline(),
    table.header([*Rules*], [*Blocks*], [*Largest block*], [*$|cal(P)| times |cal(C)|$*], [*Contexts*]),
    table.hline(stroke: 0.5pt),
    [6], [10], [7 attributes (4 parent, 3 child)], [$108 times 8$], [4],
    [12], [6], [8 attributes (5 + 3)], [$216 times 8$], [9],
    [18], [3], [12 attributes (6 + 6)], [$648 times 96$], [36],
    [24], [3], [12 attributes (6 + 6)], [$648 times 96$], [54],
    [30], [3], [12 attributes (6 + 6)], [$648 times 96$], [108],
    table.hline(),
  ),
  caption: [Blocks of the trials catalogue with uniform tables. The other blocks for 18 to 30 rules are {sponsor, scope} and {status, outcome}. Contexts are defined in @sec:contexts.],
) <tab:blocks>

== Enumeration

Within a block, every combination of the parent attributes is enumerated, $cal(P) = product_(a in B_p) D(a)$, and so is every combination of the child attributes, $cal(C) = product_(b in B_c) D(b)$. For $p in cal(P)$ and $c in cal(C)$:

- $"valid"_p (p) = 1$ when no parent rule of the block fails on $p$;
- $"valid"_c (c) = 1$ when no child rule of the block fails on $c$;
- for each parent–child rule $r$ of the block: $h^P_r (p) = 1$ when its parent part holds on $p$, and $h^C_r (c) = 1$ when its child part holds on $c$;
- the child $c$ *passes* with parent $p$ when
  $ "pass"(p, c) = "valid"_c (c) and not or.big_(r in R_"pc") (h^P_r (p) and h^C_r (c)). $ <eq:pass>

All of these are boolean arrays over $cal(P)$ or $cal(C)$, computed once with numpy. The network gives $P(p)$ for every parent combination and $P(c | p)$ for every child combination.

== Two target distributions <sec:targets>

Let
$ Z(p) = sum_(c in cal(C)) P(c | p) thin "pass"(p, c) $ <eq:z>
be the probability that one child drawn from the network passes the rules with parent $p$. A parent combination with $Z(p) > 0$ is *feasible*: it can have a child.

*Joint conditioning* is the network of @eq:network conditioned on "no rule fails anywhere in the family". For a parent with $n$ children,
$ Pr(p, c_1, dots, c_n) prop P(p) thin "valid"_p (p) product_(i=1)^n P(c_i | p) thin "pass"(p, c_i). $
Summing over the children gives
$ "Pr"_"joint" (p | n) prop P(p) thin "valid"_p (p) thin Z(p)^n. $ <eq:joint>
This is the distribution that rejection sampling of whole families would give.

*Sequential conditioning* (the default) draws the parent from the network conditioned on the parent rules and on feasibility, and only then the children:
$ "Pr"_"seq" (p | n) prop P(p) thin "valid"_p (p) thin [n = 0 or Z(p) > 0]. $ <eq:seq>

In both cases, given the parent, the children are independent and each is drawn from
$ Pr(c | p) = P(c | p) thin "pass"(p, c) \/ Z(p). $ <eq:child>

*Why sequential is the default.* Under joint conditioning, a parent's values depend on its number of children: parents for which few child combinations pass become rare as $n$ grows, because $Z(p)^n$ falls quickly (@sec:example). The parent table then no longer follows its frequency tables, and its distribution changes with the fan-out. Sequential conditioning keeps the distribution of a parent the same for every $n >= 1$. It also matches the GouDa pipeline: GouDa draws the parents first, and `make_consistent` redraws a violating row uniformly from the valid ones. With uniform tables and no dependencies, sequential conditioning gives the distribution that pipeline aims for, without any redraws. Both conditionings are exact, each for its own target, and the test suite checks both against brute force.

== Contexts <sec:contexts>

Computing $Z(p)$ for every $p in cal(P)$ costs $|cal(P)| dot |cal(C)|$ operations. It can be computed once per *context* instead. Let $X subset.eq B_p$ be the parent attributes on which a child attribute of the block depends in the network. The context of $p$ is
$ kappa(p) = ( (h^P_r (p))_(r in R_"pc"), thick (p_a)_(a in X) ). $
By @eq:pass, $"pass"(p, c)$ depends on $p$ only through the bits $h^P_r (p)$. By @eq:network, $P(c | p)$ depends on $p$ only through $(p_a)_(a in X)$. Hence $Z(p)$ and @eq:child depend on $p$ only through $kappa(p)$. The distinct contexts are found with `np.unique` over the rows of $kappa$. Their number is at most $min(|cal(P)|, 2^(|R_"pc"|) product_(a in X) d_a)$, and is usually much smaller: 108 contexts for the 648 parent combinations with 30 rules (@tab:blocks). For a context $u$, the vector
$ w_u (c) = P(c | u) dot "valid"_c (c) dot not or.big_(r : u_r = 1) h^C_r (c) $
holds the unnormalised child probabilities, and $Z_u = sum_c w_u (c)$.

== The algorithm

For every block:

+ Enumerate $cal(P)$ and $cal(C)$ (`np.indices`, first attribute slowest). Refuse the block if either has more than `max_combinations` elements.
+ Compute $P(p)$, $"valid"_p$, $"valid"_c$, $h^P$ and $h^C$, the contexts $kappa(p)$, and $Z_u$ for every context $u$.
+ *Parents.* Group the parent rows by their label: $min(n(t), 1)$ for sequential conditioning, $n(t)$ itself for joint. For each label compute the weights of @eq:seq or @eq:joint, and draw all parents of the group at once (`rng.choice`). The joint weights are computed in log space and shifted by their maximum before exponentiating, because $Z^n$ underflows for large $n$. If all weights of a group are zero, generation stops with an error naming the block: no parent combination passes the rules (and leaves a valid child combination). This happens only when the rules contradict each other.
+ *Children.* Group the children by the context of their parent, and draw all children of a group at once from $w_u \/ Z_u$.

The codes of all blocks are then put together into one parent table and one child table.

== A worked example <sec:example>

Take one parent attribute, population $in$ {adult, paediatric, mixed}, two child attributes, age_group $in$ {adult, paediatric} and consent $in$ {self, parental}, uniform frequency tables, and four rules of the catalogue:

#table(
  columns: (auto, 1fr),
  [PC2], [$not ("population" in {"paediatric"} and "age_group" in {"adult"})$],
  [PC3], [$not ("population" in {"adult"} and "age_group" in {"paediatric"})$],
  [C1], [$not ("age_group" in {"paediatric"} and "consent" in {"self"})$],
  [C6], [$not ("age_group" in {"adult"} and "consent" in {"parental"})$],
)

$cal(C)$ has four combinations, each with $P(c) = 1\/4$. C1 and C6 leave two valid ones: (adult, self) and (paediatric, parental). The three populations give three contexts:

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    table.hline(),
    table.header([*population*], [*Active rule*], [*Children that pass*], [*$Z$*], [*$"Pr"_"joint"$, $n = 1$*], [*$"Pr"_"joint"$, $n = 5$*]),
    table.hline(stroke: 0.5pt),
    [adult], [PC3], [(adult, self)], [1/4], [1/4], [1/34 = 0.029],
    [paediatric], [PC2], [(paediatric, parental)], [1/4], [1/4], [1/34 = 0.029],
    [mixed], [—], [both], [1/2], [1/2], [32/34 = 0.941],
    table.hline(),
  ),
  caption: [The example under joint conditioning. Under sequential conditioning every population has probability 1/3 for every $n$.],
) <tab:example>

Under joint conditioning, the weight of a population is $1/3 dot Z^n$. With five children a mixed trial is 32 times as likely as an adult one. Rejection sampling would almost only produce mixed trials. In 100,000 generated parents, datagen gives 0.030 / 0.030 / 0.941 for joint conditioning with $n = 5$, 0.251 / 0.247 / 0.502 with $n = 1$, and 0.334 / 0.331 / 0.335 under sequential conditioning for both values of $n$. In a mixed trial each child is (adult, self) or (paediatric, parental), each with probability 1/2.

== Correctness

*Proposition.* For every block, the parent combinations are distributed as in @eq:seq or @eq:joint, and given the parent, the children are independent and distributed as in @eq:child. Over all blocks, the clean data are distributed as the network conditioned on the rules (in the chosen sense), and they contain no violation.

*Argument.* The weights in step 3 are exactly the right-hand sides of @eq:seq and @eq:joint, evaluated for every element of $cal(P)$, and `rng.choice` draws proportionally to them. The weights in step 4 are $P(c | p) thin "pass"(p, c)$, because $w_u = w_(kappa(p))$ (@sec:contexts). A child with $"pass"(p, c) = 0$ has weight 0, and a parent with $"valid"_p (p) = 0$ has weight 0, so no rule can fail. A parent with children always has $Z(p) > 0$, so step 4 always has a valid child to draw. Independence of the blocks (@sec:blocks) extends the result to the whole schema.

The test suite checks this empirically: on a small schema, the frequencies of 200,000 draws match the brute-force conditional distribution (total variation below 0.01) for both conditionings; and for every rule prefix 6…30 of the catalogue, under uniform, skewed and correlated settings, the clean data have 0 violations.

*Parents without children.* Under sequential conditioning a childless parent is drawn with weight $P(p) "valid"_p (p)$, without the feasibility factor. It may therefore take a combination that would leave no valid child. The rules only constrain rows that exist, so such a parent is consistent. This is intended, and tested.

== Cost and limits <sec:cost>

Setting up a block costs $O(|cal(P)| + |cal(C)| + U |cal(C)|)$ with $U$ contexts. Drawing costs $O(N + M)$ for $N$ parents and $M$ children, plus one `rng.choice` call per group. The guard refuses a block with more than `max_combinations` ($10^6$) combinations on either side, or with $U |cal(C)| > 10 dot$ `max_combinations`, and names the block. The remedy is to split the block by removing a rule or a dependency. The largest block of the catalogue has $648 times 96$ combinations, far below the limit.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    table.hline(),
    table.header([*Rules*], [*Parents*], [*Clean data*], [*Errors*], [*Statistics*], [*Total*]),
    table.hline(stroke: 0.5pt),
    [18], [1,000], [2 ms], [11 ms], [3 ms], [17 ms],
    [18], [10,000], [17 ms], [59 ms], [31 ms], [0.11 s],
    [18], [100,000], [0.17 s], [0.57 s], [0.24 s], [0.99 s],
    [30], [1,000], [3 ms], [12 ms], [4 ms], [21 ms],
    [30], [10,000], [18 ms], [67 ms], [38 ms], [0.13 s],
    [30], [100,000], [0.17 s], [0.61 s], [0.40 s], [1.19 s],
    table.hline(),
  ),
  caption: [Generation time with 5 children per parent and 5% errors (best of three runs, without writing files). The total includes a setup of 2–3 ms.],
)

== The rules change the frequencies

The frequency tables describe the network *before* the rules. The data follow the network *conditioned on* the rules, so their realised frequencies differ from the tables wherever the rules exclude values. The statistics report both, with their total variation distance (@sec:stats). With 30 rules and uniform tables, the largest gap is `placebo`: the table says 50% Yes, the data have 17.7% (total variation 0.323). The reason is that PC4 and PC9 forbid placebos in trials without a control or with an active control, so only the children of placebo-controlled trials can have Yes. Those are a minority, and C2 removes paediatric participants from them as well.

This is a property of the target, not an error of the sampler. Without rules the realised frequencies match the tables, which is tested. A spec that needs given realised frequencies would have to calibrate the tables, which is listed as an extension in @sec:limits.

= Error injection <sec:errors>

Errors are injected into a copy of the clean data by `inject_errors(schema, rules, clean, error_spec, rng)`. `ErrorSpec` controls where they go, how many there are, which wrong value they take, and whether a rule can see them. It is a separate specification: saved alone as `error_spec.json`, and stored next to the dataset spec in `dataset.json`.

== Scopes, rates and counts

A *scope* is one attribute in its table. The eligible scopes are the attributes used by the rules (`eligible="rule_attributes"`, the default), all attributes (`"all"`), or a given list. Keys and foreign keys are never eligible. The rate of a scope is taken from the first that is set of `rate_by_attribute`, `rate_by_table` and `rate`. The number of errors $k$ in a scope with $N$ rows and rate $r$ is
$ k = floor(r N + 1\/2) quad (#[`count="exact"`, the default]) quad "or" quad k tilde "Binomial"(N, r) quad (#[`"bernoulli"`]). $
With exact counts the realised rate equals the nominal rate up to the rounding per scope: 5% of 1,000 parents gives exactly 50 errors in every parent attribute. For very small tables the rounding shows. The benchmark smoke test with 50 parents and 150 children gives 3 + 8 errors per attribute pair, 5.54% instead of 5%.

== Location

`row_weights` concentrate errors in chosen rows. Each entry gives a table, a condition `where` (attribute → values) and a `factor`. The weight of a row is the product of the factors of the entries that match it, judged on the clean data. For the child table, the condition may also name parent attributes, and is then judged on the joined row: for example, "registrations of phase I trials get three times as many errors". A factor of 0 excludes rows. The $k$ rows of a scope are drawn without replacement with probabilities proportional to their weights (numpy's sequential weighted draw, so the inclusion probabilities are close to, but not exactly, proportional to the weights).

== Which wrong value

The wrong value comes from a *mechanism*. Each mechanism defines, for a clean value $v_i$, a set of allowed dirty values and their weights:

- `random`: every other value of the domain, with equal weight;
- `neighbour`: the values next to $v_i$ in the domain order, $v_j$ with $|i - j| = 1$ (typing errors on an ordinal scale);
- `confusion`: a matrix of weights `confusion[attr][clean][dirty]` given by the user, with its diagonal set to zero.

`mechanism` gives a weight $lambda_m$ per mechanism. For each picked row, a mechanism is drawn with probability proportional to $lambda_m$ times the indicator that it has an allowed value for that row (`confusion` has none for a clean value that its matrix does not list). The dirty value is then drawn from that mechanism's weights. *No mechanism can return the clean value*, which the code asserts. This is why the realised rate equals the nominal one, unlike in GouDa.

== Aiming errors at the rules <sec:detectable>

A repair algorithm can only find an error that makes some rule fail. `detectable_share` controls the share of such errors. For a row $i$ of scope $a$ and a rule $r$ that involves $a$, let $F_r (i) subset.eq D(a)$ be the values that make $r$ fail when written into the cell, with every other cell clean:

- for a parent attribute: $v in S_a (r)$, the other parent atoms of $r$ hold on $t_i$, and if $r$ has a child part, at least one child of $t_i$ satisfies it;
- for a child attribute: $v in S_a (r)$ and every other atom of $r$ holds on the joined row of child $i$.

Let $A(i)$ be the values allowed by the mechanisms with positive weight, and $T$ the target rules (all rules, or `target_rules`). Then
$ "Det"(i) = A(i) inter union.big_(r in T) F_r (i), quad "Sil"(i) = A(i) without union.big_(r in.rev a) F_r (i). $
The second union runs over every rule $r$ that involves $a$. A *detectable* value makes a target rule fail. A *silent* value makes no rule fail. A value that only makes a non-target rule fail is neither, and is never used.

With $k$ errors in the scope, $k_"det" = floor(s k + 1\/2)$ rows get a detectable value and $k - k_"det"$ a silent one, where $s$ is `detectable_share`. Detectable rows are picked first, among the rows with $"Det"(i) != emptyset$, then silent rows among the remaining rows with $"Sil"(i) != emptyset$. A row is never picked twice. When too few rows of a kind exist, the *shortfall* is recorded in the statistics. With `on_shortfall="fill"` the missing errors are taken from the other kind, so the error count stays exact and the share moves.

Shortfalls are real properties of the rules. With all 30 rules, `age_group`, `consent`, `placebo` and `outcome` have no silent wrong value in any row: the rules determine them completely (consent follows age group through C1 and C6, placebo follows the control through PC1, PC4 and PC9, outcome follows status through CS1–CS3). Every wrong value in these attributes makes a rule fail.

== Final classification

$F_r (i)$ is judged on the clean data, one cell at a time. After all errors are in, two errors in the same family can mask or expose each other. Every error is therefore classified again on the dirty data, with the definition of `detectable_recall` in `algo1_bench`:

- a parent error on $a$ is detectable when a parent rule involving $a$ fails on the dirty parent, or another rule involving $a$ fails on the dirty parent joined with any of its dirty children;
- a child error on $a$ is detectable when a child or parent–child rule involving $a$ fails on the dirty parent joined with that dirty child.

The log keeps both the intended kind and the final classification. @tab:share shows the effect on the catalogue.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    table.hline(),
    table.header([*Setting*], [*Errors (rate)*], [*Intended*], [*Detectable after injection*]),
    table.hline(stroke: 0.5pt),
    [no target (default)], [2,400 (5.00%)], [—], [1,782 (74.3%)],
    [`detectable_share=1`], [2,400 (5.00%)], [2,400 detectable], [97.8%],
    [`detectable_share=0.5`], [1,900 (3.96%)], [1,200 detectable, 700 silent], [63.1%],
    [`detectable_share=0`], [1,400 (2.92%)], [1,400 silent], [3.0%],
    [`detectable_share=0`, fill], [2,400 (5.00%)], [1,000 detectable, 1,400 silent], [44.0%],
    table.hline(),
  ),
  caption: [Errors aimed at the rules: 30 rules, 1,000 parents, 5 children each, 5% per attribute, seed 0. With a share below 1, the four fully determined child attributes cannot get silent errors (shortfall), so their errors are skipped unless `on_shortfall="fill"`.],
) <tab:share>

== The error log

`errors.csv` has one row per error: table, key, attribute, clean value, dirty value, mechanism, intended kind (`any`, `detectable` or `silent`) and the final classification. It is the ground truth for evaluating a repair. The tests check that it matches the difference between the clean and the dirty files exactly.

= Reproducibility <sec:seeds>

`numpy.random.SeedSequence(seed).spawn(3)` gives three independent streams: one for the fan-out, one for the clean values and one for the errors. Each stage only reads its own stream. The generator uses the first two. The injector uses the third, unless `ErrorSpec.seed` gives the errors a seed of their own. This gives three guarantees, all tested on the bytes of the files:

+ The same specs give byte-identical files (same SHA-256).
+ Running `generate` and `inject` separately, with the clean data written to disk and read back in between, gives the same files as running them in one go.
+ The `ErrorSpec` never changes the clean files. An error-rate sweep therefore runs on the same clean data. The benchmark uses this: it gives every dataset a seed that hashes the parents, children, rules and replicate, but not the error rate.

`dataset.json` records the version and commit, both specifications, the timings, the SHA-256 of every file and the statistics, so a folder can be checked and regenerated later.

= Learning from real data <sec:fit>

`fit(parents, children, fk)` learns a `DatasetSpec` from a real parent and child table:

+ *Schema.* Taken from the caller, or inferred: every column except the keys becomes an attribute with the sorted distinct values as its domain. A column with more than `max_domain` (50) distinct values is left out and reported. A child column whose name is also a parent column is renamed `<column>_<child>`. An empty value becomes the value `<missing>`. Children without a parent are dropped and counted.
+ *Frequency tables.* Counts plus a Laplace constant $alpha$ (default 1).
+ *Structure.* By default, a Chow–Liu tree: the spanning tree with maximal mutual information. The parent attributes are connected first, using the parent rows. The child attributes are then attached one by one to the attribute already in the tree with which they share the most information (Prim's algorithm), using the joined rows. The edges point away from the root, so they never run from child to parent. An edge is kept only when a G-test rejects independence:
  $ G = 2 N thin "MI"(a, b) tilde chi^2_((d_a - 1)(d_b - 1)), quad "keep if" p < 10^(-3). $
  The p-value uses the Wilson–Hilferty approximation, so no scipy is needed. A given list of edges, or `"none"`, can replace the learned structure.
+ *Conditional tables.* Counts plus $alpha$ per observed parent configuration. Configurations never seen fall back to the frequency table.
+ *Fan-out.* The observed numbers of children, including 0, as an `Empirical` distribution.
+ *Rules.* Rules are not learned. Given rules are checked on the input, and their violations are reported. The generator conditions on them anyway.

The result is an ordinary spec that can be edited, saved as JSON and generated from, at any size. The tests check that a known structure is recovered: the conditional tables within a total variation of 0.02 at 100,000 rows, the Chow–Liu edges of a known tree, and no edges for independent attributes.

= Statistics and output <sec:stats>

`generate` computes the statistics below and stores them in `dataset.json`. `python -m datagen stats DIR` recomputes them from the files.

- *Rows* per table, and *children per parent*: mean, minimum, maximum, share with 0, 1 and at least 2 children, and the histogram.
- *Frequencies*: for every attribute, the target $q$ (the network marginal without rules) and the realised frequencies $hat(q)$, with the total variation distance $"TV"(q, hat(q)) = 1/2 sum_v |q(v) - hat(q)(v)|$.
- *Association*: Cramér's V of every network edge,
  $ V = sqrt(chi^2 / (N dot (min(r, c) - 1))), $
  on the parent rows for parent edges and on the joined rows otherwise. Empty rows and columns of the contingency table are left out.
- *Violations* in the clean data (always 0) and, after injection, in the dirty data, in total and per rule.
- *Errors* (after injection): count and rate, overall and per table and attribute; the number and share of detectable errors; counts per mechanism and per intended kind; the shortfalls.

#figure(
  table(
    columns: (auto, 1fr),
    table.hline(),
    table.header([*File*], [*Contents*]),
    table.hline(stroke: 0.5pt),
    [`<Parent>_clean.csv`], [Key and parent attributes],
    [`<Child>_clean.csv`], [Key, foreign key and child attributes],
    table.hline(stroke: 0.5pt),
    [`<Parent>_dirty.csv`, `<Child>_dirty.csv`], [The same columns with the errors (after `inject`)],
    [`errors.csv`], [The error log (after `inject`)],
    table.hline(stroke: 0.5pt),
    [`dataset.json`], [Format version 3; generator name, version, git commit and whether the checkout was clean; name and labels; the full dataset and error specifications; every file with its role, rows and SHA-256; timings; statistics. Written last, so a folder without it is incomplete.],
    table.hline(),
  ),
  caption: [A dataset folder (`FORMAT.md`). Keys are written as strings, and the files are read back with `keep_default_na=False` so that values such as `none` stay strings. Writing clean data into a folder removes dirty files left by an earlier `inject`. A reader refuses other format versions and files whose SHA-256 changed.],
)

The command line offers the same functions:

```
python -m datagen example trials --rules 18 --parents 1000 --children 5 --out spec.json
python -m datagen error-spec --rate 0.05 --out error_spec.json
python -m datagen generate spec.json --out out/clean --seed 3
python -m datagen inject out/clean --errors error_spec.json --out out/trials
python -m datagen generate spec.json --out out/trials --seed 3 --errors error_spec.json
python -m datagen stats out/trials
python -m datagen regenerate out/trials --out out/again
python -m datagen sweep sweep.json --out out/sweep
python -m datagen fit parent.csv child.csv --fk trial_id --rules spec.json --out fitted.json
```

A *sweep file* holds a base dataset spec, a base error spec and a list of points. Each point names its folder, keeps a prefix or a named subset of the rules, overrides fields of the two specs and adds labels. `sweep` writes one folder per point and skips folders that are already complete, so an interrupted sweep can be restarted.

= Validation

The test suite (88 tests, `python -m pytest`) checks every property claimed above:

#table(
  columns: (auto, 1fr),
  table.hline(),
  table.header([*Area*], [*What is checked*]),
  table.hline(stroke: 0.5pt),
  [Sampler (18)], [0 violations for every rule prefix 6–30; brute-force agreement for both conditionings; that the conditionings differ when the children matter; childless parents; mixed fan-out; marginals without rules; shapes of the frequency tables; dependency strength vs association; parent→child dependencies; refused edges and cycles; blocks; the size guard; contradictory rules; seeds.],
  [Errors (24)], [Exact counts per scope and no unchanged values; keys untouched; log equals the data difference; rate hierarchy; child rate 0; `eligible="all"`; Bernoulli rate; clean data unchanged by error settings; each mechanism; row weights (including 0 and parent conditions on child rows); classification equal to `detectable_recall`; detectable share 0 and 1; shortfall and fill; target rules; invalid specs; the injector on hand-made clean data; generator and injector do not import each other.],
  [Fan-out (11)], [Each distribution's mean, support and zeros; zero- and one-child parents in the output.],
  [Fit (8)], [Recovery of known tables and Chow–Liu edges; dropping independent attributes; fitted specs generate similar data; empirical fan-out with childless parents; rule-violation report; schema inference.],
  [Input/output (16), sweeps (9), CLI (2)], [JSON round trips of both specs; the contents of `dataset.json`; clean and dirty folder round trips; changed files and other format versions refused; regenerating from `dataset.json` gives the same bytes; same seed gives the same bytes; error change keeps the clean bytes; two steps equal one step; error seed; the commit is never taken from an enclosing repository; sweep overrides, rule selection, refused points, restarts, and a sweep equal to `generate` plus `inject`; the full command-line workflow.],
  table.hline(),
)

*End to end.* With the trials catalogue, 30 rules, 1,000 parents, 5 children each and 5% random errors: 400 of 8,000 parent cells and 2,000 of 40,000 child cells are dirty (both exactly 5.00%), the clean data have 0 violations, the dirty data 2,752, and 74.3% of the errors are detectable.

= Use in the benchmark

The benchmark (`algo1_bench`, a separate repository) does not import datagen. It writes a sweep file with one point per benchmark configuration: its own catalogue schema and the first $r$ rules, uniform tables, `Fixed(children)`, the error spec `rate = error_pct/100`, and a seed per point. datagen turns that file into folders, and the benchmark reads the folders with pandas and json alone: it checks the format version and the SHA-256 of every file, takes the schema and rules from `dataset.json`, and records the datagen version and commit with every result. A datagen folder skips `load_gouda` and `make_consistent`, and passes every other check of `gendb/validate.py`: two tables, correct keys, constant rules, finite domains, and a valid ground truth.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    table.hline(),
    table.header([*Dataset*], [*Realised errors*], [*Clean violations*], [*F1*], [*Detectable recall*]),
    table.hline(stroke: 0.5pt),
    [1,000 parents, 5 children, 5%, 18 rules], [5.00%], [0], [0.356], [0.570],
    [1,000 parents, 5 children, 5%, 30 rules], [5.00%], [0], [0.622], [0.720],
    [10,000 parents, 5 children, 5%, 18 rules], [5.00%], [0], [0.347], [0.557],
    table.hline(),
  ),
  caption: [Algorithm 1 on datagen data. More rules make more errors visible, so recall rises with the number of rules. The repaired data have 0 violations in every run.],
)

= Limitations and extensions <sec:limits>

- *Two tables.* The schema is a tree, but the sampler handles one parent and one child. A deeper tree would apply the same construction level by level, with the context of a row including the relevant bits of all its ancestors.
- *Constant rules only.* Variable rules (equality between attributes) and denial constraints across siblings are not supported. Rules across siblings would break the independence of children given their parent.
- *Enumeration.* Blocks must be small enough to enumerate. Large domains or long chains of rules can exceed the limit; a Gibbs sampler or a junction-tree method would scale further, at the cost of exactness or simplicity.
- *Network.* One dependency per attribute; no child→parent edges; children are independent given their parent; fitting learns a tree.
- *Realised frequencies.* The frequency tables are targets before the rules. Iterative proportional fitting of the tables would make the realised frequencies match a target after the rules.
- *Errors.* Only value substitutions. Missing values, duplicate rows, wrong foreign keys and errors in keys are not generated, because Algorithm 1 assumes correct keys and complete rows.
- *Row selection.* Weighted selection without replacement is numpy's sequential draw; the inclusion probabilities are only approximately proportional to the row weights.
