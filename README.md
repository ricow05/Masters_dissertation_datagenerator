# datagen

A data generator for parent and child tables whose clean data satisfies a set of constant
selection rules exactly, and a separate error injector you can control. The generator can
also learn the distributions from real tables.

The two halves are separate subpackages that never import each other:

* `datagen.generator`: `DatasetSpec` → `generate` → `CleanData` (and `fit`);
* `datagen.injector`: `ErrorSpec` → `inject_errors(schema, rules, clean, error_spec, rng)`. It
  needs only the schema, the rules and the clean data, so it also works on data that
  datagen did not generate.

`datagen.inject(clean_data, error_spec)` connects them. The same clean data can get any number
of error specs.

The package is standalone and depends only on numpy and pandas. Its output is a folder of CSV
files with a `dataset.json` that records the datagen version and commit and every parameter
of the data. **`FORMAT.md` is the contract** for programs that use the data: they read the
folder with a CSV and a JSON reader and never import datagen.

```
pip install -e .            # or: pip install -r requirements.txt
python -m pytest -q         # ~90 tests, ~12 s
```

Releases are git tags (`v0.3.0`, ...). The version is in `pyproject.toml` and
`datagen/__init__.py` and is written into every `dataset.json`, together with the commit when
datagen runs from its git checkout. Generate data you report on from a clean, tagged checkout.

Two documents explain the generator in more depth:

- `docs/generator.typ`: how it works, with the sampler, the error injector and the fitting
  derived step by step. Compile it with `typst compile docs/generator.typ`.
- `demo.ipynb`: every feature on the trials example, with the statistics of the generated
  data. It is stored with its outputs and writes to `out/demo/` when run (Run All, about 20 s).

To run the notebook, use a venv in this folder with the `notebook` extra (matplotlib, ipykernel,
Jupyter Notebook):

```
py -3.14 -m venv .venv
.venv/Scripts/python -m pip install -e ".[test,notebook]"
.venv/Scripts/jupyter notebook demo.ipynb     # or select .venv/Scripts/python.exe as the kernel in VS Code
```

## Quick start

```
python -m datagen example trials --rules 18 --parents 1000 --children 5 --out spec.json
python -m datagen error-spec --rate 0.05 --out error_spec.json        # then edit it for the other settings
python -m datagen generate spec.json --out out/clean18 [--seed 7] [--parents 5000]
python -m datagen inject out/clean18 --errors error_spec.json --out out/trials18 [--seed 3]
python -m datagen generate spec.json --out out/trials18 --errors error_spec.json   # both steps at once
python -m datagen stats out/trials18 [--json]
python -m datagen regenerate out/trials18 --out out/again               # same bytes from dataset.json
python -m datagen sweep sweep.json --out out/sweep [--force]           # one folder per point
python -m datagen fit parent.csv child.csv --fk trial_id --out fitted.json [--rules spec.json]
```

`generate` and `inject` also take `--name NAME` and `--label key=value` (repeatable; the value
is read as JSON when it parses, so `replicate=2` stores a number). Both are stored in
`dataset.json`; datagen does not interpret them. A sweep file (format in `FORMAT.md`) holds a
base spec and a list of points with their overrides, and is how a benchmark asks for its data.

The same steps in Python:

```python
from datagen import generate, inject, write, Skewed, Depends, Poisson, ErrorSpec, RowWeight
from datagen.examples import trials

spec = trials.spec(
    count=18,                                        # first 18 of the 30 trial rules
    parents=1000,
    fanout=Poisson(3),                               # children per parent; includes 0 and 1
    marginals={"phase": Skewed(1.0)},                # phase I most frequent
    dependencies=[Depends("scope", "sponsor", 0.6)],
)
clean = generate(spec)                # clean.clean (coded), clean.stats
errors = ErrorSpec(rate=0.05, rate_by_table={"Trial": 0.10},
                   row_weights=[RowWeight("Trial", {"phase": "I"}, 3.0)],
                   detectable_share=0.5)
data = inject(clean, errors)          # data.clean, data.dirty (coded), data.errors (DataFrame), data.stats
write(data, "out/example")            # CSVs + errors.csv + dataset.json
frames = data.frames()                # {"Trial_clean", "Trial_dirty", "Registration_clean", "Registration_dirty"}
write(clean, "out/example_clean")     # clean data only
```

The injector on clean data from elsewhere:

```python
from datagen.data import Coded
from datagen.injector import inject_errors

clean = Coded.from_frames(schema, parent_df, child_df)       # string tables, keys and foreign key included
injection = inject_errors(schema, rules, clean, ErrorSpec(rate=0.05), rng=0)
injection.dirty, injection.log, injection.shortfalls
```

A schema of your own:

```python
from datagen import DatasetSpec, Rule, Fixed, two_tables

schema = two_tables("Shop", "shop_id", {"kind": ("food", "tech"), "size": ("S", "M", "L")},
                    "Order", "order_id", {"pay": ("cash", "card"), "big": ("Yes", "No")})
rules = [Rule.of({"kind": "food", "big": "Yes"}, "R1", "food shops do not take big orders"),
         Rule.of({"size": "S", "pay": "card"}, "R2")]
spec = DatasetSpec(schema, rules, fanout=Fixed(4), parents=500, seed=1)
```

## Model

**Schema.** There is one parent table and one child table, linked by a foreign key. Every
attribute has a finite, ordered domain of strings. Attribute names must be unique across both
tables and must not clash with a key column. The schema is stored as a list of tables, so a
deeper tree can be added later. For now, any other shape raises `NotImplementedError`.

**Rules.** A rule `NOT(a1 ∈ S1 ∧ … ∧ ak ∈ Sk)` forbids that combination of values. It is a
*parent*, *child* or *parent-child* rule, depending on which tables its attributes come from.
A parent-child rule is evaluated on every joined (parent, child) row.

**Frequency tables (marginals).** Each attribute has one, defined over the domain order.
Unlisted attributes are uniform.

| class | meaning |
|---|---|
| `Uniform()` | equal weights |
| `Normal(mean=None, sd=1.0)` | normal density at the value indices; `mean` is an index or a value (default: centre) |
| `Skewed(s=1.0, towards="first")` | Zipf, weight 1/rank^s; `towards` is `"first"`, `"last"` or a value |
| `Frequencies({value: weight})` | explicit weights; missing values get 0 |

**Correlation (Bayesian network).** An attribute can have at most one dependency:

* `Depends(attr, on, strength=ρ, mapping=None)` gives
  `P(attr | on=a) = (1−ρ)·marginal(attr) + ρ·[attr = mapping(a)]`. By default, `mapping`
  aligns the two domains by rank. A mapping you give must cover every value of `on`.
* `CPT(attr, on=(a, b, …), rows={(va, vb, …): {value: weight}})` gives the table explicitly.
  Rows you leave out fall back to the marginal.

Edges may run inside a table or from a parent attribute to a child attribute, never from
child to parent. The graph must be acyclic.

**Children per parent (fan-out).** The options are `Fixed(k)`, `UniformRange(low, high)`
(inclusive), `Poisson(lam)`, `Geometric(p)` (mean (1−p)/p) and `Empirical({k: weight})`.
`Fixed(5)` reproduces the GouDa setting.

## How clean data is drawn

Clean data comes from the network *conditioned exactly on the rules*. There is no rejection
and no repair step.

1. **Blocks.** Attributes are grouped into blocks: connected components of the graph that
   links two attributes when they share a rule or a network edge. Blocks are sampled
   independently. Without network edges they are the attribute classes of Algorithm 1.
2. **Enumeration.** In each block, all parent combinations and all child combinations are
   enumerated. A parent combination is *valid* if it passes the parent rules. It is
   *feasible* if at least one child combination passes the child and parent-child rules with
   it. This is computed once for each *context* (which parent-child rules have their parent
   part satisfied, plus the values of parent attributes that child attributes depend on).
3. **Parents.** The `conditioning` setting chooses how parents are weighted:
   * `"sequential"` (default): weight P(p) · valid(p), times feasible(p) when the parent has
     at least one child. A parent's distribution then does not depend on how many children
     it has.
   * `"joint"`: weight P(p) · valid(p) · Z(p)^n, where Z(p) is the probability that one child
     drawn from the network passes the rules with p, and n is the number of children. This
     is exactly the distribution that rejection sampling of whole families would give, so
     parents with many children shift towards permissive values.
4. **Children.** Each child is drawn from P(c | p), restricted to the combinations that pass
   the rules with its parent.

The tests check both conditionings against a brute-force conditional distribution.

The rules change the realised frequencies compared with the targets. For example, with the
trial rules, `placebo` is almost determined by `control`. `stats` reports this gap as the
total variation distance (TVD) per attribute.

`max_combinations` (default 10⁶ per side of a block) guards against blocks too large to
enumerate. The trials schema needs at most 5,832 × 576.

## Errors

Errors come from the injector (`datagen.injector`), separately from the clean data. An
`ErrorSpec` is saved as `error_spec.json`. Its fields (defaults in brackets):

| field | meaning |
|---|---|
| `rate` [0.05], `rate_by_table`, `rate_by_attribute` | error rate per (table, attribute) scope; attribute overrides table, which overrides `rate` |
| `eligible` [`"rule_attributes"`] | which attributes can get errors: `"rule_attributes"`, `"all"`, or a list. Keys and foreign keys never get errors |
| `count` [`"exact"`] | `"exact"`: exactly round(rate·N) cells per scope. `"bernoulli"`: Binomial(N, rate) cells |
| `row_weights` [none] | `RowWeight(table, where={attr: values}, factor)` multiplies the chance that matching rows get an error; factor 0 excludes them. For child rows, `where` may name parent attributes |
| `mechanism` [`{"random": 1}`] | mixture of `random` (any other value), `neighbour` (adjacent value in the domain order) and `confusion` |
| `confusion` | `{attr: {clean: {dirty: weight}}}`, used by the `confusion` mechanism |
| `detectable_share` [None] | if set, this share of each scope's errors gets a value that makes a rule fail, and the rest a value that keeps every rule satisfied (judged on the clean data) |
| `target_rules` [all] | rule names that count for `detectable_share` |
| `on_shortfall` [`"skip"`] | when too few cells can take the wanted kind of value: `"skip"` makes fewer errors and reports the shortfall; `"fill"` uses the other kind |
| `seed` [None] | seed of the errors; None takes the error stream of the clean data's seed (see Seeds) |

A dirty value always differs from the clean one, so the realised rate equals the nominal
rate. With GouDa, a nominal 5% gave 2.8%.

After injection, each error is classified again on the dirty data, using the same definition
as `detectable_recall` in `algo1_bench`. This matters because errors in the same family can
mask or expose each other.

## Seeds

The spec's `seed` is split by `numpy.random.SeedSequence` into separate streams for fan-out,
clean data and errors. The generator uses the first two. The injector uses the third, unless
`ErrorSpec.seed` (or `inject --seed`) gives the errors a seed of their own. The same specs and
seeds give byte-identical files, whether you run `generate` and `inject` separately or in one
go. **The error settings never change the clean files**, so an error sweep runs on the same
clean data.

## Learning from real data

```python
from datagen import fit, save_spec
result = fit(parents_df, children_df, fk="trial_id", rules=my_rules)   # schema=None: inferred
save_spec(result.spec, "fitted.json"); print(result.report)
```

* **Domains** come from the schema you pass, or are inferred from the data: sorted distinct
  values, with NaN becoming `"<missing>"`. Columns with more than `max_domain` (50) values
  are dropped. Child columns whose names clash with parent columns are renamed
  `<column>_<child>`.
* **Marginals and CPTs** are counts plus `alpha` (1), i.e. Laplace smoothing.
* **Structure**:
  * `"chow-liu"` (default) builds a maximum mutual-information tree. Parent attributes are
    linked first, using the parent rows. Child attributes are then attached using the joined
    rows. An edge is kept only if a G-test rejects independence at `significance` (1e-3).
  * `"none"` makes all attributes independent.
  * A list of `(on, attr)` pairs fixes the structure.
* **Fan-out** becomes `Empirical`, including parents with no children. Orphan children are
  dropped and counted.
* **Rules** are not learned. Rules you pass are checked on the input, and the violations are
  reported. The generator still conditions on them.

On the CLI, CSVs are read as strings, with no automatic NaN detection. Use `--na-values` to
mark strings as missing. The report is written next to the spec as `<name>_report.json`.

## Output folder

The full description is in `FORMAT.md` (format 3).

```
<Parent>_clean.csv  <Child>_clean.csv      key, (foreign key,) attributes (keys "0".."n-1")
-- after inject:
<Parent>_dirty.csv  <Child>_dirty.csv      the same columns, with the errors
errors.csv          table, key, attribute, clean, dirty, mechanism, intended, detectable
-- always, written last:
dataset.json        generator version and commit, name, labels, the full spec and error spec,
                    every file with its role, rows and sha256, timings, stats
```

Writing clean data into a folder removes dirty files left there by an earlier `inject`.
`datagen.read` refuses folders of other format versions and files whose sha256 changed.
`load_spec` and `load_error_spec` accept a `dataset.json` as well as a separate spec file.

`stats` (in `dataset.json`, or recomputed with `datagen stats DIR`) contains:

* row counts and the children-per-parent histogram;
* target vs realised marginals (TVD);
* Cramér's V per network edge;
* violations in the clean data (always 0) and, after injection, the dirty data, overall and
  per rule;
* after injection: errors per table, attribute, mechanism and intent, the detectable share,
  and any shortfalls.

`datagen.read(DIR)` loads a folder back (`description`, `spec`, `error_spec`, coded
`clean`/`dirty`, `errors`, `frames`, `name`, `labels`, `generator`; the error fields are None
for clean data). `datagen.read_clean(DIR)` gives its clean
data as `CleanData`, ready for `inject`.

## Spec and error spec

Stored in `dataset.json` (`spec`, `error_spec`), or alone as `spec.json` and `error_spec.json`.

```json
{
  "format_version": 2,
  "schema": {"tables": [
    {"name": "Trial", "key": "trial_id", "attributes": {"phase": ["I", "II", "III", "IV"], "...": []}},
    {"name": "Registration", "key": "reg_id", "parent": "Trial", "foreign_key": "trial_id",
     "attributes": {"age_group": ["adult", "paediatric"], "...": []}}]},
  "rules": [{"name": "P1", "atoms": {"design": ["single-arm"], "control": ["active", "placebo"]},
             "meaning": "A single-arm trial has no comparator arm."}],
  "marginals": {"phase": {"type": "skewed", "s": 1.0, "towards": "first"},
                "blinding": {"type": "normal", "mean": "double", "sd": 0.8}},
  "dependencies": [
    {"type": "depends", "attr": "scope", "on": "sponsor", "strength": 0.4},
    {"type": "cpt", "attr": "arm", "on": ["design"], "rows": [{"given": ["crossover"], "p": {"both": 1}}]}],
  "fanout": {"type": "poisson", "lam": 2.5},
  "parents": 1000, "seed": 0, "conditioning": "sequential", "max_combinations": 1000000
}
```

```json
{
  "format_version": 1,
  "rate": 0.05, "rate_by_table": {}, "rate_by_attribute": {"sex": 0.1},
  "eligible": "rule_attributes", "count": "exact",
  "row_weights": [{"table": "Trial", "where": {"phase": ["I"]}, "factor": 3.0}],
  "mechanism": {"random": 0.7, "neighbour": 0.3}, "confusion": {},
  "detectable_share": null, "target_rules": null, "on_shortfall": "skip", "seed": null
}
```

Fan-out types: `fixed {k}`, `uniform_range {low, high}`, `poisson {lam}`,
`geometric {p}`, `empirical {weights: {"0": w, "1": w, …}}`.

## Layout

```
datagen/
  schema.py rules.py data.py           shared: schema, rules, coded data
  generator/                           clean data; never imports the injector
    spec.py                            DatasetSpec (spec.json)
    distributions.py network.py fanout.py
    sampler.py                         exact conditioning on the rules
    generate.py stats.py fit.py        generate -> CleanData, its statistics, learning
  injector/                            errors; never imports the generator
    spec.py                            ErrorSpec, RowWeight (error_spec.json)
    inject.py stats.py                 injection, classification, error statistics
  pipeline.py                          inject(CleanData, ErrorSpec) -> DirtyData
  io.py stats.py                       folders (dataset.json), terminal summary
  provenance.py                        version and git commit for dataset.json
  sweep.py                             sweep files: many folders in one run
  __main__.py                          CLI
  examples/trials.py                   Trial/Registration schema and the 30 rules
tests/
FORMAT.md                              the folder format: the contract with readers
docs/generator.typ                     how the generator works
demo.ipynb                             walkthrough with outputs
```
