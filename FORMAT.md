# datagen dataset folder, format 3

This file is the contract between datagen and the programs that use its output. A reader
needs only a CSV reader and a JSON reader; it never imports datagen. A change that breaks a
reader written against this file raises `format_version`.

## Folder layout

```
<name>/
  dataset.json              what the folder holds and how it was made (written last)
  <Parent>_clean.csv        clean parent table
  <Child>_clean.csv         clean child table
  <Parent>_dirty.csv        dirty parent table        (only when errors were injected)
  <Child>_dirty.csv         dirty child table         (only when errors were injected)
  errors.csv                one row per injected error (only when errors were injected)
```

`dataset.json` is written after all CSV files, and removed first when a folder is rewritten.
**A folder without `dataset.json` is incomplete and must not be used.** Find files through
`dataset.json` → `files` (by `table` and `role`), not by guessing names.

## dataset.json

| field | meaning |
|---|---|
| `format` | always `"datagen-dataset"` |
| `format_version` | `3`. Refuse any other value. |
| `generator` | `{"name": "datagen", "version": "0.3.0", "commit": "<40 hex>" or null, "dirty": bool or null}`. `commit` is the git commit of the datagen checkout that made the data, `dirty` whether that checkout had uncommitted changes. Both are null when datagen did not run from its own git checkout. |
| `created` | UTC time, ISO 8601 |
| `name` | dataset name (by default the folder name) |
| `labels` | free key-value pairs given by whoever generated the data (e.g. `{"replicate": 2}`); datagen does not interpret them |
| `spec` | the full dataset spec: schema, rules, distributions, number of parents, seed (below) |
| `error_spec` | the full error spec, or `null` for clean data only |
| `files` | `{file name: {"table", "role", "rows", "sha256"}}`, `role` one of `clean`, `dirty`, `error_log` (the error log has no `table`) |
| `timings` | seconds per step, e.g. `{"clean_s": 0.4, "errors_s": 0.1, ...}` |
| `stats` | statistics of the data: rows, children per parent, realised marginals, rule violations, error counts. Informative only; their layout may grow without a format change. |

`spec` and `error_spec` are complete: `datagen regenerate <folder> --out <new>` rebuilds the
same bytes from them with the same datagen version.

### spec

```json
{
  "format_version": 2,
  "schema": {"tables": [
    {"name": "Trial", "key": "trial_id", "attributes": {"phase": ["I", "II", "III", "IV"], ...},
     "parent": null, "foreign_key": null},
    {"name": "Registration", "key": "reg_id", "attributes": {"sex": ["female", "male"], ...},
     "parent": "Trial", "foreign_key": "trial_id"}
  ]},
  "rules": [
    {"name": "P1", "atoms": {"design": ["single-arm"], "control": ["active", "placebo"]},
     "meaning": "A single-arm trial has no comparator arm."}
  ],
  "marginals": {...}, "dependencies": [...],
  "fanout": {"type": "fixed", "k": 5},
  "parents": 1000, "seed": 7, "conditioning": "sequential", "max_combinations": 1000000
}
```

- The schema has exactly two tables: one parent (`parent: null`) and one child whose
  `foreign_key` column holds the parent's `key`. Attribute names are unique over both tables.
- `attributes` lists each attribute's domain in order. Every value in the CSV files is in it.
- A **rule** is `NOT(a1 in S1 AND a2 in S2 AND ...)`: it *fails* on a row when every atom
  holds. A rule whose attributes are all in one table is checked per row of that table; a
  rule over both tables is checked on every joined (parent, child) row. The clean tables
  satisfy every rule.
- `marginals`, `dependencies` and `fanout` describe how the data was drawn. A reader that
  only needs the data and the rules can ignore them.

### error_spec

```json
{"format_version": 1, "rate": 0.05, "rate_by_table": {}, "rate_by_attribute": {},
 "eligible": "rule_attributes", "count": "exact", "row_weights": [],
 "mechanism": {"random": 1.0}, "confusion": {}, "detectable_share": null,
 "target_rules": null, "on_shortfall": "skip", "seed": null}
```

`rate` is the share of eligible cells that get an error. The other fields are explained in
`datagen/injector/inject.py`.

## CSV files

- UTF-8, comma separated, header row, `\n` line endings, no index column.
- Every value is a string. **Read with all columns as strings and without NA detection**
  (pandas: `pd.read_csv(path, dtype=str, keep_default_na=False)`): values such as `none` or
  `NA` are domain values, not missing values. There are no missing values.
- Keys are the strings `0`, `1`, ... in the parent table and in the child table.
- Clean and dirty files of one table have the same rows in the same order with the same keys
  and foreign keys. Errors change attribute values only, never keys.

### errors.csv

| column | meaning |
|---|---|
| `table` | table of the changed cell |
| `key` | key of the changed row in that table |
| `attribute` | changed column |
| `clean`, `dirty` | the value before and after |
| `mechanism` | how the dirty value was chosen: `random`, `neighbour` or `confusion` |
| `intended` | what the injector aimed for: `any`, `detectable` or `silent` |
| `detectable` | `True` when a rule involving the attribute fails on the dirty data at that cell (a parent cell: on the dirty parent row or one of its joined rows; a child cell: on the dirty child row or its joined row) |

## Checking a folder

A reader should check `format` and `format_version`, then compare the sha256 of every file
listed in `files` with the recorded value. `datagen.read(folder)` does all of this, plus the
key and foreign-key checks; a reader without datagen repeats the first two in a few lines.

## Sweep files

`datagen sweep SWEEP.json --out DIR` writes one folder per point (`DIR/<point name>/`).
Complete folders are skipped, so an interrupted sweep can be restarted; `--force` rewrites.

```json
{
  "format": "datagen-sweep", "format_version": 1,
  "spec": { "...": "a full spec as above" },
  "error_spec": { "rate": 0.05 },
  "points": [
    {"name": "p1000_k5_e5_r18_rep0", "rules": 18,
     "spec": {"parents": 1000, "fanout": {"type": "fixed", "k": 5}, "seed": 123},
     "error_spec": {"rate": 0.05}, "labels": {"replicate": 0}}
  ]
}
```

- `name`: folder name, letters, digits, `.`, `_` and `-`, unique in the file.
- `rules`: an integer keeps the first n rules of the base spec; a list of names keeps those
  rules (in base order); missing keeps all.
- `spec`, `error_spec`: top-level fields that replace those of the base specs.
  `"error_spec": null` in a point, or no base `error_spec`, gives clean data only.
- `labels`: copied into the folder's `dataset.json`.

Each folder's `dataset.json` holds the resolved specs, so the folders do not depend on the
sweep file afterwards.
