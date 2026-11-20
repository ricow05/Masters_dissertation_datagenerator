"""Command line: ``python -m datagen {generate,inject,sweep,regenerate,fit,example,error-spec,stats} ...``."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .generator import generate
from .injector import ErrorSpec
from .io import (clean_of, load_error_spec, load_spec, read, read_description, recompute_stats, save_error_spec,
                 save_spec, write)
from .pipeline import inject
from .rules import Rule
from .stats import summary
from .sweep import run_sweep


def _labels(pairs):
    labels = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--label {pair!r}: expected key=value")
        try:
            labels[key] = json.loads(value)          # numbers and booleans keep their type
        except json.JSONDecodeError:
            labels[key] = value
    return labels


def _wrote(folder, data):
    print(f"wrote {folder}")
    print(summary(data.stats))
    return 0


def _generate(args):
    spec = load_spec(args.spec)
    if args.parents is not None:
        spec = replace(spec, parents=args.parents)
    data = generate(spec, seed=args.seed)
    if args.errors:
        data = inject(data, load_error_spec(args.errors))
    return _wrote(write(data, args.out, name=args.name, labels=_labels(args.label)), data)


def _inject(args):
    error_spec = load_error_spec(args.errors)
    if args.seed is not None:
        error_spec = replace(error_spec, seed=args.seed)
    loaded = read(args.directory)
    data = inject(clean_of(loaded), error_spec)
    return _wrote(write(data, args.out, name=args.name, labels={**loaded.labels, **_labels(args.label)}), data)


def _regenerate(args):
    source = Path(args.source)
    description = read_description(source) if source.is_dir() else json.loads(source.read_text(encoding="utf-8"))
    path = source / "dataset.json" if source.is_dir() else source
    data = generate(load_spec(path))
    if description.get("error_spec") is not None:
        data = inject(data, load_error_spec(path))
    return _wrote(write(data, args.out, name=description.get("name"), labels=description.get("labels")), data)


def _sweep(args):
    folders = run_sweep(args.sweep, args.out, force=args.force)
    print(f"{len(folders)} datasets in {args.out}")
    return 0


def _load_rules(path):
    if path is None:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("rules", [])
    return [Rule.from_dict(r) for r in data]


def _fit(args):
    from .generator.fit import fit

    def read(path):
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=args.na_values or [])

    result = fit(
        read(args.parent),
        read(args.child),
        args.fk,
        rules=_load_rules(args.rules),
        structure=args.structure,
        alpha=args.alpha,
        significance=args.significance,
        parent_key=args.parent_key,
        child_key=args.child_key,
        parent_name=args.parent_name,
        child_name=args.child_name,
        max_domain=args.max_domain,
        rows=args.parents,
    )
    save_spec(result.spec, args.out)
    report = args.report or Path(args.out).with_name(Path(args.out).stem + "_report.json")
    Path(report).write_text(json.dumps(result.report, indent=2, default=str) + "\n", encoding="utf-8")
    kept = sum(e["kept"] for e in result.report["edges"])
    print(f"wrote {args.out} and {report}")
    print(f"rows {result.report['rows']}, edges kept {kept} of {len(result.report['edges'])}")
    if result.report["dropped_columns"]:
        print(f"dropped columns: {result.report['dropped_columns']}")
    if "rule_violations" in result.report:
        print(f"rule violations in the input: {result.report['rule_violations']['total']}")
    return 0


def _example(args):
    from .examples import trials

    spec = trials.spec(count=args.rules, parents=args.parents, children=args.children, seed=args.seed)
    save_spec(spec, args.out)
    print(f"wrote {args.out} ({args.rules} rules, {args.parents} parents x {args.children} children)")
    return 0


def _error_spec(args):
    save_error_spec(ErrorSpec(rate=args.rate, eligible=args.eligible, seed=args.seed), args.out)
    print(f"wrote {args.out} (rate {args.rate}, eligible {args.eligible}); edit it for the other settings")
    return 0


def _stats(args):
    stats = recompute_stats(args.directory)
    if args.json:
        print(json.dumps(stats, indent=2))
        return 0
    g = read_description(args.directory)["generator"]
    commit = f"commit {g['commit'][:10]}{' + uncommitted changes' if g['dirty'] else ''}" if g["commit"] else "commit unknown"
    print(f"made by {g['name']} {g['version']} ({commit})")
    print(summary(stats))
    return 0


def parser():
    p = argparse.ArgumentParser(prog="datagen", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a folder of clean data from a spec.json")
    g.add_argument("spec", help="spec.json (or a folder's dataset.json)")
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int)
    g.add_argument("--parents", type=int, help="override the number of parent rows")
    g.add_argument("--errors", help="error_spec.json: inject these errors right away (same as running inject)")
    g.add_argument("--name", help="dataset name in dataset.json (default: the folder name)")
    g.add_argument("--label", action="append", metavar="KEY=VALUE", help="a label for dataset.json (repeatable)")
    g.set_defaults(run=_generate)

    i = sub.add_parser("inject", help="add errors to a folder of clean data")
    i.add_argument("directory", help="a folder written by generate (dirty files in it are ignored)")
    i.add_argument("--errors", required=True, help="error_spec.json (or a folder's dataset.json)")
    i.add_argument("--out", required=True, help="the folder for clean + dirty data (may be the input folder)")
    i.add_argument("--seed", type=int, help="seed of the errors (default: error_spec.json, else the clean data's seed)")
    i.add_argument("--name", help="dataset name in dataset.json (default: the folder name)")
    i.add_argument("--label", action="append", metavar="KEY=VALUE", help="a label added to those of the input folder")
    i.set_defaults(run=_inject)

    w = sub.add_parser("sweep", help="generate one folder per point of a sweep file")
    w.add_argument("sweep", help="sweep.json (see FORMAT.md)")
    w.add_argument("--out", required=True, help="parent folder; each point gets <out>/<name>")
    w.add_argument("--force", action="store_true", help="regenerate folders that are already complete")
    w.set_defaults(run=_sweep)

    x = sub.add_parser("regenerate", help="generate a dataset again from its dataset.json")
    x.add_argument("source", help="a dataset folder or its dataset.json")
    x.add_argument("--out", required=True)
    x.set_defaults(run=_regenerate)

    f = sub.add_parser("fit", help="learn a spec.json from a parent and a child CSV")
    f.add_argument("parent")
    f.add_argument("child")
    f.add_argument("--fk", required=True, help="foreign-key column of the child table")
    f.add_argument("--out", required=True)
    f.add_argument("--report", help="where to write the fit report (default: <out>_report.json)")
    f.add_argument("--rules", help="JSON file with a list of rules, or a spec.json")
    f.add_argument("--structure", default="chow-liu", choices=["chow-liu", "none"])
    f.add_argument("--alpha", type=float, default=1.0)
    f.add_argument("--significance", type=float, default=1e-3)
    f.add_argument("--parent-key")
    f.add_argument("--child-key")
    f.add_argument("--parent-name", default="Parent")
    f.add_argument("--child-name", default="Child")
    f.add_argument("--max-domain", type=int, default=50)
    f.add_argument("--parents", type=int, help="parent rows to generate (default: as many as in the input)")
    f.add_argument("--na-values", nargs="*", help="strings to treat as missing (default: none)")
    f.set_defaults(run=_fit)

    e = sub.add_parser("example", help="write the clinical-trials example spec")
    e.add_argument("name", choices=["trials"])
    e.add_argument("--rules", type=int, default=18)
    e.add_argument("--parents", type=int, default=1000)
    e.add_argument("--children", type=int, default=5)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", required=True)
    e.set_defaults(run=_example)

    r = sub.add_parser("error-spec", help="write an error_spec.json with random errors at one rate")
    r.add_argument("--rate", type=float, default=0.05)
    r.add_argument("--eligible", default="rule_attributes", choices=["rule_attributes", "all"])
    r.add_argument("--seed", type=int, help="seed of the errors (default: the clean data's seed)")
    r.add_argument("--out", required=True)
    r.set_defaults(run=_error_spec)

    s = sub.add_parser("stats", help="recompute and print the statistics of a dataset folder (clean or dirty)")
    s.add_argument("directory")
    s.add_argument("--json", action="store_true")
    s.set_defaults(run=_stats)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
