"""Many dataset folders from one sweep file: ``python -m datagen sweep sweep.json --out DIR``.

A sweep file holds a base dataset spec, an optional base error spec, and a list of points.
Each point names its folder and overrides fields of the two specs (format: ``FORMAT.md``)::

    {
      "format": "datagen-sweep", "format_version": 1,
      "spec": { ...a full dataset spec... },
      "error_spec": { "rate": 0.05 },
      "points": [
        {"name": "p1000_r18_e5", "rules": 18, "spec": {"parents": 1000, "seed": 7},
         "error_spec": {"rate": 0.05}, "labels": {"replicate": 0}}
      ]
    }

``rules`` keeps the first n rules of the base spec (an integer) or the named ones (a list).
``"error_spec": null`` in a point (or no base error spec) gives clean data only.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .generator.generate import generate
from .generator.spec import DatasetSpec
from .injector.spec import ErrorSpec
from .io import DATASET, write
from .pipeline import inject

FORMAT = "datagen-sweep"
FORMAT_VERSION = 1
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass
class Point:
    name: str
    spec: DatasetSpec
    error_spec: ErrorSpec | None
    labels: dict = field(default_factory=dict)


def _select_rules(rules: list, selection, name):
    if isinstance(selection, int):
        if not 0 <= selection <= len(rules):
            raise ValueError(f"point {name}: {selection} rules, the base spec has {len(rules)}")
        return rules[:selection]
    wanted = set(selection)
    unknown = wanted - {r.get("name") for r in rules}
    if unknown:
        raise ValueError(f"point {name}: unknown rules {sorted(unknown)}")
    return [r for r in rules if r.get("name") in wanted]


def parse_sweep(data: dict) -> list[Point]:
    if data.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} file")
    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"sweep format_version {data.get('format_version')}, this datagen reads {FORMAT_VERSION}")
    base_spec, base_errors = data["spec"], data.get("error_spec")
    points, names = [], set()
    for raw in data["points"]:
        name = raw["name"]
        if not _NAME.match(name):
            raise ValueError(f"point name {name!r}: use letters, digits, '.', '_' and '-' only")
        if name in names:
            raise ValueError(f"point name {name!r} appears twice")
        names.add(name)
        spec = {**base_spec, **raw.get("spec", {})}
        if "rules" in raw:
            spec["rules"] = _select_rules(base_spec.get("rules", []), raw["rules"], name)
        if "error_spec" in raw:
            errors = None if raw["error_spec"] is None else {**(base_errors or {}), **raw["error_spec"]}
        else:
            errors = base_errors
        points.append(Point(name, DatasetSpec.from_dict(spec),
                            ErrorSpec.from_dict(errors) if errors is not None else None,
                            dict(raw.get("labels", {}))))
    return points


def load_sweep(path) -> list[Point]:
    return parse_sweep(json.loads(Path(path).read_text(encoding="utf-8")))


def run_sweep(path, out, force: bool = False, progress=print) -> list[Path]:
    """Write one folder per point under ``out``. Complete folders (with ``dataset.json``) are
    kept unless ``force``, so an interrupted sweep can simply be restarted."""
    points = load_sweep(path)
    out = Path(out)
    folders = []
    for done, point in enumerate(points, 1):
        folder = out / point.name
        folders.append(folder)
        if (folder / DATASET).exists() and not force:
            continue
        start = time.perf_counter()
        data = generate(point.spec)
        if point.error_spec is not None:
            data = inject(data, point.error_spec)
        write(data, folder, name=point.name, labels=point.labels)
        progress(f"[{done}/{len(points)}] {point.name}: {time.perf_counter() - start:.1f} s")
    return folders
