"""Write and read a dataset folder (format: ``FORMAT.md``).

A folder holds ``<Parent>_clean.csv`` and ``<Child>_clean.csv``, after injection also
``<Parent>_dirty.csv``, ``<Child>_dirty.csv`` and ``errors.csv`` (the error log), and one
``dataset.json`` that describes everything: the generator version and commit, the full
dataset spec and error spec, every file with its role, row count and sha256, the timings
and the statistics. ``dataset.json`` is written last, so a folder without it is incomplete.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import Coded
from .generator.generate import CleanData, clean_stats
from .generator.spec import DatasetSpec
from .injector import stats as error_stats
from .injector.inject import eligible_attributes
from .injector.spec import ErrorSpec
from .pipeline import DirtyData, merge_stats
from .provenance import generator_info

FORMAT = "datagen-dataset"
FORMAT_VERSION = 3
DATASET = "dataset.json"
ERRORS = "errors.csv"


def _save_json(data, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_dataset(data) -> bool:
    return isinstance(data, dict) and data.get("format") == FORMAT


def save_spec(spec: DatasetSpec, path) -> Path:
    return _save_json(spec.to_dict(), path)


def load_spec(path) -> DatasetSpec:
    """A ``spec.json``, or the spec inside a folder's ``dataset.json``."""
    data = _load_json(path)
    return DatasetSpec.from_dict(data["spec"] if _is_dataset(data) else data)


def save_error_spec(error_spec: ErrorSpec, path) -> Path:
    return _save_json(error_spec.to_dict(), path)


def load_error_spec(path) -> ErrorSpec:
    """An ``error_spec.json``, or the error spec inside a folder's ``dataset.json``."""
    data = _load_json(path)
    if _is_dataset(data):
        if data["error_spec"] is None:
            raise ValueError(f"{path} describes clean data only (no error spec)")
        data = data["error_spec"]
    return ErrorSpec.from_dict(data)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _csv(df, path):
    df.to_csv(path, index=False, lineterminator="\n")


def write(data: CleanData | DirtyData, directory, name: str | None = None, labels: dict | None = None) -> Path:
    """Write clean data (``generate``) or clean and dirty data (``inject``) to ``directory``.

    The folder is created if needed and returned. ``name`` (default: the folder name) and
    ``labels`` (free key-value pairs, e.g. a replicate number) are stored in ``dataset.json``.
    Writing clean data removes dirty files left by an earlier ``inject``, so a folder never
    mixes two datasets.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / DATASET).unlink(missing_ok=True)              # incomplete until written again
    schema = data.spec.schema
    files = {}
    for frame_name, frame in data.frames().items():
        table, role = frame_name.rsplit("_", 1)
        _csv(frame, out / f"{frame_name}.csv")
        files[f"{frame_name}.csv"] = {"table": table, "role": role, "rows": len(frame)}
    if isinstance(data, DirtyData):
        _csv(data.errors, out / ERRORS)
        files[ERRORS] = {"role": "error_log", "rows": len(data.errors)}
    else:
        for stale in (f"{schema.parent.name}_dirty.csv", f"{schema.child.name}_dirty.csv", ERRORS):
            (out / stale).unlink(missing_ok=True)
    for file_name, entry in files.items():
        entry["sha256"] = sha256(out / file_name)
    error_spec = data.error_spec if isinstance(data, DirtyData) else None
    description = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "generator": generator_info(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": name if name is not None else out.name,
        "labels": dict(labels or {}),
        "spec": data.spec.to_dict(),
        "error_spec": error_spec.to_dict() if error_spec is not None else None,
        "files": files,
        "timings": data.timings,
        "stats": data.stats,
    }
    _save_json(description, out / DATASET)
    return out


@dataclass
class Loaded:
    """A dataset folder. ``error_spec``, ``dirty`` and ``errors`` are None for clean data."""

    description: dict                  # dataset.json
    spec: DatasetSpec
    clean: Coded
    frames: dict = field(default_factory=dict)
    error_spec: ErrorSpec | None = None
    dirty: Coded | None = None
    errors: pd.DataFrame | None = None

    @property
    def has_errors(self):
        return self.dirty is not None

    @property
    def name(self) -> str:
        return self.description["name"]

    @property
    def labels(self) -> dict:
        return self.description["labels"]

    @property
    def generator(self) -> dict:
        return self.description["generator"]


def _read_csv(path):
    # keep_default_na=False: a value such as "none" or "NA" must stay a string.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_description(directory) -> dict:
    """``dataset.json`` of a folder, after checking its format."""
    path = Path(directory) / DATASET
    if not path.exists():
        raise ValueError(f"{directory}: no {DATASET} (not a datagen folder, or an incomplete one)")
    description = _load_json(path)
    if not _is_dataset(description):
        raise ValueError(f"{path}: not a {FORMAT} file")
    if description.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: format_version {description.get('format_version')}, this datagen reads {FORMAT_VERSION}")
    return description


def read(directory, verify: bool = True) -> Loaded:
    """Load a folder. With ``verify`` every file must match its sha256 in ``dataset.json``."""
    directory = Path(directory)
    description = read_description(directory)
    if verify:
        for file_name, entry in description["files"].items():
            if sha256(directory / file_name) != entry["sha256"]:
                raise ValueError(f"{directory / file_name}: sha256 differs from {DATASET}")
    spec = DatasetSpec.from_dict(description["spec"])
    by_role = {(e.get("table"), e["role"]): n for n, e in description["files"].items()}
    parent, child = spec.schema.parent.name, spec.schema.child.name

    def frame(table, role):
        return _read_csv(directory / by_role[(table, role)])

    frames = {f"{parent}_clean": frame(parent, "clean"), f"{child}_clean": frame(child, "clean")}
    clean = Coded.from_frames(spec.schema, frames[f"{parent}_clean"], frames[f"{child}_clean"])
    loaded = Loaded(description, spec, clean, frames)
    if description["error_spec"] is None:
        return loaded

    frames[f"{parent}_dirty"], frames[f"{child}_dirty"] = frame(parent, "dirty"), frame(child, "dirty")
    dirty = Coded.from_frames(spec.schema, frames[f"{parent}_dirty"], frames[f"{child}_dirty"])
    if (clean.parent_keys != dirty.parent_keys).any() or (clean.child_keys != dirty.child_keys).any():
        raise ValueError("keys differ between the clean and dirty files")
    if (clean.fk != dirty.fk).any():
        raise ValueError("foreign keys differ between the clean and dirty files")
    errors = _read_csv(directory / by_role[(None, "error_log")])
    if "detectable" in errors:
        errors["detectable"] = errors["detectable"].map({"True": True, "False": False})
    loaded.error_spec = ErrorSpec.from_dict(description["error_spec"])
    loaded.dirty, loaded.errors = dirty, errors
    return loaded


def clean_of(loaded: Loaded) -> CleanData:
    """The clean data of a loaded folder, ready for ``inject``."""
    return CleanData(loaded.spec, loaded.clean, clean_stats(loaded.spec, loaded.clean))


def read_clean(directory) -> CleanData:
    """The clean data of a folder, ready for ``inject``. Dirty files are ignored."""
    return clean_of(read(directory))


def recompute_stats(directory) -> dict:
    """Statistics recomputed from the files of a dataset folder."""
    loaded = read(directory)
    spec = loaded.spec
    stats = clean_stats(spec, loaded.clean)
    if not loaded.has_errors:
        return stats
    eligible = eligible_attributes(spec.schema, spec.rules, loaded.error_spec)
    injected = error_stats.compute(spec.schema, spec.rules, loaded.clean, loaded.dirty, loaded.errors, eligible)
    return merge_stats(stats, injected)
