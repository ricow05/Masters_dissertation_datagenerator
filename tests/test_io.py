import json

import pytest

import datagen
from conftest import slow_violations
from datagen import (
    CPT, Depends, ErrorSpec, Frequencies, Normal, Poisson, RowWeight, Skewed, generate, inject, load_error_spec,
    load_spec, read, read_clean, recompute_stats, save_error_spec, save_spec, write,
)
from datagen.examples import trials
from datagen.generator.spec import DatasetSpec
from datagen.io import DATASET, FORMAT, FORMAT_VERSION, sha256
from datagen.provenance import generator_info

DATA_FILES = ["Trial_clean.csv", "Trial_dirty.csv", "Registration_clean.csv", "Registration_dirty.csv", "errors.csv"]
CLEAN_FILES = {"Trial_clean.csv", "Registration_clean.csv", DATASET}
DIRTY_FILES = CLEAN_FILES | {"Trial_dirty.csv", "Registration_dirty.csv", "errors.csv"}


def rich_spec(**kw):
    values = dict(
        count=30,
        parents=400,
        fanout=Poisson(2.5),
        marginals={"phase": Skewed(1.3), "blinding": Normal("double", 0.8), "control": Frequencies({"none": 1, "placebo": 2, "active": 2})},
        dependencies=[Depends("scope", "sponsor", 0.4), CPT("arm", ("design",), {("crossover",): {"both": 1}})],
    )
    values.update(kw)
    return trials.spec(**values)


def rich_errors(**kw):
    values = dict(
        rate=0.05,
        rate_by_attribute={"sex": 0.1},
        mechanism={"random": 0.6, "neighbour": 0.4},
        row_weights=[RowWeight("Trial", {"phase": "I"}, 3.0)],
        detectable_share=0.5,
    )
    values.update(kw)
    return ErrorSpec(**values)


def description(folder):
    return json.loads((folder / DATASET).read_text())


def test_spec_json_round_trip(tmp_path):
    spec = rich_spec()
    path = save_spec(spec, tmp_path / "spec.json")
    again = load_spec(path)
    assert again.to_dict() == spec.to_dict()
    saved = json.loads(path.read_text())
    assert saved["format_version"] == 2
    assert "errors" not in saved


def test_error_spec_json_round_trip(tmp_path):
    errors = rich_errors(seed=4, confusion={"phase": {"I": {"II": 1}}})
    path = save_error_spec(errors, tmp_path / "error_spec.json")
    assert load_error_spec(path).to_dict() == errors.to_dict()


def test_loaded_spec_generates_the_same_data(tmp_path):
    spec = rich_spec()
    again = load_spec(save_spec(spec, tmp_path / "spec.json"))
    a, b = generate(spec).frames(), generate(again).frames()
    for name in a:
        assert a[name].equals(b[name])


def test_clean_folder_round_trip(tmp_path):
    data = generate(rich_spec())
    assert "errors" not in data.stats and "dirty" not in data.stats["violations"]
    folder = write(data, tmp_path / "clean")
    assert {p.name for p in folder.iterdir()} == CLEAN_FILES
    loaded = read(folder)
    assert not loaded.has_errors and loaded.dirty is None and loaded.error_spec is None
    assert (loaded.clean.fk == data.clean.fk).all()
    assert recompute_stats(folder) == description(folder)["stats"]


def test_dataset_json_describes_everything(tmp_path):
    data = inject(generate(rich_spec()), rich_errors())
    folder = write(data, tmp_path / "ds", labels={"replicate": 2})
    d = description(folder)
    assert (d["format"], d["format_version"]) == (FORMAT, FORMAT_VERSION)
    assert d["generator"]["name"] == "datagen" and d["generator"]["version"] == datagen.__version__
    assert set(d["generator"]) == {"name", "version", "commit", "dirty"}
    assert d["name"] == "ds" and d["labels"] == {"replicate": 2}
    assert d["spec"] == data.spec.to_dict() and d["error_spec"] == data.error_spec.to_dict()
    assert d["files"]["Trial_dirty.csv"] == {"table": "Trial", "role": "dirty", "rows": 400,
                                             "sha256": sha256(folder / "Trial_dirty.csv")}
    assert d["files"]["errors.csv"]["role"] == "error_log"
    assert d["files"]["errors.csv"]["rows"] == len(data.errors)
    assert set(d["timings"]) >= {"clean_s", "errors_s"}
    assert write(generate(rich_spec()), tmp_path / "c", name="other").name == "c"
    assert description(tmp_path / "c")["name"] == "other" and description(tmp_path / "c")["error_spec"] is None


def test_write_read_round_trip(tmp_path):
    data = inject(generate(rich_spec()), rich_errors())
    folder = write(data, tmp_path / "ds")
    assert {p.name for p in folder.iterdir()} == DIRTY_FILES
    d = description(folder)
    assert set(d["files"]) == DIRTY_FILES - {DATASET}

    loaded = read(folder)
    assert loaded.name == "ds" and loaded.labels == {} and loaded.generator == d["generator"]
    assert loaded.error_spec.to_dict() == data.error_spec.to_dict()
    assert (loaded.clean.fk == data.clean.fk).all()
    for a in data.spec.schema.parent_attrs:
        assert (loaded.dirty.parents[a] == data.dirty.parents[a]).all()
    assert loaded.frames["Trial_clean"]["control"].isin(["none", "placebo", "active"]).all()   # "none" stays a string
    assert list(loaded.errors["detectable"]) == list(data.errors["detectable"])
    assert slow_violations(data.spec.schema, data.spec.rules, loaded.frames["Trial_clean"], loaded.frames["Registration_clean"]) == 0

    stats = recompute_stats(folder)
    assert stats["violations"] == d["stats"]["violations"]
    assert stats["errors"]["errors"] == d["stats"]["errors"]["errors"]
    assert stats["errors"]["detectable"] == d["stats"]["errors"]["detectable"]


def test_a_changed_file_is_refused(tmp_path):
    folder = write(inject(generate(rich_spec(parents=50)), rich_errors()), tmp_path / "ds")
    path = folder / "Trial_dirty.csv"
    read(folder)
    content = path.read_bytes()
    path.write_bytes(content[: content.rstrip(b"\n").rfind(b"\n") + 1])     # drop the last row
    with pytest.raises(ValueError, match="sha256"):
        read(folder)
    with pytest.raises(ValueError, match="foreign keys without a parent"):     # past the hash, the structure check
        read(folder, verify=False)


def test_other_folders_are_refused(tmp_path):
    folder = write(generate(rich_spec(parents=50)), tmp_path / "ds")
    d = description(folder)
    (folder / DATASET).write_text(json.dumps({**d, "format_version": 2}))
    with pytest.raises(ValueError, match="format_version 2"):
        read(folder)
    (folder / DATASET).unlink()
    with pytest.raises(ValueError, match="no dataset.json"):
        read(folder)


def test_writing_clean_data_removes_old_dirty_files(tmp_path):
    clean = generate(rich_spec())
    write(inject(clean, rich_errors()), tmp_path / "ds")
    write(clean, tmp_path / "ds")
    assert {p.name for p in (tmp_path / "ds").iterdir()} == CLEAN_FILES


def test_inject_into_a_written_clean_folder_equals_one_step(tmp_path):
    clean = write(generate(rich_spec()), tmp_path / "clean")
    two_steps = write(inject(read_clean(clean), rich_errors()), tmp_path / "two")
    one_step = write(inject(generate(rich_spec()), rich_errors()), tmp_path / "one")
    for name in DATA_FILES:
        assert sha256(two_steps / name) == sha256(one_step / name), name


def test_dataset_json_regenerates_the_folder(tmp_path):
    original = write(inject(generate(rich_spec()), rich_errors(seed=3)), tmp_path / "a")
    spec, errors = load_spec(original / DATASET), load_error_spec(original / DATASET)
    again = write(inject(generate(spec), errors), tmp_path / "b")
    for name in DATA_FILES:
        assert sha256(again / name) == sha256(original / name), name
    with pytest.raises(ValueError, match="clean data only"):
        load_error_spec(write(generate(spec), tmp_path / "c") / DATASET)


def test_same_seed_same_files_and_error_change_keeps_clean_files(tmp_path):
    a = write(inject(generate(rich_spec()), rich_errors()), tmp_path / "a")
    b = write(inject(generate(rich_spec()), rich_errors()), tmp_path / "b")
    c = write(inject(generate(rich_spec()), ErrorSpec(rate=0.2)), tmp_path / "c")
    for name in DATA_FILES:
        assert sha256(a / name) == sha256(b / name), name
    assert description(a)["spec"] == description(b)["spec"] == description(c)["spec"]
    for name in ("Trial_clean.csv", "Registration_clean.csv"):
        assert sha256(a / name) == sha256(c / name)
    assert sha256(a / "Trial_dirty.csv") != sha256(c / "Trial_dirty.csv")


def test_error_seed():
    clean = generate(rich_spec(seed=1))
    default = inject(clean, ErrorSpec(rate=0.1)).errors
    assert inject(clean, ErrorSpec(rate=0.1)).errors.equals(default)
    seeded = inject(clean, ErrorSpec(rate=0.1, seed=5)).errors
    assert inject(clean, ErrorSpec(rate=0.1, seed=5)).errors.equals(seeded)
    assert not seeded.equals(default)
    assert not inject(clean, ErrorSpec(rate=0.1, seed=6)).errors.equals(seeded)


def test_generator_info_never_takes_an_enclosing_repository(tmp_path, monkeypatch):
    import datagen.provenance as provenance

    info = generator_info()
    assert info["version"] == datagen.__version__
    assert info["commit"] is None or len(info["commit"]) == 40
    nested = tmp_path / "not_a_checkout"
    nested.mkdir()
    monkeypatch.setattr(provenance, "ROOT", nested)
    provenance.generator_info.cache_clear()
    try:
        assert provenance.generator_info()["commit"] is None
    finally:
        provenance.generator_info.cache_clear()


def test_seed_override():
    spec = trials.spec(parents=200, seed=1)
    assert generate(spec, seed=2).spec.seed == 2
    assert not generate(spec, seed=2).frames()["Trial_clean"].equals(generate(spec).frames()["Trial_clean"])


def test_schema_validation():
    from datagen import Table, Schema
    with pytest.raises(ValueError, match="more than one table"):
        Schema([Table("P", "pid", {"x": ("a",)}), Table("C", "cid", {"x": ("a",)}, parent="P")])
    with pytest.raises(NotImplementedError):
        Schema([Table("P", "pid", {"x": ("a",)})])
    with pytest.raises(ValueError, match="key column"):
        Schema([Table("P", "pid", {"x": ("a",)}), Table("C", "cid", {"pid": ("a",)}, parent="P")])
    with pytest.raises(ValueError, match="not in the domain"):
        generate(DatasetSpec(trials.schema(), [trials.rules(1)[0].__class__.of({"phase": "V"}, "R")], parents=5))
