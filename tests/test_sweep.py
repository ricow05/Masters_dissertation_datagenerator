import json

import pytest

from datagen import ErrorSpec, generate, inject, write
from datagen.examples import trials
from datagen.io import sha256
from datagen.sweep import load_sweep, parse_sweep, run_sweep

DATA_FILES = ("Trial_clean.csv", "Registration_clean.csv", "Trial_dirty.csv", "Registration_dirty.csv", "errors.csv")


def base(**kw):
    sweep = {
        "format": "datagen-sweep", "format_version": 1,
        "spec": trials.spec(count=8, parents=80).to_dict(),
        "error_spec": {"rate": 0.05},
        "points": [],
    }
    sweep.update(kw)
    return sweep


def test_rules_by_prefix_and_by_name():
    names = [r["name"] for r in base()["spec"]["rules"]]
    points = parse_sweep(base(points=[
        {"name": "first3", "rules": 3},
        {"name": "named", "rules": [names[5], names[1]]},
        {"name": "all"},
    ]))
    assert [r.name for r in points[0].spec.rules] == names[:3]
    assert [r.name for r in points[1].spec.rules] == [names[1], names[5]]        # base order is kept
    assert len(points[2].spec.rules) == 8


def test_overrides_and_error_specs():
    points = parse_sweep(base(points=[
        {"name": "a", "spec": {"parents": 30, "seed": 4}},
        {"name": "b", "error_spec": {"rate": 0.2, "seed": 1}},
        {"name": "c", "error_spec": None, "labels": {"replicate": 3}},
    ]))
    a, b, c = points
    assert (a.spec.parents, a.spec.seed) == (30, 4)
    assert a.error_spec.rate == 0.05 and b.error_spec.rate == 0.2 and b.error_spec.seed == 1
    assert c.error_spec is None and c.labels == {"replicate": 3}
    assert parse_sweep(base(error_spec=None, points=[{"name": "x"}]))[0].error_spec is None


@pytest.mark.parametrize("points, message", [
    ([{"name": "a"}, {"name": "a"}], "twice"),
    ([{"name": "../up"}], "letters"),
    ([{"name": "a", "rules": 99}], "99 rules"),
    ([{"name": "a", "rules": ["nope"]}], "unknown rules"),
])
def test_bad_points_are_refused(points, message):
    with pytest.raises(ValueError, match=message):
        parse_sweep(base(points=points))


def test_other_files_are_refused():
    with pytest.raises(ValueError, match="not a datagen-sweep"):
        parse_sweep({**base(), "format": "x"})
    with pytest.raises(ValueError, match="format_version 2"):
        parse_sweep({**base(), "format_version": 2})


def test_sweep_equals_generate_and_inject(tmp_path):
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(base(points=[{"name": "p", "rules": 4, "spec": {"seed": 11},
                                              "error_spec": {"rate": 0.1}}])))
    assert load_sweep(path)[0].name == "p"
    [folder] = run_sweep(path, tmp_path / "out", progress=lambda _: None)
    direct = write(inject(generate(trials.spec(count=4, parents=80, seed=11)), ErrorSpec(rate=0.1)), tmp_path / "direct")
    for name in DATA_FILES:
        assert sha256(folder / name) == sha256(direct / name), name
    assert json.loads((folder / "dataset.json").read_text())["name"] == "p"


def test_complete_folders_are_kept_unless_forced(tmp_path):
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(base(points=[{"name": "a"}, {"name": "b"}])))
    out = tmp_path / "out"
    said = []
    run_sweep(path, out, progress=said.append)
    assert len(said) == 2
    (out / "b" / "dataset.json").unlink()                       # an interrupted folder
    said.clear()
    run_sweep(path, out, progress=said.append)
    assert len(said) == 1 and said[0].startswith("[2/2] b")
    said.clear()
    run_sweep(path, out, force=True, progress=said.append)
    assert len(said) == 2
