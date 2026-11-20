import json

from datagen.__main__ import main
from datagen.io import sha256

DATA_FILES = ("Trial_clean.csv", "Registration_clean.csv", "Trial_dirty.csv", "Registration_dirty.csv", "errors.csv")


def description(folder):
    return json.loads((folder / "dataset.json").read_text())


def test_example_generate_inject_stats_fit(tmp_path, capsys):
    spec = tmp_path / "spec.json"
    assert main(["example", "trials", "--rules", "18", "--parents", "1000", "--children", "5", "--out", str(spec)]) == 0
    errors = tmp_path / "error_spec.json"
    assert main(["error-spec", "--rate", "0.05", "--out", str(errors)]) == 0

    clean = tmp_path / "clean"
    assert main(["generate", str(spec), "--out", str(clean), "--seed", "3", "--label", "replicate=1",
                 "--label", "note=\"first\""]) == 0
    d = description(clean)
    assert d["spec"]["seed"] == 3 and d["error_spec"] is None
    assert d["labels"] == {"replicate": 1, "note": "first"}
    assert "errors" not in d["stats"]
    assert d["stats"]["violations"]["clean"] == 0
    assert not (clean / "Trial_dirty.csv").exists()

    out = tmp_path / "ds"
    assert main(["inject", str(clean), "--errors", str(errors), "--out", str(out), "--name", "my_set"]) == 0
    d = description(out)
    assert d["name"] == "my_set" and d["labels"] == {"replicate": 1, "note": "first"}
    assert d["error_spec"]["rate"] == 0.05
    assert d["stats"]["errors"]["rate"] == 0.05
    assert d["stats"]["violations"]["clean"] == 0

    one_step = tmp_path / "one_step"
    assert main(["generate", str(spec), "--out", str(one_step), "--seed", "3", "--errors", str(errors)]) == 0
    for name in ("Trial_dirty.csv", "Registration_dirty.csv", "errors.csv"):
        assert sha256(one_step / name) == sha256(out / name)

    again = tmp_path / "regenerated"
    assert main(["regenerate", str(out), "--out", str(again)]) == 0
    for name in DATA_FILES:
        assert sha256(again / name) == sha256(out / name), name
    assert description(again)["labels"] == description(out)["labels"]

    reseeded = tmp_path / "reseeded"
    assert main(["inject", str(clean), "--errors", str(errors), "--out", str(reseeded), "--seed", "9"]) == 0
    assert sha256(reseeded / "errors.csv") != sha256(out / "errors.csv")

    capsys.readouterr()
    assert main(["stats", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "violations           clean 0, dirty" in printed
    assert "made by datagen" in printed
    assert main(["stats", str(clean)]) == 0
    assert "no errors injected" in capsys.readouterr().out

    fitted = tmp_path / "fitted.json"
    assert main(["fit", str(out / "Trial_clean.csv"), str(out / "Registration_clean.csv"), "--fk", "trial_id",
                 "--rules", str(spec), "--parent-name", "Trial", "--child-name", "Registration",
                 "--out", str(fitted)]) == 0
    report = json.loads((tmp_path / "fitted_report.json").read_text())
    assert report["rule_violations"]["total"] == 0
    assert main(["generate", str(fitted), "--out", str(tmp_path / "again"), "--parents", "200"]) == 0
    again = description(tmp_path / "again")
    assert again["stats"]["rows"]["Trial"] == 200
    assert again["stats"]["violations"]["clean"] == 0


def test_sweep_command(tmp_path, capsys):
    spec = tmp_path / "spec.json"
    assert main(["example", "trials", "--rules", "6", "--parents", "60", "--children", "3", "--out", str(spec)]) == 0
    sweep = {
        "format": "datagen-sweep", "format_version": 1,
        "spec": json.loads(spec.read_text()),
        "error_spec": {"rate": 0.05},
        "points": [
            {"name": "a", "rules": 3, "spec": {"seed": 1}, "labels": {"replicate": 0}},
            {"name": "b", "spec": {"seed": 2}, "error_spec": None},
        ],
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(sweep))
    out = tmp_path / "out"
    assert main(["sweep", str(path), "--out", str(out)]) == 0
    assert len(description(out / "a")["spec"]["rules"]) == 3
    assert description(out / "a")["labels"] == {"replicate": 0}
    assert description(out / "b")["error_spec"] is None
    capsys.readouterr()
    assert main(["sweep", str(path), "--out", str(out)]) == 0          # nothing left to do
    assert "[1/2]" not in capsys.readouterr().out
