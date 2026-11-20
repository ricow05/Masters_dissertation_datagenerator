import math

import numpy as np
import pytest

from conftest import slow_detectable
from datagen import ErrorSpec, Poisson, RowWeight, generate, inject
from datagen.examples import trials

SCHEMA = trials.schema()


def run(errors, count=18, parents=1000, **kw):
    return inject(generate(trials.spec(count=count, parents=parents, **kw)), errors)


def test_exact_count_per_scope_and_no_noop_errors():
    data = run(ErrorSpec(rate=0.05))
    log = data.errors
    assert (log["clean"] != log["dirty"]).all()
    per_scope = log.groupby("attribute").size()
    for a in data.stats["errors"]["eligible_attributes"]:
        n = data.clean.n_parents if SCHEMA.is_parent_attr(a) else data.clean.n_children
        assert per_scope[a] == math.floor(0.05 * n + 0.5)
    assert data.stats["errors"]["rate"] == pytest.approx(0.05)


def test_keys_and_foreign_keys_are_untouched():
    frames = run(ErrorSpec(rate=0.2)).frames()
    assert frames["Trial_clean"]["trial_id"].equals(frames["Trial_dirty"]["trial_id"])
    for column in ("reg_id", "trial_id"):
        assert frames["Registration_clean"][column].equals(frames["Registration_dirty"][column])


def test_log_matches_the_data():
    data = run(ErrorSpec(rate=0.1), fanout=Poisson(3))
    frames = data.frames()
    differences = 0
    for table, key in (("Trial", "trial_id"), ("Registration", "reg_id")):
        clean, dirty = frames[f"{table}_clean"].set_index(key), frames[f"{table}_dirty"].set_index(key)
        for _, e in data.errors[data.errors["table"] == table].iterrows():
            assert clean.at[e["key"], e["attribute"]] == e["clean"]
            assert dirty.at[e["key"], e["attribute"]] == e["dirty"]
        differences += int((clean != dirty).to_numpy().sum())
    assert differences == len(data.errors)


def test_rate_by_table_and_attribute():
    data = run(ErrorSpec(rate=0.05, rate_by_table={"Registration": 0.0}, rate_by_attribute={"phase": 0.2, "sex": 0.1}))
    by_attribute = data.stats["errors"]["by_attribute"]
    assert by_attribute["phase"]["rate"] == pytest.approx(0.2)
    assert by_attribute["design"]["rate"] == pytest.approx(0.05)
    assert by_attribute["sex"]["rate"] == pytest.approx(0.1)
    assert by_attribute["age_group"]["errors"] == 0


def test_child_rate_zero_gives_no_child_errors():
    data = run(ErrorSpec(rate=0.05, rate_by_table={"Registration": 0.0}))
    assert (data.errors["table"] == "Trial").all()


def test_eligible_all_includes_attributes_outside_rules():
    data = run(ErrorSpec(rate=0.05, eligible="all"), count=6)
    assert "sponsor" in set(data.errors["attribute"])
    assert "sponsor" not in set(run(ErrorSpec(rate=0.05), count=6).errors["attribute"])


def test_bernoulli_rate():
    data = run(ErrorSpec(rate=0.05, count="bernoulli"), parents=4000)
    assert data.stats["errors"]["rate"] == pytest.approx(0.05, abs=0.004)


def test_changing_errors_keeps_clean_data():
    a = run(ErrorSpec(rate=0.05), parents=500).frames()
    b = run(ErrorSpec(rate=0.2, mechanism={"neighbour": 1}), parents=500).frames()
    assert a["Trial_clean"].equals(b["Trial_clean"])
    assert a["Registration_clean"].equals(b["Registration_clean"])
    assert not a["Trial_dirty"].equals(b["Trial_dirty"])


# -- mechanisms ------------------------------------------------------------------------
def test_neighbour_moves_one_step_in_the_domain_order():
    data = run(ErrorSpec(rate=0.1, eligible="all", mechanism={"neighbour": 1.0}))
    for _, e in data.errors.iterrows():
        domain = SCHEMA.domains[e["attribute"]]
        assert abs(domain.index(e["clean"]) - domain.index(e["dirty"])) == 1
    assert (data.errors["mechanism"] == "neighbour").all()


def test_confusion_uses_only_the_given_pairs():
    confusion = {"phase": {"I": {"II": 1}, "II": {"I": 1, "III": 1}}, "sex": {"female": {"male": 1}}}
    data = run(ErrorSpec(rate=0.1, eligible=["phase", "sex"], mechanism={"confusion": 1.0}, confusion=confusion))
    log = data.errors
    assert set(zip(log["clean"], log["dirty"])) <= {("I", "II"), ("II", "I"), ("II", "III"), ("female", "male")}
    assert (log["mechanism"] == "confusion").all()
    assert set(log["clean"][log["attribute"] == "phase"]) <= {"I", "II"}


def test_mixture_of_mechanisms():
    data = run(ErrorSpec(rate=0.1, mechanism={"random": 0.5, "neighbour": 0.5}), parents=3000)
    shares = data.errors["mechanism"].value_counts(normalize=True)
    assert shares["random"] == pytest.approx(0.5, abs=0.03)


# -- location ----------------------------------------------------------------------------
def test_row_weight_zero_excludes_rows():
    rw = [RowWeight("Trial", {"phase": ["II", "III", "IV"]}, 0.0)]
    data = run(ErrorSpec(rate=0.05, rate_by_table={"Registration": 0}, row_weights=rw))
    frames = data.frames()
    phase = frames["Trial_clean"].set_index("trial_id")["phase"]
    assert len(data.errors) > 0
    assert (phase[data.errors["key"]] == "I").all()


def test_row_weights_on_child_rows_by_parent_attribute():
    rw = [RowWeight("Registration", {"population": "paediatric"}, 20.0)]
    data = run(ErrorSpec(rate=0.02, rate_by_table={"Trial": 0}, row_weights=rw), parents=2000)
    frames = data.frames()
    joined = frames["Registration_clean"].merge(frames["Trial_clean"], on="trial_id").set_index("reg_id")
    share_rows = (joined["population"] == "paediatric").mean()
    share_errors = (joined.loc[data.errors["key"], "population"] == "paediatric").mean()
    assert share_errors > 2 * share_rows


# -- detectability -----------------------------------------------------------------------
def test_final_classification_matches_detectable_recall():
    data = run(ErrorSpec(rate=0.08), count=30, fanout=Poisson(3))
    assert list(data.errors["detectable"]) == slow_detectable(SCHEMA, data.spec.rules, data.frames(), data.errors)


@pytest.mark.parametrize("attribute", ["design", "control", "sex", "arm"])
def test_detectable_share_one_and_zero_on_a_single_attribute(attribute):
    for share in (1.0, 0.0):
        data = run(ErrorSpec(rate=0.05, eligible=[attribute], detectable_share=share), count=30)
        log = data.errors
        assert len(log) > 0
        assert log["detectable"].all() if share == 1.0 else not log["detectable"].any()
        assert (log["intended"] == ("detectable" if share else "silent")).all()


def test_detectable_share_on_all_attributes():
    data = run(ErrorSpec(rate=0.02, detectable_share=1.0), count=30)
    assert (data.errors["intended"] == "detectable").all()
    assert data.errors["detectable"].mean() > 0.9           # errors in one family can mask each other
    half = run(ErrorSpec(rate=0.05, detectable_share=0.5), count=30)
    short = {(s["attribute"], s["kind"]): s["wanted"] - s["got"] for s in half.stats["errors"]["shortfalls"]}
    for a in half.stats["errors"]["eligible_attributes"]:
        n = half.clean.n_parents if SCHEMA.is_parent_attr(a) else half.clean.n_children
        k = math.floor(0.05 * n + 0.5)
        k_det = math.floor(0.5 * k + 0.5)
        got = half.errors[half.errors["attribute"] == a]["intended"].value_counts()
        assert got.get("detectable", 0) == k_det - short.get((a, "detectable"), 0)
        assert got.get("silent", 0) == k - k_det - short.get((a, "silent"), 0)


def test_shortfall_is_reported_and_can_be_filled():
    errors = ErrorSpec(rate=0.5, eligible=["sex"], detectable_share=1.0)
    skip = run(errors, count=30)
    [shortfall] = skip.stats["errors"]["shortfalls"]
    assert shortfall["attribute"] == "sex" and shortfall["got"] < shortfall["wanted"] == 2500
    assert len(skip.errors) == shortfall["got"]
    filled = run(ErrorSpec(rate=0.5, eligible=["sex"], detectable_share=1.0, on_shortfall="fill"), count=30)
    assert len(filled.errors) == 2500
    assert (filled.errors["intended"] == "silent").sum() == 2500 - shortfall["got"]


def test_target_rules_restrict_detectable_errors():
    data = run(ErrorSpec(rate=0.05, eligible=["pregnant"], detectable_share=1.0, target_rules=["C3"]), count=30)
    frames = data.frames()
    child = frames["Registration_dirty"].set_index("reg_id").loc[data.errors["key"]]
    assert (child["sex"] == "male").all() and (child["pregnant"] == "Yes").all()


def test_invalid_specs_are_rejected():
    for bad in (
        ErrorSpec(rate=1.5),
        ErrorSpec(rate_by_table={"Nope": 0.1}),
        ErrorSpec(mechanism={"typo": 1}),
        ErrorSpec(detectable_share=2),
        ErrorSpec(target_rules=["ZZ"]),
        ErrorSpec(row_weights=[RowWeight("Trial", {"age_group": "adult"}, 2)]),
    ):
        with pytest.raises(ValueError):
            run(bad, parents=10)


# -- the injector on its own ---------------------------------------------------------------
def test_injector_works_on_data_it_did_not_generate():
    """Hand-made clean frames, no generator involved."""
    import pandas as pd

    from datagen import Rule, two_tables
    from datagen.data import Coded
    from datagen.injector import inject_errors

    schema = two_tables("Shop", "shop_id", {"kind": ("food", "tech")}, "Order", "order_id", {"big": ("Yes", "No")})
    rules = [Rule.of({"kind": "food", "big": "Yes"}, "R1")]
    shops = pd.DataFrame({"shop_id": [str(i) for i in range(200)], "kind": ["food", "tech"] * 100})
    orders = pd.DataFrame({"order_id": [str(i) for i in range(400)], "shop_id": [str(i // 2) for i in range(400)],
                           "big": ["No"] * 400})
    clean = Coded.from_frames(schema, shops, orders)
    a = inject_errors(schema, rules, clean, ErrorSpec(rate=0.1), rng=1)
    b = inject_errors(schema, rules, clean, ErrorSpec(rate=0.1), rng=1)
    assert a.log.equals(b.log)
    assert len(a.log) == 20 + 40
    assert (a.log["clean"] != a.log["dirty"]).all()
    detectable = a.log[a.log["table"] == "Order"]
    assert detectable["detectable"].sum() > 0     # big=Yes in a food shop's order breaks R1


def test_generator_and_injector_do_not_import_each_other():
    import ast
    from pathlib import Path

    import datagen

    root = Path(datagen.__file__).parent
    for package, forbidden in (("generator", "injector"), ("injector", "generator")):
        for path in (root / package).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    assert forbidden not in (node.module or ""), f"{path.name} imports {node.module}"
