import numpy as np
import pandas as pd
import pytest

from datagen import DatasetSpec, Depends, ErrorSpec, Fixed, Poisson, Skewed, generate, inject, load_spec, save_spec
from datagen.examples import trials
from datagen.generator.fit import chi2_sf, fit, infer_schema, mutual_information
from datagen.generator.network import Network
from datagen.generator.stats import total_variation

SCHEMA = trials.schema()
TRUE_EDGES = [("phase", "design"), ("design", "control"), ("sponsor", "scope"), ("control", "placebo"), ("age_group", "consent")]


@pytest.fixture(scope="module")
def truth():
    spec = DatasetSpec(
        SCHEMA,
        [],
        marginals={"phase": Skewed(1.0), "status": Skewed(0.5, towards="last")},
        dependencies=[
            Depends("design", "phase", 0.7),
            Depends("control", "design", 0.7),
            Depends("scope", "sponsor", 0.8),
            Depends("placebo", "control", 0.8, mapping={"none": "No", "placebo": "Yes", "active": "No"}),
            Depends("consent", "age_group", 0.7),
        ],
        fanout=Fixed(2),
        parents=50_000,
        seed=5,
    )
    data = generate(spec)
    frames = data.frames()
    return spec, data, frames["Trial_clean"], frames["Registration_clean"]


def conditional_gap(true_net, fitted_net, data, attr):
    """Expected TVD between the fitted and the true P(attr | parents), over the observed parents."""
    true_table, fitted_table = true_net.tables[attr], fitted_net.tables[attr]
    on = true_net.parents[attr]
    if not on:
        return total_variation(true_table, fitted_table)
    columns = data.clean.parents if SCHEMA.is_parent_attr(attr) else data.clean.joined()
    values = columns[on[0]]
    weights = np.bincount(values, minlength=true_table.shape[0]) / len(values)
    return sum(w * total_variation(true_table[i], fitted_table[i]) for i, w in enumerate(weights))


def test_known_structure_recovers_the_distributions(truth):
    spec, data, parents, children = truth
    result = fit(parents, children, "trial_id", schema=SCHEMA, structure=TRUE_EDGES)
    true_net = Network(SCHEMA, spec.marginals, spec.dependencies)
    fitted_net = Network(SCHEMA, result.spec.marginals, result.spec.dependencies)
    for attr in SCHEMA.attrs:
        assert conditional_gap(true_net, fitted_net, data, attr) < 0.02, attr


def test_chow_liu_recovers_the_edges_and_drops_independent_attributes(truth):
    _, _, parents, children = truth
    result = fit(parents, children, "trial_id", schema=SCHEMA)
    kept = {frozenset((e["on"], e["attr"])) for e in result.report["edges"] if e["kept"]}
    assert kept == {frozenset(e) for e in TRUE_EDGES}
    for e in result.report["edges"]:
        assert not (SCHEMA.is_parent_attr(e["attr"]) and not SCHEMA.is_parent_attr(e["on"]))   # never child -> parent
    assert len(result.report["edges"]) == len(SCHEMA.attrs) - 1                                 # a spanning tree before the test


def test_fitted_spec_generates_similar_data(truth, tmp_path):
    _, data, parents, children = truth
    result = fit(parents, children, "trial_id", schema=SCHEMA)
    spec = load_spec(save_spec(result.spec, tmp_path / "spec.json"))
    assert spec.to_dict() == result.spec.to_dict()
    again = generate(spec, seed=99)
    for a in SCHEMA.attrs:
        assert again.stats["marginals"][a]["tvd"] < 0.02
    for edge in again.stats["edges"]:
        original = next(e for e in data.stats["edges"] if {e["on"], e["attr"]} == {edge["on"], edge["attr"]})
        assert edge["cramers_v"] == pytest.approx(original["cramers_v"], abs=0.03)


def test_empirical_fanout_includes_childless_parents():
    data = generate(trials.spec(parents=20_000, fanout=Poisson(1.5)))
    frames = data.frames()
    result = fit(frames["Trial_clean"], frames["Registration_clean"], "trial_id", schema=SCHEMA, structure="none")
    weights = result.spec.fanout.weights
    total = sum(weights.values())
    assert weights[0] / total == pytest.approx(np.exp(-1.5), abs=0.01)
    assert result.spec.dependencies == []


def test_rule_violations_of_the_input_are_reported():
    data = inject(generate(trials.spec(count=18, parents=1000)), ErrorSpec(rate=0.1))
    frames = data.frames()
    result = fit(frames["Trial_dirty"], frames["Registration_dirty"], "trial_id", schema=SCHEMA, rules=data.spec.rules)
    assert result.report["rule_violations"]["total"] == data.stats["violations"]["dirty"]
    regenerated = generate(result.spec)
    assert regenerated.stats["violations"]["clean"] == 0       # the generator conditions on the rules


def test_inferred_schema_from_raw_tables():
    parents = pd.DataFrame({
        "pid": [1, 2, 3, 4],
        "name": ["a", "b", "c", "d"],
        "kind": ["x", "y", "x", None],
        "status": ["on", "off", "on", "on"],
    })
    children = pd.DataFrame({
        "cid": [10, 11, 12, 13, 14],
        "pid": [1, 1, 2, 9, 3],
        "status": ["ok", "bad", "ok", "ok", "ok"],
    })
    result = fit(parents, children, "pid", max_domain=3, parent_name="Shop", child_name="Order")
    schema = result.spec.schema
    assert schema.parent.key == "pid" and schema.child.key == "cid" and schema.child.foreign_key == "pid"
    assert "name" in result.report["dropped_columns"]
    assert schema.domains["kind"] == ("<missing>", "x", "y")
    assert result.report["renamed_columns"] == {"status": "status_order"}
    assert schema.domains["status_order"] == ("bad", "ok")
    assert result.report["orphan_children_dropped"] == 1
    assert result.spec.fanout.weights == {0: 1.0, 1: 2.0, 2: 1.0}


def test_statistics_helpers():
    assert chi2_sf(0, 4) == 1.0
    assert chi2_sf(9.488, 4) == pytest.approx(0.05, abs=0.003)      # chi2(4) 95% quantile
    assert chi2_sf(23.21, 10) == pytest.approx(0.01, abs=0.001)
    x = np.repeat(np.arange(2), 500)
    assert mutual_information(x, x, 2, 2) == pytest.approx(np.log(2))
    assert mutual_information(x, np.tile(np.arange(2), 500), 2, 2) == pytest.approx(0.0, abs=1e-9)


def test_infer_schema_requires_the_key_columns():
    with pytest.raises(ValueError, match="foreign key"):
        infer_schema(pd.DataFrame({"pid": [1]}), pd.DataFrame({"x": [1]}), "pid")
