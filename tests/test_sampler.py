from collections import Counter
from itertools import product

import numpy as np
import pytest

from conftest import slow_violations
from datagen import (
    CPT, DatasetSpec, Depends, Fixed, Frequencies, Normal, Poisson, Rule, Skewed, Uniform, generate,
    two_tables,
)
from datagen.examples import trials
from datagen.generator.network import Network
from datagen.generator.sampler import CombinationLimit, Sampler, find_blocks
from datagen.generator.stats import cramers_v, total_variation


# -- rules hold ------------------------------------------------------------------------
SETTINGS = {
    "uniform": dict(fanout=Fixed(5)),
    "skewed": dict(
        fanout=Poisson(3),
        marginals={
            "phase": Skewed(1.5),
            "design": Skewed(1.0, towards="last"),
            "control": Frequencies({"none": 1, "placebo": 5, "active": 2}),
            "blinding": Normal("double", 0.7),
            "age_group": Skewed(2.0, towards="last"),
            "status": Normal(),
        },
    ),
    "correlated": dict(
        fanout=Poisson(2),
        dependencies=[
            Depends("control", "design", 0.7),
            Depends("blinding", "control", 0.6),
            Depends("scope", "sponsor", 0.5),
            Depends("age_group", "population", 0.8),
            Depends("placebo", "control", 0.9, mapping={"none": "No", "placebo": "Yes", "active": "No"}),
            Depends("outcome", "status", 0.5),
            CPT("arm", ("design",), {("crossover",): {"both": 9, "experimental": 1}}),
        ],
    ),
}


@pytest.mark.parametrize("setting", SETTINGS)
def test_clean_data_satisfies_every_rule_prefix(setting):
    schema = trials.schema()
    for count in range(1, trials.MAX_RULES + 1):
        spec = trials.spec(count=count, parents=300, seed=count, **SETTINGS[setting])
        data = generate(spec)
        frames = data.frames()
        found = slow_violations(schema, spec.rules, frames["Trial_clean"], frames["Registration_clean"])
        assert found == 0, f"{setting}, {count} rules: {found} violations"
        assert data.stats["violations"]["clean"] == 0


# -- exactness against brute force -----------------------------------------------------
def tiny_network():
    return dict(
        marginals={"x": Skewed(1.0)},
        dependencies=[Depends("y", "x", 0.6), Depends("u", "x", 0.5), Depends("w", "u", 0.4)],
    )


def brute_force(tiny_rules, n_children, conditioning):
    """Exact distribution of a parent and of a (parent, child) pair, written out by hand."""
    px = dict(zip("abc", np.array([1, 1 / 2, 1 / 3]) / (1 + 1 / 2 + 1 / 3)))
    y_of = {"a": "0", "b": "1", "c": "1"}          # rank alignment of 3 values onto 2
    u_of = {"a": "0", "b": "1", "c": "2"}
    w_of = {"0": "lo", "1": "hi", "2": "hi"}

    def p_parent(x, y):
        return px[x] * (0.4 * 0.5 + 0.6 * (y == y_of[x]))

    def p_child(x, u, w):
        return (0.5 / 3 + 0.5 * (u == u_of[x])) * (0.6 * 0.5 + 0.4 * (w == w_of[u]))

    def ok(row, kinds):
        return not any(r.fails(row) for r in tiny_rules if r.name.startswith(kinds))

    parents = [dict(x=x, y=y) for x, y in product("abc", "01")]
    children = [dict(u=u, w=w) for u, w in product("012", ("lo", "hi"))]
    weight, child_dist = {}, {}
    for p in parents:
        key = (p["x"], p["y"])
        cw = {(c["u"], c["w"]): p_child(p["x"], c["u"], c["w"]) * ok({**p, **c}, ("C", "PC")) for c in children}
        z = sum(cw.values())
        w = p_parent(*key) * ok(p, ("P1",))
        if n_children:
            w *= (z > 0) if conditioning == "sequential" else z**n_children
        weight[key] = w
        child_dist[key] = {k: v / z for k, v in cw.items()} if z else {}
    total = sum(weight.values())
    parent_dist = {k: v / total for k, v in weight.items()}
    pair_dist = {(k, c): parent_dist[k] * q for k in parent_dist for c, q in child_dist[k].items()}
    return parent_dist, pair_dist


def empirical(counter, keys):
    n = sum(counter.values())
    return np.array([counter.get(k, 0) / n for k in keys])


@pytest.mark.parametrize("conditioning", ["sequential", "joint"])
def test_sampler_matches_brute_force(tiny_schema, tiny_rules, conditioning):
    spec = DatasetSpec(
        tiny_schema, tiny_rules, fanout=Fixed(2), parents=100_000, seed=7,
        conditioning=conditioning, **tiny_network(),
    )
    frames = generate(spec).frames()
    parents, children = frames["P_clean"], frames["C_clean"]
    parent_dist, pair_dist = brute_force(tiny_rules, 2, conditioning)

    seen = Counter(zip(parents["x"], parents["y"]))
    keys = list(parent_dist)
    assert total_variation(empirical(seen, keys), [parent_dist[k] for k in keys]) < 0.01
    assert parent_dist[("c", "0")] == parent_dist[("c", "1")] == 0     # x=c leaves no valid child

    joined = children.merge(parents, left_on="pid", right_on="pid")
    seen = Counter(zip(zip(joined["x"], joined["y"]), zip(joined["u"], joined["w"])))
    keys = list(pair_dist)
    assert set(seen) <= set(keys)
    assert total_variation(empirical(seen, keys), [pair_dist[k] for k in keys]) < 0.015


def test_conditionings_differ_when_children_matter(tiny_rules):
    seq, _ = brute_force(tiny_rules, 2, "sequential")
    joint, _ = brute_force(tiny_rules, 2, "joint")
    assert total_variation(list(seq.values()), list(joint.values())) > 0.02


def test_parent_without_children_may_take_an_implied_infeasible_value(tiny_schema, tiny_rules):
    spec = DatasetSpec(tiny_schema, tiny_rules, fanout=Fixed(0), parents=60_000, seed=1, **tiny_network())
    frames = generate(spec).frames()
    parent_dist, _ = brute_force(tiny_rules, 0, "sequential")
    seen = Counter(zip(frames["P_clean"]["x"], frames["P_clean"]["y"]))
    keys = list(parent_dist)
    assert parent_dist[("c", "1")] > 0.1
    assert total_variation(empirical(seen, keys), [parent_dist[k] for k in keys]) < 0.01
    assert len(frames["C_clean"]) == 0


def test_mixed_fanout_uses_feasibility_only_for_parents_with_children(tiny_schema, tiny_rules):
    spec = DatasetSpec(tiny_schema, tiny_rules, fanout=Poisson(1.0), parents=20_000, seed=3, **tiny_network())
    data = generate(spec)
    frames = data.frames()
    counts = data.clean.children_per_parent()
    x = frames["P_clean"]["x"].to_numpy()
    assert (x[counts > 0] != "c").all()
    assert (x[counts == 0] == "c").any()
    assert slow_violations(tiny_schema, tiny_rules, frames["P_clean"], frames["C_clean"]) == 0


# -- distributions and correlation -----------------------------------------------------
def test_without_rules_realised_marginals_match_the_targets():
    schema = trials.schema()
    marginals = {"phase": Skewed(1.2), "blinding": Normal(0.5, 0.8), "control": Frequencies({"placebo": 3, "active": 1})}
    spec = DatasetSpec(schema, [], marginals=marginals, fanout=Fixed(1), parents=50_000)
    stats = generate(spec).stats
    for a in schema.attrs:
        assert stats["marginals"][a]["tvd"] < 0.01, a
    assert stats["marginals"]["control"]["realised"]["none"] == 0


def test_marginal_shapes():
    domain = ("I", "II", "III", "IV")
    p = Skewed(1.0).probabilities(domain)
    assert np.allclose(p, np.array([1, 1 / 2, 1 / 3, 1 / 4]) / sum([1, 1 / 2, 1 / 3, 1 / 4]))
    assert np.allclose(Skewed(1.0, towards="last").probabilities(domain), p[::-1])
    n = Normal().probabilities(domain)
    assert np.allclose(n, n[::-1]) and n[1] > n[0]
    assert Normal("I", 0.5).probabilities(domain).argmax() == 0
    assert np.allclose(Uniform().probabilities(domain), 0.25)
    with pytest.raises(ValueError):
        Frequencies({"V": 1}).probabilities(domain)


def test_dependency_strength_controls_association():
    schema = trials.schema()
    v = []
    for strength in (0.0, 0.5, 1.0):
        spec = DatasetSpec(schema, [], dependencies=[Depends("scope", "sponsor", strength)],
                           fanout=Fixed(1), parents=20_000)
        data = generate(spec)
        v.append(data.stats["edges"][0]["cramers_v"])
    assert v[0] < 0.03 < v[1] < v[2]
    assert v[2] == pytest.approx(1.0)


def test_parent_to_child_dependency():
    schema = trials.schema()
    spec = DatasetSpec(schema, [], dependencies=[Depends("age_group", "population", 1.0,
                       mapping={"adult": "adult", "paediatric": "paediatric", "mixed": "adult"})],
                       fanout=Fixed(3), parents=2000)
    frames = generate(spec).frames()
    joined = frames["Registration_clean"].merge(frames["Trial_clean"], on="trial_id")
    expected = joined["population"].map({"adult": "adult", "paediatric": "paediatric", "mixed": "adult"})
    assert (joined["age_group"] == expected).all()


def test_network_rejects_child_to_parent_edges_and_cycles():
    schema = trials.schema()
    with pytest.raises(ValueError, match="cannot depend on child"):
        Network(schema, {}, [Depends("phase", "age_group")])
    with pytest.raises(ValueError, match="cycle"):
        Network(schema, {}, [Depends("phase", "design"), Depends("design", "phase")])
    with pytest.raises(ValueError, match="more than one"):
        Network(schema, {}, [Depends("phase", "design"), Depends("phase", "control")])


# -- blocks and limits -----------------------------------------------------------------
def test_blocks_follow_rules_and_edges():
    schema = trials.schema()
    rules = trials.rules(12)
    blocks = find_blocks(schema, rules, Network(schema))
    as_sets = [set(b.attrs) for b in blocks]
    assert {"sponsor", "scope"} in as_sets
    assert {"status", "outcome"} in as_sets
    with_edge = find_blocks(schema, rules, Network(schema, {}, [Depends("scope", "phase")]))
    assert {"sponsor", "scope"} not in [set(b.attrs) for b in with_edge]


def test_combination_limit_names_the_block():
    schema = trials.schema()
    with pytest.raises(CombinationLimit, match="phase"):
        Sampler(schema, trials.rules(30), Network(schema), max_combinations=100)


def test_inconsistent_rules_raise():
    schema = two_tables("P", "pid", {"x": ("a", "b")}, "C", "cid", {"u": ("0", "1")})
    rules = [Rule.of({"x": "a"}, "R1"), Rule.of({"x": "b"}, "R2")]
    spec = DatasetSpec(schema, rules, parents=10)
    with pytest.raises(ValueError, match="no parent combination"):
        generate(spec)


# -- seeds ------------------------------------------------------------------------------
def test_same_seed_same_data_and_other_seed_other_data():
    a = generate(trials.spec(parents=500, seed=11)).frames()
    b = generate(trials.spec(parents=500, seed=11)).frames()
    c = generate(trials.spec(parents=500, seed=12)).frames()
    for name in a:
        assert a[name].equals(b[name])
    assert not a["Trial_clean"].equals(c["Trial_clean"])


def test_cramers_v_extremes():
    x = np.repeat(np.arange(3), 99)
    assert cramers_v(x, x, 3, 3) == pytest.approx(1.0)
    assert cramers_v(x, np.tile(np.arange(3), 99), 3, 3) == pytest.approx(0.0)
