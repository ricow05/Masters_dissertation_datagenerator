import numpy as np
import pytest

from datagen import Empirical, ErrorSpec, Fixed, Geometric, Poisson, UniformRange, generate, inject
from datagen.examples import trials
from datagen.generator.fanout import fanout_from_dict


def test_fixed_gives_exactly_k():
    data = generate(trials.spec(parents=400, children=7))
    assert set(data.clean.children_per_parent()) == {7}
    assert data.stats["children_per_parent"]["histogram"] == {"7": 400}


def test_poisson_mean_and_zeros():
    counts = Poisson(2.0).sample(np.random.default_rng(0), 50_000)
    assert counts.mean() == pytest.approx(2.0, abs=0.03)
    assert (counts == 0).mean() == pytest.approx(np.exp(-2), abs=0.01)


def test_uniform_range_is_inclusive():
    counts = UniformRange(1, 3).sample(np.random.default_rng(0), 10_000)
    assert set(counts) == {1, 2, 3}


def test_geometric_mean():
    counts = Geometric(0.25).sample(np.random.default_rng(0), 50_000)
    assert counts.min() == 0
    assert counts.mean() == pytest.approx(3.0, abs=0.08)


def test_empirical_weights():
    counts = Empirical({0: 1, 2: 3}).sample(np.random.default_rng(0), 40_000)
    assert set(counts) == {0, 2}
    assert (counts == 2).mean() == pytest.approx(0.75, abs=0.01)


@pytest.mark.parametrize("fanout", [Fixed(3), UniformRange(0, 4), Poisson(1.5), Geometric(0.3), Empirical({1: 2, 4: 1})])
def test_round_trip(fanout):
    assert fanout_from_dict(fanout.to_dict()) == fanout


def test_zero_and_one_child_parents_are_generated():
    data = inject(generate(trials.spec(parents=2000, fanout=Poisson(1.0))), ErrorSpec(rate=0.05))
    cpp = data.stats["children_per_parent"]
    assert cpp["share_0"] > 0.3 and cpp["share_1"] > 0.3
    assert data.stats["violations"]["clean"] == 0
