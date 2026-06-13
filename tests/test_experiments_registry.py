"""Tests for paper-algorithm registry and operator taxonomy (Stage 4.4 / 4.5)."""

import pytest
from torch import nn

from swarmtorch.experiments.registry import (
    OPERATOR_TAXONOMY,
    PAPER_ALGORITHMS,
    PAPER_HPO_SEARCHERS,
    build_algorithm_factory,
    operator_group,
)


def test_paper_short_list_size_is_sane():
    # ~12 algorithms keeps the headline table legible.
    assert 8 <= len(PAPER_ALGORITHMS) <= 20


def test_taxonomy_covers_every_paper_algorithm():
    classified = {algo for names in OPERATOR_TAXONOMY.values() for algo in names}
    assert classified == set(PAPER_ALGORITHMS)


def test_taxonomy_groups_distinct():
    seen: set[str] = set()
    for group, names in OPERATOR_TAXONOMY.items():
        for name in names:
            assert name not in seen, f"{name} appears in multiple groups"
            seen.add(name)


def test_operator_group_lookup():
    assert operator_group("PSO") == "velocity-based"
    assert operator_group("CMAES") == "distribution-based"
    assert operator_group("Adam") == "gradient"
    assert operator_group("nonexistent") == "unknown"


@pytest.mark.parametrize("name", ["PSO", "CMAES", "Adam"])
def test_factory_builds_valid_optimizer(name):
    model = nn.Linear(4, 2)
    factory = build_algorithm_factory(name, swarm_size=8)
    opt = factory(model)
    # Every algorithm should expose .step at minimum.
    assert hasattr(opt, "step")


def test_factory_unknown_name_raises():
    with pytest.raises(KeyError):
        build_algorithm_factory("not_a_real_algo")


def test_paper_hpo_searchers_distinct():
    assert len(set(PAPER_HPO_SEARCHERS)) == len(PAPER_HPO_SEARCHERS)
