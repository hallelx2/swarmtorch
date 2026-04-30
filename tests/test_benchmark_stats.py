"""Tests for statistical tests against scipy reference (Stage 2.3)."""

import numpy as np
import pytest
from scipy import stats

from swarmtorch.benchmark.stats import (
    friedman_test,
    nemenyi_critical_difference,
    rank_algorithms,
    wilcoxon_test,
)


def test_friedman_matches_scipy_reference():
    np.random.seed(0)
    scores = np.random.rand(10, 4)
    ours = friedman_test(scores)
    ref_stat, ref_p = stats.friedmanchisquare(*scores.T)
    assert ours.statistic == pytest.approx(ref_stat, rel=1e-9)
    assert ours.pvalue == pytest.approx(ref_p, rel=1e-9)
    assert ours.n_algorithms == 4
    assert ours.n_tasks == 10


def test_friedman_rejects_when_one_algo_dominates():
    # Algorithm 0 always best: scores 1, 2, 3, 4 across columns.
    scores = np.tile([1.0, 2.0, 3.0, 4.0], (10, 1))
    res = friedman_test(scores)
    assert res.reject_null


def test_friedman_does_not_reject_when_random():
    np.random.seed(123)
    scores = np.random.rand(20, 5)
    res = friedman_test(scores)
    # Pure noise — should usually fail to reject. Allow occasional false
    # positives by checking p > 0.01 rather than > 0.05.
    assert res.pvalue > 0.01


def test_nemenyi_cd_matches_known_value():
    # Demsar 2006 example: k=4 algorithms, N=14 datasets, alpha=0.05.
    # CD = 2.569 * sqrt(4*5 / (6*14)) = 2.569 * sqrt(0.2381) = 1.253.
    cd = nemenyi_critical_difference(n_algorithms=4, n_tasks=14, alpha=0.05)
    assert cd == pytest.approx(1.253, abs=0.01)


def test_nemenyi_cd_invalid_alpha():
    with pytest.raises(ValueError):
        nemenyi_critical_difference(4, 10, alpha=0.025)


def test_nemenyi_cd_invalid_k():
    with pytest.raises(ValueError):
        nemenyi_critical_difference(50, 10)


def test_rank_algorithms_basic():
    scores = np.array([[3.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
    ranks = rank_algorithms(scores)
    # Row 0 ranks: 3, 1, 2. Row 1 ranks: 2, 3, 1. Mean: 2.5, 2.0, 1.5.
    assert ranks == pytest.approx([2.5, 2.0, 1.5])


def test_wilcoxon_matches_scipy():
    np.random.seed(0)
    a = np.random.rand(20)
    b = a + np.random.rand(20) * 0.1
    ours = wilcoxon_test(a, b)
    ref = stats.wilcoxon(a, b, zero_method="pratt")
    assert ours.statistic == pytest.approx(float(ref.statistic), rel=1e-9)
    assert ours.pvalue == pytest.approx(float(ref.pvalue), rel=1e-9)
    assert ours.n == 20


def test_wilcoxon_too_small_sample():
    with pytest.raises(ValueError):
        wilcoxon_test([1.0, 2.0], [1.5, 2.5])


def test_friedman_validates_shape():
    with pytest.raises(ValueError):
        friedman_test(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        friedman_test(np.array([[1.0]]))
    # 2 algorithms is below scipy's minimum.
    with pytest.raises(ValueError):
        friedman_test(np.array([[1.0, 2.0], [3.0, 4.0]]))
