"""Statistical tests for algorithm comparisons.

Implements the Demsar (2006) protocol that the metaheuristic literature
expects: Friedman test for the global null, Nemenyi post-hoc for
pairwise comparisons via critical-difference (CD) diagrams, and
Wilcoxon signed-rank for the headline two-method comparison.

The CD lookup table is embedded so we don't pull in
``scikit-posthocs`` as a dependency for two constants.

References:
    Demsar (2006). Statistical Comparisons of Classifiers over Multiple
        Data Sets. Journal of Machine Learning Research 7, 1-30.
    Garcia & Herrera (2008). An Extension on "Statistical Comparisons of
        Classifiers over Multiple Data Sets" for all Pairwise Comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


# Studentized-range critical values q_alpha for the two-tailed Nemenyi test.
# Indexed by k = number of algorithms. Source: Demsar 2006 Table 5.
_Q_ALPHA_05 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
    13: 3.313,
    14: 3.354,
    15: 3.391,
    16: 3.426,
    17: 3.458,
    18: 3.489,
    19: 3.517,
    20: 3.544,
}
_Q_ALPHA_10 = {
    2: 1.645,
    3: 2.052,
    4: 2.291,
    5: 2.459,
    6: 2.589,
    7: 2.693,
    8: 2.780,
    9: 2.855,
    10: 2.920,
}


@dataclass
class FriedmanResult:
    statistic: float
    pvalue: float
    n_algorithms: int
    n_tasks: int

    @property
    def reject_null(self) -> bool:
        return self.pvalue < 0.05


@dataclass
class WilcoxonResult:
    statistic: float
    pvalue: float
    n: int

    @property
    def significant(self) -> bool:
        return self.pvalue < 0.05


def friedman_test(scores: np.ndarray) -> FriedmanResult:
    """Friedman omnibus test.

    Args:
        scores: 2-D array of shape ``(n_tasks, n_algorithms)``. Each row
            is one task; columns are algorithm scores. Lower is better.

    Returns:
        :class:`FriedmanResult` with the chi-squared statistic and
        p-value. Reject the null (all algorithms equal) when ``pvalue <
        alpha``.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got shape {scores.shape}")
    # scipy.stats.friedmanchisquare requires at least 3 algorithms.
    if scores.shape[0] < 2 or scores.shape[1] < 3:
        raise ValueError(
            f"need >= 2 tasks and >= 3 algorithms, got shape {scores.shape}"
        )
    statistic, pvalue = stats.friedmanchisquare(*scores.T)
    return FriedmanResult(
        statistic=float(statistic),
        pvalue=float(pvalue),
        n_algorithms=int(scores.shape[1]),
        n_tasks=int(scores.shape[0]),
    )


def rank_algorithms(scores: np.ndarray) -> np.ndarray:
    """Average rank per algorithm across tasks (lower-is-better).

    Returns a 1-D array of length ``n_algorithms``.
    """
    scores = np.asarray(scores, dtype=float)
    # Within each row, rank columns ascending (best gets rank 1).
    ranks = np.array([stats.rankdata(row) for row in scores])
    return ranks.mean(axis=0)


def nemenyi_critical_difference(
    n_algorithms: int,
    n_tasks: int,
    alpha: float = 0.05,
) -> float:
    """Critical difference (CD) for the Nemenyi post-hoc test.

    Two algorithms whose average ranks differ by less than CD are not
    statistically significantly different at level ``alpha``.

    ``CD = q_alpha * sqrt(k(k+1) / (6N))`` where ``k`` is the number of
    algorithms and ``N`` is the number of tasks (Demsar 2006 Eq. 8).
    """
    if alpha == 0.05:
        table = _Q_ALPHA_05
    elif alpha == 0.10:
        table = _Q_ALPHA_10
    else:
        raise ValueError(
            f"alpha must be 0.05 or 0.10 (have lookup table for these); got {alpha}"
        )
    if n_algorithms not in table:
        raise ValueError(
            f"n_algorithms={n_algorithms} not in CD lookup table for alpha={alpha}; "
            f"supported: {sorted(table)}"
        )
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    q = table[n_algorithms]
    return float(q * np.sqrt(n_algorithms * (n_algorithms + 1) / (6.0 * n_tasks)))


def wilcoxon_test(
    a: Sequence[float],
    b: Sequence[float],
    zero_method: str = "pratt",
) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test on paired scores.

    Use this for the headline "best metaheuristic vs Adam" comparison
    after Friedman has rejected the global null.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"a and b must have same shape, got {a_arr.shape} vs {b_arr.shape}"
        )
    if a_arr.size < 5:
        # scipy will warn for tiny samples; surface that as a clear error.
        raise ValueError(
            f"Wilcoxon needs at least ~5 paired observations, got {a_arr.size}"
        )
    res = stats.wilcoxon(a_arr, b_arr, zero_method=zero_method)
    return WilcoxonResult(
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        n=int(a_arr.size),
    )
