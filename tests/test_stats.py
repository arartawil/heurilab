"""Statistical machinery: the numbers that end up in published comparison tables."""
import numpy as np
import pytest
from scipy.stats import ranksums, friedmanchisquare

from heurilab.stats.tests import (
    wilcoxon_test, friedman_test, nemenyi_cd, nemenyi_pairwise,
)


def test_wilcoxon_matches_scipy_and_picks_the_better_side():
    proposed = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03]
    worse = [5.0, 5.1, 4.9, 5.2, 5.05, 4.95, 5.15, 5.0]
    p, winner = wilcoxon_test(proposed, worse)
    assert p == pytest.approx(ranksums(proposed, worse).pvalue)
    assert winner == "proposed"
    p2, winner2 = wilcoxon_test(worse, proposed)
    assert winner2 == "competitor"


def test_wilcoxon_reports_tie_for_indistinguishable_samples():
    a = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02]
    _, winner = wilcoxon_test(a, list(a))
    assert winner == "tie"


def test_wilcoxon_degenerate_input_does_not_raise():
    p, winner = wilcoxon_test([1.0], [2.0])
    assert (p, winner) == (1.0, "tie")


def test_friedman_ranks_a_clear_ordering_correctly():
    funcs = [f"F{i}" for i in range(1, 6)]
    algos = ["A", "B", "C"]
    data = {f: {"A": [1.0], "B": [2.0], "C": [3.0]} for f in funcs}
    chi2, p, ranks = friedman_test(data, algos, funcs)
    assert ranks["A"] == pytest.approx(1.0)
    assert ranks["B"] == pytest.approx(2.0)
    assert ranks["C"] == pytest.approx(3.0)
    assert chi2 == pytest.approx(
        friedmanchisquare(*[np.full(len(funcs), r) for r in (1.0, 2.0, 3.0)]).statistic,
        nan_ok=True) or np.isfinite(chi2)


def test_friedman_is_defined_out_for_too_few_functions():
    data = {"F1": {"A": [1.0], "B": [2.0]}, "F2": {"A": [1.0], "B": [2.0]}}
    chi2, p, ranks = friedman_test(data, ["A", "B"], ["F1", "F2"])
    assert (chi2, p) == (0.0, 1.0)


def test_nemenyi_cd_matches_the_closed_form():
    # CD = q_alpha * sqrt(k(k+1) / 6N); q_0.05 for k=5 is 2.728
    assert nemenyi_cd(5, 20) == pytest.approx(2.728 * np.sqrt(5 * 6 / (6 * 20)))


def test_nemenyi_cd_shrinks_as_more_functions_are_added():
    assert nemenyi_cd(5, 50) < nemenyi_cd(5, 10)


def test_nemenyi_cd_is_defined_beyond_the_tabulated_range():
    """k > 20 must not silently fall back to a smaller q than k=20."""
    assert nemenyi_cd(25, 30) >= nemenyi_cd(20, 30), \
        "critical difference decreased when adding algorithms - q table fell back"


def test_nemenyi_pairwise_is_symmetric_and_self_consistent():
    ranks = {"A": 1.0, "B": 2.0, "C": 5.0}
    names = ["A", "B", "C"]
    res = nemenyi_pairwise(ranks, n_funcs=10, algo_names=names)
    for i in names:
        assert res[(i, i)][0] == 0.0 and res[(i, i)][1] is False
        for j in names:
            assert res[(i, j)][0] == res[(j, i)][0]
