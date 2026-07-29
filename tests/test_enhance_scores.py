"""The diagnostic scores must measure the algorithm, not the base class.

``exploration`` previously scored the *initial* population diversity. Initial
diversity is produced by ``_Base._init_pop()``, so it is identical for every
algorithm in the library and, on the diagnostic suite's search range, saturated
the 0-100 scale at 100 for all of them. The score therefore discriminated
nothing and the advisor could never suggest a fix for premature convergence --
the weakness it most often needed to report.

It now measures diversity *maintenance* (mean XPL% after Hussain et al., 2019).
These tests pin that.
"""
import numpy as np
import pytest

from heurilab.analyzer.enhance import (
    _TrackingWrapper, _calc_diversity, _exploration_percentage,
)
from heurilab.core.functions import F1
from heurilab.algorithms import PSO, GWO, DE

POP, DIM, ITER = 30, 10, 120
LB, UB = -100.0, 100.0


def _xpl(cls, seed=1):
    wrapper = _TrackingWrapper(F1, POP)
    cls(pop_size=POP, dim=DIM, lb=LB, ub=UB, max_iter=ITER,
        obj_func=wrapper, seed=seed).optimize()
    return _exploration_percentage(_calc_diversity(wrapper.positions))


def test_a_population_that_never_moves_scores_fully_explorative():
    """Constant diversity means none was lost: XPL% = 100 by construction."""
    assert _exploration_percentage([5.0] * 50) == pytest.approx(100.0)


def test_a_population_that_collapses_immediately_scores_near_zero():
    assert _exploration_percentage([10.0] + [0.0] * 99) < 2.0


@pytest.mark.parametrize("diversities", [[], [3.0], [0.0, 0.0, 0.0]])
def test_degenerate_input_scores_zero_rather_than_raising(diversities):
    assert _exploration_percentage(diversities) == 0.0


def test_score_stays_inside_the_nominal_range():
    rng = np.random.default_rng(0)
    for _ in range(100):
        d = list(rng.random(rng.integers(2, 80)) * rng.choice([1e-6, 1.0, 1e6]))
        assert 0.0 <= _exploration_percentage(d) <= 100.0


def test_exploration_discriminates_between_algorithms():
    """The regression: these three must not all receive the same score.

    Before the fix every algorithm scored exactly 100.0, because the quantity
    being measured was the seeded uniform draw they all inherit.
    """
    scores = {name: _xpl(cls) for name, cls in
              [("PSO", PSO), ("GWO", GWO), ("DE", DE)]}
    assert len(set(round(s, 6) for s in scores.values())) == 3, scores
    assert max(scores.values()) - min(scores.values()) > 10.0, scores


def test_greedy_differential_evolution_retains_more_spread_than_grey_wolf():
    """A directional check, not merely a numeric one.

    GWO drives the whole population towards three leaders and is known to
    contract quickly; DE's greedy per-individual acceptance preserves spread.
    Any rewrite of the metric that inverts this ordering is wrong.
    """
    assert _xpl(DE) > _xpl(GWO)
