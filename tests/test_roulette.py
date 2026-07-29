"""Roulette-wheel selection must survive a converged population.

``numpy.random.Generator.choice`` requires ``p`` to sum to 1 within a tight
tolerance. Several published pseudo-codes normalise with ``w / (w.sum() + eps)``,
which is short of 1 by construction and fails outright once every weight
collapses to zero -- the state a converging population reaches. These tests pin
the behaviour that replaced it.
"""
import numpy as np
import pytest

from heurilab.algorithms.base import roulette_probabilities
from heurilab.algorithms import ALL_ALGORITHMS

# The three algorithms whose selection step feeds ``rng.choice(p=...)`` from a
# fitness-derived weight vector that can collapse.
ROULETTE_USERS = ["SBO", "BBO", "IMODE"]
_REG = dict(ALL_ALGORITHMS)


def test_uniform_weights_normalise_to_one():
    p = roulette_probabilities(np.ones(7))
    np.testing.assert_allclose(p.sum(), 1.0)
    np.testing.assert_allclose(p, np.full(7, 1 / 7))


def test_all_zero_weights_fall_back_to_uniform():
    """A converged population prefers no candidate; uniform is the right reading."""
    p = roulette_probabilities(np.zeros(30))
    np.testing.assert_allclose(p.sum(), 1.0)
    np.testing.assert_allclose(p, np.full(30, 1 / 30))


def test_choice_accepts_every_normalised_vector():
    """The real contract: ``choice`` must not reject the vector we hand it.

    ``choice`` tests ``abs(sum(p) - 1) < sqrt(eps)``. The old idiom fell short
    of that by a whole epsilon-relative margin once the weights collapsed; this
    sweeps magnitudes and lengths and requires every result to be accepted.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = rng.random(rng.integers(2, 60)) * rng.choice([1e-18, 1.0, 1e18])
        p = roulette_probabilities(w)
        np.testing.assert_allclose(p.sum(), 1.0)
        rng.choice(p.size, p=p)          # raises if the sum is out of tolerance


@pytest.mark.parametrize("weights", [
    np.array([np.nan, 1.0, 2.0]),
    np.array([np.inf, 1.0]),
    np.array([-5.0, -1.0]),              # negative weights are not probabilities
    np.array([0.0, 0.0, 0.0]),
])
def test_degenerate_weights_still_yield_a_valid_distribution(weights):
    p = roulette_probabilities(weights)
    np.testing.assert_allclose(p.sum(), 1.0)
    assert np.all(p >= 0) and np.all(np.isfinite(p))
    np.random.default_rng(0).choice(p.size, p=p)


@pytest.mark.parametrize("name", ROULETTE_USERS)
def test_converged_population_does_not_crash_selection(name):
    """The regression itself: a long run on an easy objective collapses fitness.

    Before the fix these raised ``ValueError: Probabilities do not sum to 1``
    partway through, so any campaign long enough to converge would die.
    """
    def sphere(x):
        return float(np.sum(np.asarray(x) ** 2))

    algo = _REG[name](pop_size=30, dim=10, lb=-100.0, ub=100.0,
                      max_iter=300, obj_func=sphere, seed=1)
    best, fit, conv = algo.optimize()
    assert np.isfinite(fit)
    assert len(conv) == 301
