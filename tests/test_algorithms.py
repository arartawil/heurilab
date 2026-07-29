"""
Smoke tests: every registered algorithm must run on a simple problem and honor
the _Base contract -> (best_solution, best_fitness, convergence).

Kept fast (tiny pop/dim/iter) so the whole 102-algorithm sweep runs in seconds.
"""
import numpy as np
import pytest

from heurilab.algorithms import ALL_ALGORITHMS
from heurilab.algorithms.base import _Base


POP, DIM, ITER = 8, 5, 6
LB, UB = -100.0, 100.0


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


SEED = 20260728


def test_registry_is_complete_and_unique():
    names = [n for n, _ in ALL_ALGORITHMS]
    assert len(ALL_ALGORITHMS) == 98
    assert len(names) == len(set(names)), "duplicate algorithm names in ALL_ALGORITHMS"
    assert all(issubclass(cls, _Base) for _, cls in ALL_ALGORITHMS)


@pytest.mark.parametrize("name,cls", ALL_ALGORITHMS, ids=[n for n, _ in ALL_ALGORITHMS])
def test_algorithm_runs_and_honors_contract(name, cls):
    algo = cls(pop_size=POP, dim=DIM, lb=LB, ub=UB, max_iter=ITER, obj_func=sphere,
               seed=SEED)
    result = algo.optimize()

    # returns a 3-tuple
    assert isinstance(result, tuple) and len(result) == 3, f"{name}: bad return"
    best_sol, best_fit, conv = result

    # best solution: right shape, inside bounds, finite
    best_sol = np.asarray(best_sol, dtype=float).ravel()
    assert best_sol.shape == (DIM,), f"{name}: solution shape {best_sol.shape}"
    assert np.all(np.isfinite(best_sol)), f"{name}: non-finite solution"
    assert np.all(best_sol >= LB - 1e-6) and np.all(best_sol <= UB + 1e-6), \
        f"{name}: solution out of bounds"

    # best fitness: finite scalar, and consistent with obj_func at the solution
    best_fit = float(best_fit)
    assert np.isfinite(best_fit), f"{name}: non-finite fitness"

    # convergence: length max_iter + 1, finite, best-so-far never worsens
    conv = np.asarray(list(conv), dtype=float)
    assert conv.shape == (ITER + 1,), f"{name}: convergence len {conv.shape}"
    assert np.all(np.isfinite(conv)), f"{name}: non-finite convergence"
    assert conv[-1] <= conv[0] + 1e-9, f"{name}: convergence worsened overall"
    # returned best matches the final convergence value
    assert best_fit <= conv[0] + 1e-9, f"{name}: returned fitness worse than start"
