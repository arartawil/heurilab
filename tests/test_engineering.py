"""
Engineering design problems.

The load-bearing test here is :func:`test_optimiser_reaches_published_optimum`.
A mistyped coefficient or a flipped constraint sign will not raise — it will
quietly produce a different optimisation problem — and the only reliable way to
notice is to optimise the problem and check the answer against the literature.
That is what this module does.
"""
import numpy as np
import pytest

from heurilab.algorithms import LSHADE
from heurilab.engineering.problems import (
    ENGINEERING_PROBLEMS, PROBLEMS, PROBLEMS_BY_NAME, EngineeringProblem,
    get_engineering_problems, penalised, PENALTY,
)

IDS = [p.name for p in PROBLEMS]


def test_registry_is_consistent():
    assert len(PROBLEMS) >= 12
    assert len(PROBLEMS_BY_NAME) == len(PROBLEMS), "duplicate problem names"
    assert len(ENGINEERING_PROBLEMS) == len(PROBLEMS)
    for name, fn, dim, lb, ub in ENGINEERING_PROBLEMS:
        assert callable(fn)
        assert len(lb) == len(ub) == dim


@pytest.mark.parametrize("p", PROBLEMS, ids=IDS)
def test_bounds_are_well_formed(p: EngineeringProblem):
    lb, ub = np.asarray(p.lb, float), np.asarray(p.ub, float)
    assert lb.shape == ub.shape == (p.dim,)
    assert np.all(ub > lb), f"{p.name}: non-increasing bounds"


@pytest.mark.parametrize("p", PROBLEMS, ids=IDS)
def test_objective_and_constraints_are_finite_across_the_box(p: EngineeringProblem):
    """No NaN/inf anywhere an optimiser might legitimately sample."""
    rng = np.random.default_rng(0)
    lb, ub = np.asarray(p.lb, float), np.asarray(p.ub, float)
    for x in rng.uniform(lb, ub, size=(200, p.dim)):
        assert np.isfinite(p.objective(x)), f"{p.name}: non-finite objective at {x}"
        gs = p.constraints(x)
        assert len(gs) == p.n_constraints
        assert all(np.isfinite(g) for g in gs), f"{p.name}: non-finite constraint at {x}"


@pytest.mark.parametrize("p", PROBLEMS, ids=IDS)
def test_penalty_never_improves_on_the_raw_objective(p: EngineeringProblem):
    """The penalised form must equal the objective when feasible, exceed it otherwise."""
    rng = np.random.default_rng(1)
    lb, ub = np.asarray(p.lb, float), np.asarray(p.ub, float)
    f = p.penalised()
    for x in rng.uniform(lb, ub, size=(100, p.dim)):
        raw, pen = p.objective(x), f(x)
        if p.is_feasible(x):
            assert pen == pytest.approx(raw, rel=1e-12)
        else:
            assert pen > raw


@pytest.mark.parametrize("p", PROBLEMS, ids=IDS)
def test_optimiser_reaches_published_optimum(p: EngineeringProblem):
    """L-SHADE must reach the published optimum, and do it feasibly.

    Guards against transcription errors in the objective or constraints: a wrong
    formulation optimises to a different value (or to a better-than-published
    value, which means a constraint is too weak).
    """
    best, best_x = np.inf, None
    for seed in (1, 2, 3):
        sol, fit, _ = LSHADE(pop_size=50, dim=p.dim, lb=p.lb, ub=p.ub,
                             max_iter=600, obj_func=p.penalised(), seed=seed).optimize()
        if fit < best:
            best, best_x = fit, np.asarray(sol, dtype=float)

    assert p.is_feasible(best_x, tol=1e-4), (
        f"{p.name}: best solution violates constraints by "
        f"{p.violation(best_x):.3e}")

    rel = abs(best - p.best_known) / max(abs(p.best_known), 1e-9)
    assert rel < 0.02, (
        f"{p.name}: reached {best:.8g}, published optimum is {p.best_known:.8g} "
        f"(relative error {rel:.2%}). Either the formulation is wrong or the "
        f"reference value is.")


@pytest.mark.parametrize("p", PROBLEMS, ids=IDS)
def test_published_optimum_is_not_beaten(p: EngineeringProblem):
    """Beating the literature by a wide margin means a constraint is too loose."""
    best = min(
        LSHADE(pop_size=50, dim=p.dim, lb=p.lb, ub=p.ub, max_iter=600,
               obj_func=p.penalised(), seed=s).optimize()[1]
        for s in (1, 2, 3)
    )
    slack = 0.02 * max(abs(p.best_known), 1e-9)
    assert best > p.best_known - slack, (
        f"{p.name}: found {best:.8g}, better than the published {p.best_known:.8g}. "
        f"A constraint is probably missing or has the wrong sign.")


def test_every_problem_cites_a_source():
    for p in PROBLEMS:
        assert p.reference and len(p.reference) > 5, f"{p.name} has no reference"


def test_penalised_form_is_picklable():
    """Required for engineering runs under n_jobs != 1."""
    import pickle
    for p in PROBLEMS:
        f = pickle.loads(pickle.dumps(p.penalised()))
        x = np.asarray(p.lb, float)
        assert np.isfinite(f(x))


def test_get_engineering_problems_subset_and_errors():
    subset = get_engineering_problems(["Welded Beam Design", "Gear Train Design"])
    assert [p.name for p in subset] == ["Welded Beam Design", "Gear Train Design"]
    assert len(get_engineering_problems()) == len(PROBLEMS)
    with pytest.raises(ValueError, match="Unknown engineering problem"):
        get_engineering_problems(["Nonexistent Problem"])


def test_custom_penalty_coefficient_is_honoured():
    p = PROBLEMS_BY_NAME["Welded Beam Design"]
    x = np.asarray(p.ub, float)          # far outside the feasible region
    assert p.violation(x) > 0
    assert penalised(p, 1.0)(x) < penalised(p, PENALTY)(x)
