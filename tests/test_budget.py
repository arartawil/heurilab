"""
Function-evaluation budgets.

Comparing metaheuristics at equal ``max_iter`` is not comparing them fairly:
algorithms differ several-fold in evaluations consumed per iteration. These
tests pin the two properties that make ``max_fes`` a fair basis - the cap is
hard, and the calibrated iteration count spends it.
"""
import numpy as np
import pytest

from heurilab import get_unimodal_suite, run_experiment
from heurilab.algorithms import DE, GWO, HHO, LSHADE, OOA, PSO, TSA, WOA
from heurilab.algorithms.base import BudgetExhausted, _Base
from heurilab.core.budget import (
    budget_report, calibrate_all, calibrate_iterations, measure_evals_per_iteration,
)

CHEAP = [("PSO", PSO), ("GWO", GWO), ("DE", DE), ("WOA", WOA)]
EXPENSIVE = [("TSA", TSA), ("OOA", OOA), ("HHO", HHO)]


def sphere(x):
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


# ── the problem the budget exists to solve ───────────────────────────

def test_equal_iterations_give_unequal_budgets():
    """The premise: at equal max_iter some algorithms search several times more."""
    cheap = measure_evals_per_iteration(PSO, pop_size=30, dim=20)[0]
    dear = measure_evals_per_iteration(TSA, pop_size=30, dim=20)[0]
    assert dear / cheap > 2.0, (
        "TSA is expected to cost far more per iteration than PSO; if this no "
        "longer holds the fairness argument needs revisiting")


# ── measurement ──────────────────────────────────────────────────────

def test_per_iteration_cost_excludes_startup():
    """Two-point measurement must not fold the initial population into the slope."""
    per_iter, startup = measure_evals_per_iteration(PSO, pop_size=30, dim=10)
    assert per_iter == pytest.approx(30.0, abs=0.5)
    assert startup >= 0.0


@pytest.mark.parametrize("name,cls", CHEAP + EXPENSIVE, ids=lambda v: getattr(v, "__name__", v))
def test_calibration_returns_a_usable_iteration_count(name, cls):
    iters, per_iter = calibrate_iterations(cls, 6000, pop_size=30, dim=10)
    assert iters >= 2
    assert per_iter > 0
    assert iters * per_iter == pytest.approx(6000, rel=0.25)


def test_more_expensive_algorithms_get_fewer_iterations():
    cheap = calibrate_iterations(PSO, 9000, pop_size=30, dim=10)[0]
    dear = calibrate_iterations(TSA, 9000, pop_size=30, dim=10)[0]
    assert dear < cheap


# ── the cap ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,cls", CHEAP + EXPENSIVE + [("LSHADE", LSHADE)],
                         ids=lambda v: getattr(v, "__name__", v))
def test_the_cap_is_never_exceeded(name, cls):
    """Even with max_iter set absurdly high, the run stops at the budget."""
    budget = 2000
    algo = cls(pop_size=20, dim=10, lb=-100.0, ub=100.0, max_iter=100000,
               obj_func=sphere, seed=1, max_fes=budget)
    sol, fit, conv = algo.optimize()
    assert algo.n_fes <= budget
    assert np.isfinite(fit)
    assert np.asarray(sol, dtype=float).shape == (10,)
    assert len(conv) >= 2
    assert np.all(np.isfinite(np.asarray(conv, dtype=float)))


def test_a_budget_terminated_run_returns_the_best_seen():
    algo = PSO(pop_size=20, dim=8, lb=-100.0, ub=100.0, max_iter=100000,
               obj_func=sphere, seed=3, max_fes=900)
    sol, fit, _ = algo.optimize()
    assert fit == pytest.approx(algo.best_fitness)
    assert fit == pytest.approx(sphere(sol), rel=1e-9)


def test_convergence_trace_is_resampled_to_max_iter():
    algo = PSO(pop_size=20, dim=8, lb=-100.0, ub=100.0, max_iter=40,
               obj_func=sphere, seed=3, max_fes=300)
    _, _, conv = algo.optimize()
    assert len(conv) == 41
    arr = np.asarray(conv, dtype=float)
    assert np.all(np.diff(arr) <= 1e-12), "best-so-far must never worsen"


def test_no_budget_means_no_cap():
    algo = PSO(pop_size=20, dim=8, lb=-100.0, ub=100.0, max_iter=25,
               obj_func=sphere, seed=1)
    algo.optimize()
    assert algo.max_fes is None
    assert algo.n_fes > 25          # ran to completion, uncapped


def test_budget_sentinel_survives_a_broad_except():
    """A user algorithm catching Exception must not be able to run past its cap."""
    class Greedy(_Base):
        def optimize(self):
            X = self._init_pop()
            for _ in range(10_000):
                for i in range(self.pop_size):
                    try:
                        self._eval(X[i])
                    except Exception:            # deliberately over-broad
                        pass
            return X[0], 0.0, [0.0]

    algo = Greedy(pop_size=10, dim=4, lb=-5.0, ub=5.0, max_iter=50,
                  obj_func=sphere, seed=1, max_fes=250)
    algo.optimize()
    assert algo.n_fes <= 250, "BudgetExhausted was swallowed by 'except Exception'"


def test_budget_exhausted_is_not_an_exception_subclass():
    assert issubclass(BudgetExhausted, BaseException)
    assert not issubclass(BudgetExhausted, Exception)


# ── campaign level ───────────────────────────────────────────────────

def test_experiment_equalises_budgets_across_algorithms(tmp_path):
    out = tmp_path / "out"
    budget = 3000
    run_experiment(
        algorithms=[("PSO", PSO), ("TSA", TSA), ("OOA", OOA)],
        benchmark_suites=[get_unimodal_suite()], output_dir=str(out),
        pop_size=30, dim=10, n_runs=2, max_fes=budget, seed=5,
        run_engineering=False,
    )
    lines = (out / "CSV Data" / "raw_runs.csv").read_text().splitlines()
    header = lines[0].split(",")
    col = header.index("FEs")
    used = [int(ln.split(",")[col]) for ln in lines[1:]]
    assert max(used) <= budget, "an algorithm exceeded the shared budget"
    assert min(used) > 0.85 * budget, "an algorithm was starved of its budget"


def test_budget_report_documents_the_spread():
    text = budget_report(CHEAP + EXPENSIVE, max_fes=6000, pop_size=30, dim=10)
    assert "evaluations/iteration" in text
    assert "calibrated max_iter" in text
    assert "Cost spread" in text


def test_calibrate_all_covers_every_algorithm():
    cal = calibrate_all(CHEAP, 5000, pop_size=20, dim=10)
    assert set(cal) == {"PSO", "GWO", "DE", "WOA"}
    assert all(iters > 0 and per > 0 for iters, per in cal.values())
