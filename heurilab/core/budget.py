"""
Function-evaluation budgets.

Comparing metaheuristics at equal ``max_iter`` is not a like-for-like
comparison. Algorithms differ in how many objective evaluations they spend per
iteration, so equal iterations means unequal search. Measured across HeuriLab's
102 algorithms at ``pop_size=30``:

===========================  ==================
algorithm                    evaluations/iter
===========================  ==================
HS                           1.8
PSO, GWO, DE, WOA, EO, SCA   30.8
HHO                          56.9
TSA, OOA                     ~91
BFO                          76.5
===========================  ==================

A 43x spread. An algorithm at 91 evaluations per iteration receives three times
the search of one at 30 when both are run for 500 iterations, and any results
table built that way flatters it.

The fix has two parts, both applied by :func:`heurilab.run_experiment` when
``max_fes`` is set:

1. **Cap the evaluations.** ``_Base`` stops the run the moment the cap is
   reached and returns the best solution found so far.
2. **Calibrate ``max_iter`` to match.** Many algorithms anneal a coefficient
   against ``max_iter`` - GWO's ``a = 2 - 2t/max_iter``, for instance. Leaving
   ``max_iter`` too high means the schedule is still mid-decay when the budget
   runs out, so the algorithm never reaches its exploitation phase.
   :func:`calibrate_iterations` measures the algorithm's cost with a short
   probe and returns the ``max_iter`` at which its schedule completes exactly
   as the budget is spent.
"""

from typing import Dict, Sequence, Tuple, Type

import numpy as np


def _probe_cost(algo_class, pop_size, dim, iterations, seed):
    counter = {"n": 0}

    def probe(x):
        counter["n"] += 1
        return float(np.sum(np.asarray(x, dtype=float) ** 2))

    algo = algo_class(pop_size=pop_size, dim=dim, lb=-100.0, ub=100.0,
                      max_iter=iterations, obj_func=probe, seed=seed)
    _, _, conv = algo.optimize()
    return counter["n"], max(len(list(conv)) - 1, 1)


def measure_evals_per_iteration(algo_class: Type,
                                pop_size: int = 30,
                                dim: int = 10,
                                seed: int = 20260728) -> Tuple[float, float]:
    """
    Marginal cost of one iteration, and the algorithm's fixed start-up cost.

    Measured from two probes of different lengths and taking the slope, so the
    one-off cost of evaluating the initial population cancels out. A single
    short probe would fold that start-up cost into the per-iteration figure and
    overestimate it - by about 8% at twelve iterations, which is enough to leave
    a calibrated run short of its budget.

    Returns
    -------
    (evals_per_iteration, startup_evals)
    """
    n1, t1 = _probe_cost(algo_class, pop_size, dim, 8, seed)
    n2, t2 = _probe_cost(algo_class, pop_size, dim, 24, seed)
    if t2 <= t1:
        return float(n2) / max(t2, 1), 0.0
    per_iter = (n2 - n1) / (t2 - t1)
    startup = max(n1 - per_iter * t1, 0.0)
    return float(per_iter), float(startup)


def calibrate_iterations(algo_class: Type,
                         max_fes: int,
                         pop_size: int = 30,
                         dim: int = 10,
                         minimum: int = 2,
                         slack: float = 1.0,
                         seed: int = 20260728) -> Tuple[int, float]:
    """
    ``max_iter`` at which this algorithm's schedules finish as its budget ends.

    Parameters
    ----------
    algo_class : type
    max_fes : int
        Evaluation budget every algorithm must share.
    pop_size, dim : int
        Settings the real run will use; cost depends on both.
    minimum : int
        Floor on the returned iteration count.
    slack : float
        Multiplier on the calibrated iteration count. Algorithms whose cost per
        iteration varies with the landscape - BFO's swim loop, ABC's scouts -
        can finish short of their budget when calibrated exactly; a slack above
        1.0 lets them run on until the hard cap binds instead. The trade-off is
        that their annealing schedules are then cut off mid-decay, so the
        default is exact calibration and the shortfall is reported rather than
        papered over.

    Returns
    -------
    (max_iter, evals_per_iteration)

    Examples
    --------
    >>> from heurilab.algorithms import PSO, TSA
    >>> calibrate_iterations(PSO, 100_000, pop_size=50)[0]   # doctest: +SKIP
    1960
    >>> calibrate_iterations(TSA, 100_000, pop_size=50)[0]   # doctest: +SKIP
    662
    """
    per_iter, startup = measure_evals_per_iteration(
        algo_class, pop_size=pop_size, dim=dim, seed=seed)
    iterations = int(slack * (max_fes - startup) / max(per_iter, 1e-9))
    return max(iterations, minimum), per_iter


def calibrate_all(algorithms: Sequence,
                  max_fes: int,
                  pop_size: int = 30,
                  dim: int = 10,
                  seed: int = 20260728) -> Dict[str, Tuple[int, float]]:
    """Calibrate a list of ``(name, class)`` pairs. Returns ``{name: (max_iter, per_iter)}``."""
    return {name: calibrate_iterations(cls, max_fes, pop_size, dim, seed=seed)
            for name, cls in algorithms}


def budget_report(algorithms: Sequence,
                  max_fes: int,
                  pop_size: int = 30,
                  dim: int = 10,
                  seed: int = 20260728) -> str:
    """
    Markdown table of the per-algorithm cost and calibrated iteration count.

    Include this in a paper's experimental setup: it documents that every
    algorithm received the same number of evaluations, and makes the underlying
    cost differences visible instead of hiding them.
    """
    cal = calibrate_all(algorithms, max_fes, pop_size, dim, seed)
    per = np.array([v[1] for v in cal.values()], dtype=float)
    lines = [
        f"Evaluation budget: {max_fes:,} objective evaluations per run "
        f"(population {pop_size}).", "",
        "| algorithm | evaluations/iteration | calibrated max_iter | "
        "budget at equal iterations |", "|---|---|---|---|",
    ]
    baseline = float(np.median(per))
    for name, (iters, p) in sorted(cal.items(), key=lambda kv: -kv[1][1]):
        rel = p / baseline
        note = "fair" if 0.9 <= rel <= 1.1 else f"{rel:.2f}x the median"
        lines.append(f"| {name} | {p:.1f} | {iters} | {note} |")
    lines += ["", f"Cost spread across this set: "
                  f"{per.max() / max(per.min(), 1e-9):.1f}x. "
                  "Comparing at equal iterations would give the most expensive "
                  "algorithm that multiple of the cheapest one's search."]
    return "\n".join(lines)
