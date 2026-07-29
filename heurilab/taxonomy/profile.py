"""
Separating implementation cost from algorithmic cost.

Wall-clock time is the most misread number in a metaheuristic comparison. When
two algorithms differ in measured seconds, three quite different things could be
responsible:

1. **Evaluation count.** Algorithms consume different numbers of objective
   evaluations per iteration. At equal ``max_iter`` an algorithm with an inner
   swim or scout loop simply gets a bigger search budget.
2. **Implementation style.** A population update written as a Python ``for``
   loop over individuals is far slower than the same update written as one numpy
   array operation. This is a property of the code, not of the algorithm.
3. **Algorithmic overhead.** Genuine per-iteration work such as sorting,
   distance matrices or covariance updates.

Only the third is a property of the method. This module measures all three so
they can be told apart, and so a paper can report the fair currency - function
evaluations - alongside any timing it quotes.

Object orientation itself is not on that list, and deliberately so: attribute
lookup on ``self`` costs on the order of tens of nanoseconds, which is nothing
beside an objective evaluation. Subclassing ``_Base`` does not make an algorithm
slower. Looping over the population in Python does.
"""

import inspect
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Type

import numpy as np

from heurilab.taxonomy.detect import _class_source


# ── Objectives of contrasting cost ───────────────────────────────────

class _CountingObjective:
    """Near-free objective, so measured time is dominated by algorithm overhead."""

    def __init__(self):
        self.n = 0

    def __call__(self, x):
        self.n += 1
        return float(np.sum(np.asarray(x) ** 2))


class _ExpensiveObjective:
    """Deliberately costly objective, mimicking a real simulation-based problem."""

    def __init__(self, work: int = 220):
        self.n = 0
        self.work = work

    def __call__(self, x):
        self.n += 1
        arr = np.asarray(x, dtype=float)
        acc = 0.0
        for _ in range(self.work):
            acc += float(np.sum(np.sin(arr) ** 2 + np.cos(arr) ** 2))
        return float(np.sum(arr ** 2)) + 0.0 * acc


#: Python-level iteration over the population, the usual cause of slow updates.
_PER_INDIVIDUAL_LOOP = re.compile(
    r"for\s+\w+\s+in\s+range\s*\(\s*(?:self\.)?(?:pop_size|n_pop|N|self\.pop_size)",
    re.I)

#: Whole-population array operations.
_VECTORIZED_OP = re.compile(
    r"np\.(?:sum|mean|std|argsort|sort|clip|abs|exp|sqrt|where|maximum|minimum|"
    r"linalg|dot|outer|tile|repeat|einsum)\s*\(|"
    r"self\.rng\.\w+\s*\(\s*\(\s*self\.pop_size", re.I)


@dataclass
class AlgorithmProfile:
    """Cost breakdown for one algorithm."""
    name: str
    iterations: int
    evaluations: int
    evals_per_iteration: float
    seconds_cheap: float             # total time, near-free objective
    seconds_expensive: float         # total time, costly objective
    overhead_us_per_eval: float      # algorithm's own cost per evaluation
    objective_share: float           # fraction of time in the objective (costly case)
    population_loops: int            # Python loops over the population
    vectorized_ops: int              # whole-array operations
    vectorization_ratio: float

    def __repr__(self):
        return (f"AlgorithmProfile({self.name}, {self.evals_per_iteration:.1f} evals/iter, "
                f"{self.overhead_us_per_eval:.1f} us/eval overhead, "
                f"vectorization {self.vectorization_ratio:.2f})")


def profile_algorithm(name: str, algo_class: Type, dim: int = 20,
                      pop_size: int = 30, max_iter: int = 40,
                      seed: int = 20260728,
                      expensive: bool = True) -> AlgorithmProfile:
    """
    Measure where an algorithm's runtime actually goes.

    Parameters
    ----------
    name, algo_class
        The algorithm to profile.
    dim, pop_size, max_iter, seed : int
        Probe settings.
    expensive : bool
        Also time the algorithm against a costly objective, which is what shows
        how little implementation style matters on realistic problems. Set
        ``False`` to halve the profiling time.

    Returns
    -------
    AlgorithmProfile
    """
    cheap = _CountingObjective()
    t0 = time.perf_counter()
    algo = algo_class(pop_size=pop_size, dim=dim, lb=-100.0, ub=100.0,
                      max_iter=max_iter, obj_func=cheap, seed=seed)
    _, _, conv = algo.optimize()
    t_cheap = time.perf_counter() - t0
    n_evals = cheap.n
    iterations = max(len(list(conv)) - 1, 1)

    if expensive:
        costly = _ExpensiveObjective()
        t1 = time.perf_counter()
        algo2 = algo_class(pop_size=pop_size, dim=dim, lb=-100.0, ub=100.0,
                           max_iter=max_iter, obj_func=costly, seed=seed)
        algo2.optimize()
        t_expensive = time.perf_counter() - t1
    else:
        t_expensive = float("nan")

    # With a near-free objective, essentially all the time is the algorithm's own.
    overhead_us = (t_cheap / max(n_evals, 1)) * 1e6
    share = (1.0 - t_cheap / t_expensive) if expensive and t_expensive > 0 else float("nan")

    src = _class_source(algo_class)
    loops = len(_PER_INDIVIDUAL_LOOP.findall(src))
    vec = len(_VECTORIZED_OP.findall(src))
    ratio = vec / (vec + loops) if (vec + loops) else 0.0

    return AlgorithmProfile(
        name=name, iterations=iterations, evaluations=n_evals,
        evals_per_iteration=n_evals / iterations,
        seconds_cheap=t_cheap, seconds_expensive=t_expensive,
        overhead_us_per_eval=overhead_us, objective_share=share,
        population_loops=loops, vectorized_ops=vec, vectorization_ratio=ratio,
    )


def profile_many(algorithms: Sequence, **kw) -> List[AlgorithmProfile]:
    """Profile a list of ``(name, class)`` pairs."""
    return [profile_algorithm(nm, cls, **kw) for nm, cls in algorithms]


def profile_table(profiles: Sequence[AlgorithmProfile]) -> str:
    """Markdown table of a set of profiles, sorted by evaluation cost."""
    rows = sorted(profiles, key=lambda p: -p.evals_per_iteration)
    lines = [
        "| algorithm | evals/iter | total evals | overhead us/eval | "
        "pop loops | vector ops | vectorization | time share in objective |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in rows:
        share = "-" if np.isnan(p.objective_share) else f"{p.objective_share:.1%}"
        lines.append(
            f"| {p.name} | {p.evals_per_iteration:.1f} | {p.evaluations} | "
            f"{p.overhead_us_per_eval:.1f} | {p.population_loops} | "
            f"{p.vectorized_ops} | {p.vectorization_ratio:.2f} | {share} |")
    return "\n".join(lines)


def budget_fairness(profiles: Sequence[AlgorithmProfile]) -> Dict[str, float]:
    """
    How unfair an equal-``max_iter`` comparison is across these algorithms.

    Returns the minimum, maximum and ratio of evaluations consumed. A ratio well
    above 1 means the algorithms did not receive comparable search budgets, and
    any results table built at equal iterations is not a like-for-like
    comparison. The remedy is to terminate on function evaluations instead.
    """
    per_iter = np.array([p.evals_per_iteration for p in profiles], dtype=float)
    names = [p.name for p in profiles]
    lo, hi = int(np.argmin(per_iter)), int(np.argmax(per_iter))
    return {
        "min_evals_per_iter": float(per_iter[lo]),
        "min_algorithm": names[lo],
        "max_evals_per_iter": float(per_iter[hi]),
        "max_algorithm": names[hi],
        "unfairness_ratio": float(per_iter[hi] / max(per_iter[lo], 1e-12)),
    }
