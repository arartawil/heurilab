"""
Shared base class for all metaheuristic algorithms.

Besides holding the common state, ``_Base`` enforces the *function-evaluation
budget*, which is what makes a comparison between algorithms fair.

Algorithms consume very different numbers of objective evaluations per
iteration: at ``pop_size=30`` a plain swarm method spends about 30, harmony
search spends 1, and a method with an inner loop over pairs spends several
hundred. Comparing them at equal ``max_iter`` therefore hands some algorithms
many times the search budget of others. Measured across HeuriLab's registry at
300 iterations, that spread reaches 396x (330 evaluations to 130,526).

Setting ``max_fes`` caps the evaluations instead. When the cap is reached, the
run stops immediately wherever it is and returns the best solution found so far,
so every algorithm is compared on exactly the same amount of search.
"""

import numpy as np


class BudgetExhausted(BaseException):
    """Raised internally when the evaluation budget runs out.

    Inherits from :class:`BaseException` rather than :class:`Exception` so that
    an algorithm containing a broad ``except Exception`` cannot swallow it and
    keep searching past its budget.
    """


def roulette_probabilities(weights):
    """Normalise non-negative ``weights`` into a probability vector.

    ``numpy.random.Generator.choice`` rejects a ``p`` whose sum differs from 1
    by more than a tight tolerance. The idiom ``w / (w.sum() + eps)``, which
    appears in several published pseudo-codes, therefore fails outright once
    the population converges and every weight collapses to zero: the epsilon
    that was meant to prevent a division by zero leaves the total short of 1.
    A degenerate or non-finite weight vector is treated as uniform, which is
    the correct reading of "no candidate is preferred".
    """
    w = np.asarray(weights, dtype=float).ravel()
    w = np.where(np.isfinite(w), w, 0.0)
    np.clip(w, 0.0, None, out=w)
    total = w.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.full(w.size, 1.0 / w.size)
    w /= total
    # Absorb the rounding residual into the largest entry, which keeps the sum
    # within ``choice``'s tolerance even for weight vectors spanning many
    # orders of magnitude.
    w[w.argmax()] += 1.0 - w.sum()
    return w


def _guard_budget(optimize):
    """Wrap an ``optimize`` method so budget exhaustion returns cleanly."""
    def wrapper(self, *args, **kwargs):
        try:
            return optimize(self, *args, **kwargs)
        except BudgetExhausted:
            return self._budget_result()
    wrapper.__name__ = getattr(optimize, "__name__", "optimize")
    wrapper.__doc__ = optimize.__doc__
    wrapper.__wrapped__ = optimize
    wrapper._heurilab_budget_guarded = True
    return wrapper


class _Base:
    """Common interface for all optimizers.

    Parameters
    ----------
    pop_size : int
        Population size.
    dim : int
        Problem dimensionality.
    lb, ub : float or array-like
        Lower/upper bounds, broadcast to ``dim``.
    max_iter : int
        Iteration budget.
    obj_func : callable
        Objective function; minimised.
    seed : int, np.random.SeedSequence, np.random.Generator or None, optional
        Controls the random stream used by the algorithm. ``None`` (the
        default) draws fresh entropy from the OS, reproducing the historical
        non-deterministic behaviour. Pass an integer for a reproducible run.
    max_fes : int, optional
        Hard cap on objective-function evaluations. ``None`` (the default)
        means the run is bounded only by ``max_iter``, which is *not* a fair
        basis for comparing algorithms with different per-iteration costs. See
        :func:`heurilab.core.budget.calibrate_iterations` for choosing a
        matching ``max_iter`` so that decay schedules still complete.

    Attributes
    ----------
    n_fes : int
        Objective evaluations consumed by the last run.
    best_solution, best_fitness
        Best point seen, tracked independently of the algorithm's own
        bookkeeping so a budget-terminated run still returns a valid answer.

    Notes
    -----
    Subclasses must draw all randomness from ``self.rng`` (a
    ``numpy.random.Generator``) rather than the global ``numpy.random``
    module, otherwise runs cannot be reproduced.
    """

    def __init_subclass__(cls, **kwargs):
        """Wrap every subclass's ``optimize`` so the budget can stop it."""
        super().__init_subclass__(**kwargs)
        own = cls.__dict__.get("optimize")
        if own is not None and not getattr(own, "_heurilab_budget_guarded", False):
            cls.optimize = _guard_budget(own)

    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func,
                 seed=None, max_fes=None):
        self.pop_size = pop_size
        self.dim = dim
        self.lb = np.broadcast_to(np.asarray(lb, dtype=float), dim).copy()
        self.ub = np.broadcast_to(np.asarray(ub, dtype=float), dim).copy()
        self.max_iter = max_iter
        self.obj_func = obj_func
        self.seed = seed
        self.rng = seed if isinstance(seed, np.random.Generator) \
            else np.random.default_rng(seed)
        self.max_fes = max_fes

        self.n_fes = 0
        self.best_solution = None
        self.best_fitness = np.inf
        self._fes_trace = []          # best-so-far after each evaluation
        self._progress_callback = None

    # ── search-space helpers ─────────────────────────────────────────

    def _init_pop(self):
        return self.rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))

    def _clip(self, x):
        return np.clip(x, self.lb, self.ub)

    # ── evaluation, budget accounting and best-so-far tracking ───────

    def _eval(self, x):
        result = self.obj_func(x)
        self.n_fes += 1

        value = float(result)
        if value < self.best_fitness:
            self.best_fitness = value
            self.best_solution = np.array(x, dtype=float, copy=True).ravel()
        self._fes_trace.append(self.best_fitness)

        if self._progress_callback is not None:
            self._progress_callback(result)

        if self.max_fes is not None and self.n_fes >= self.max_fes:
            raise BudgetExhausted
        return result

    # ── budget-terminated return value ───────────────────────────────

    def _budget_result(self):
        """Best-so-far result when a run is cut short by the evaluation cap.

        The convergence trace is resampled from evaluation space onto
        ``max_iter + 1`` points so it lines up with the traces of runs that
        finished normally, and can be averaged and plotted alongside them.
        """
        trace = self._fes_trace or [self.best_fitness]
        length = max(int(self.max_iter) + 1, 2)
        idx = np.linspace(0, len(trace) - 1, num=length)
        convergence = list(np.asarray(trace, dtype=float)[np.round(idx).astype(int)])

        solution = (self.best_solution if self.best_solution is not None
                    else self._clip(np.zeros(self.dim)))
        return solution, self.best_fitness, convergence
