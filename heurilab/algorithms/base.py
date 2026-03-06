"""
Shared base class for all metaheuristic algorithms.
"""

import numpy as np


class _Base:
    """Common interface for all optimizers."""

    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        self.pop_size = pop_size
        self.dim = dim
        self.lb = np.broadcast_to(np.asarray(lb, dtype=float), dim).copy()
        self.ub = np.broadcast_to(np.asarray(ub, dtype=float), dim).copy()
        self.max_iter = max_iter
        self.obj_func = obj_func

    def _init_pop(self):
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

    def _clip(self, x):
        return np.clip(x, self.lb, self.ub)

    def _eval(self, x):
        return self.obj_func(x)
