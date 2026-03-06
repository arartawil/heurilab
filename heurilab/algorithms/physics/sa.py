"""SA — Simulated Annealing"""

import numpy as np
from heurilab.algorithms.base import _Base


class SA(_Base):
    def optimize(self):
        T0 = 100.0
        T_min = 1e-10
        alpha = 0.99

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            T = max(T0 * (alpha ** t), T_min)

            for i in range(self.pop_size):
                # Generate neighbour
                new_X = self._clip(X[i] + np.random.randn(self.dim) * (self.ub - self.lb) * T / T0)
                new_fit = self._eval(new_X)

                delta = new_fit - fitness[i]
                if delta < 0 or np.random.rand() < np.exp(-delta / (T + 1e-16)):
                    X[i] = new_X
                    fitness[i] = new_fit

                if fitness[i] < best_fit:
                    best = X[i].copy()
                    best_fit = fitness[i]

            convergence.append(best_fit)

        return best, best_fit, convergence
