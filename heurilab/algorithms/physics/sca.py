"""SCA — Sine Cosine Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class SCA(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            a = 2
            r1 = a - a * t / self.max_iter

            for i in range(self.pop_size):
                r2 = 2 * np.pi * np.random.rand(self.dim)
                r3 = 2 * np.random.rand(self.dim)
                r4 = np.random.rand(self.dim)

                cond = r4 < 0.5
                X[i] = np.where(cond,
                                X[i] + r1 * np.sin(r2) * np.abs(r3 * best - X[i]),
                                X[i] + r1 * np.cos(r2) * np.abs(r3 * best - X[i]))
                X[i] = self._clip(X[i])

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
