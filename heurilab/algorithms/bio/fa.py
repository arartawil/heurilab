"""FA — Firefly Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class FA(_Base):
    def optimize(self):
        alpha0 = 0.5
        beta0 = 1.0
        gamma = 1.0

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            alpha = alpha0 * (0.99 ** t)

            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        r = np.linalg.norm(X[i] - X[j])
                        beta = beta0 * np.exp(-gamma * r ** 2)
                        X[i] = X[i] + beta * (X[j] - X[i]) + alpha * (np.random.rand(self.dim) - 0.5)
                        X[i] = self._clip(X[i])
                        fitness[i] = self._eval(X[i])

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
