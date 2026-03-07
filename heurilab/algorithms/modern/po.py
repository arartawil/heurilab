"""PO — Parrot Optimizer (Lian et al., 2024)"""

import math
import numpy as np
from heurilab.algorithms.base import _Base


class PO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            alpha = (1 - t / self.max_iter)  # decreasing factor
            n1 = self.pop_size // 3
            n2 = 2 * self.pop_size // 3

            for i in range(self.pop_size):
                if i < n1:
                    # Foraging behavior (exploration)
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand()
                    j = np.random.randint(self.pop_size)
                    new_X = X[i] + alpha * r1 * (X[j] - X[i]) + r2 * (best - X[i])

                elif i < n2:
                    # Staying behavior (exploitation near best)
                    r = np.random.rand(self.dim)
                    Levy = self._levy_flight()
                    new_X = best + alpha * Levy * (best - X[i]) + r * (1 - alpha) * np.random.randn(self.dim)

                else:
                    # Communication behavior (social learning)
                    j = np.random.randint(self.pop_size)
                    k = np.random.randint(self.pop_size)
                    r = np.random.rand(self.dim)

                    if fitness[j] < fitness[i]:
                        new_X = X[i] + r * (X[j] - X[k])
                    else:
                        # Flee with random displacement
                        new_X = X[i] + alpha * np.random.randn(self.dim) * (self.ub - self.lb) * 0.01

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence

    def _levy_flight(self):
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        return u / (np.abs(v) ** (1 / beta))
