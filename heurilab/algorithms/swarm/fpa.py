"""FPA — Flower Pollination Algorithm"""

import math
import numpy as np
from heurilab.algorithms.base import _Base


class FPA(_Base):
    def optimize(self):
        p = 0.8   # Switch probability
        beta = 1.5

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        sigma_u = (
            math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                if self.rng.random() < p:
                    # Global pollination via Lévy flights
                    u = self.rng.standard_normal(self.dim) * sigma_u
                    v = self.rng.standard_normal(self.dim)
                    L = u / (np.abs(v) ** (1 / beta))
                    new_X = X[i] + L * (best - X[i])
                else:
                    # Local pollination
                    j, k = self.rng.integers(0, self.pop_size, 2)
                    eps = self.rng.random()
                    new_X = X[i] + eps * (X[j] - X[k])

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
