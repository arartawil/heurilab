"""NOA — Nutcracker Optimization Algorithm (Abdel-Basset et al., 2023)"""

import math
import numpy as np
from heurilab.algorithms.base import _Base


class NOA(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]
        second_best = X[np.argsort(fitness)[1]].copy()

        convergence = [best_fit]

        # Cache memory for stored food
        cache = X.copy()
        cache_fit = fitness.copy()

        for t in range(self.max_iter):
            Pa = 0.2 + 0.6 * (t / self.max_iter)  # caching probability increases

            for i in range(self.pop_size):
                r = np.random.rand()

                if r < Pa:
                    # Phase 1: Food storage/caching (exploration)
                    r1 = np.random.randint(self.pop_size)
                    r2 = np.random.randint(self.pop_size)
                    mu = (X[r1] + X[r2]) / 2
                    new_X = mu + np.random.randn(self.dim) * np.abs(X[r1] - X[r2]) * (1 - t / self.max_iter)
                else:
                    # Phase 2: Food recovery (exploitation)
                    if np.random.rand() < 0.5:
                        # Recover from cache near best
                        Levy = self._levy_flight()
                        new_X = best + Levy * (cache[i] - best) * (1 - t / self.max_iter)
                    else:
                        # Guided by best and second best
                        r1 = np.random.rand(self.dim)
                        new_X = X[i] + r1 * (best - X[i]) + (1 - r1) * (second_best - X[i])

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit < fitness[i]:
                    # Update cache with old position
                    cache[i] = X[i].copy()
                    cache_fit[i] = fitness[i]
                    X[i] = new_X
                    fitness[i] = new_fit

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            sorted_idx = np.argsort(fitness)
            second_best = X[sorted_idx[1]].copy()

            convergence.append(best_fit)

        return best, best_fit, convergence

    def _levy_flight(self):
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        return u / (np.abs(v) ** (1 / beta))
