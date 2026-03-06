"""EP — Evolutionary Programming"""

import numpy as np
from heurilab.algorithms.base import _Base


class EP(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            # Mutation using Gaussian with adaptive step
            sigma = np.abs(X) * (1 - t / self.max_iter) + 1e-8
            offspring = self._clip(X + sigma * np.random.randn(self.pop_size, self.dim))
            offspring_fit = np.array([self._eval(offspring[i]) for i in range(self.pop_size)])

            # Tournament selection from combined pool
            combined_X = np.vstack([X, offspring])
            combined_fit = np.hstack([fitness, offspring_fit])

            # q-tournament: count wins
            q = 10
            wins = np.zeros(2 * self.pop_size, dtype=int)
            for i in range(2 * self.pop_size):
                opponents = np.random.randint(0, 2 * self.pop_size, q)
                wins[i] = np.sum(combined_fit[i] <= combined_fit[opponents])

            sorted_idx = np.argsort(-wins)[:self.pop_size]
            X = combined_X[sorted_idx]
            fitness = combined_fit[sorted_idx]

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
