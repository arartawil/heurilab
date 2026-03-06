"""SSA — Salp Swarm Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class SSA(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            c1 = 2 * np.exp(-((4 * t / self.max_iter) ** 2))

            for i in range(self.pop_size):
                if i == 0:  # Leader
                    for j in range(self.dim):
                        c2, c3 = np.random.rand(), np.random.rand()
                        if c3 < 0.5:
                            X[i, j] = best[j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])
                        else:
                            X[i, j] = best[j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])
                else:  # Follower
                    X[i] = 0.5 * (X[i] + X[i - 1])

                X[i] = self._clip(X[i])

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
