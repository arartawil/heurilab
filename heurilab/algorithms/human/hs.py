"""HS — Harmony Search"""

import numpy as np
from heurilab.algorithms.base import _Base


class HS(_Base):
    def optimize(self):
        HMCR = 0.9   # Harmony memory considering rate
        PAR = 0.3    # Pitch adjusting rate
        bw = 0.01    # Bandwidth

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            # Dynamic bandwidth
            bw_t = bw * np.exp(-5.0 * t / self.max_iter)

            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < HMCR:
                    # Memory consideration
                    idx = np.random.randint(self.pop_size)
                    new_harmony[j] = X[idx, j]

                    # Pitch adjustment
                    if np.random.rand() < PAR:
                        new_harmony[j] += bw_t * (self.ub[j] - self.lb[j]) * (2 * np.random.rand() - 1)
                else:
                    # Random selection
                    new_harmony[j] = np.random.uniform(self.lb[j], self.ub[j])

            new_harmony = self._clip(new_harmony)
            new_fit = self._eval(new_harmony)

            worst_idx = np.argmax(fitness)
            if new_fit < fitness[worst_idx]:
                X[worst_idx] = new_harmony
                fitness[worst_idx] = new_fit

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
