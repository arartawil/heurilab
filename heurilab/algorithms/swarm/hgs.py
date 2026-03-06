"""HGS — Hunger Games Search (Yang et al., 2021)"""

import numpy as np
from heurilab.algorithms.base import _Base


class HGS(_Base):
    def optimize(self):
        PUP = 0.08   # Percentage of updating positions
        LF = 1.0     # Levy flight parameter

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            worst_fit = np.max(fitness)
            best_curr = np.min(fitness)
            eps = 1e-16

            # Hunger value
            if worst_fit == best_curr:
                hunger = np.ones(self.pop_size)
            else:
                hunger = (fitness - best_curr) / (worst_fit - best_curr + eps)

            shrink = 2 * (1 - t / self.max_iter)  # Shrinking factor

            for i in range(self.pop_size):
                r1 = np.random.rand()
                r2 = np.random.rand()

                W1 = hunger[i] * np.random.rand(self.dim)

                if r1 < PUP:
                    # Random approach
                    new_X = X[i] * (1 + np.random.randn(self.dim))
                elif r2 > 0.5:
                    # Approach to best weighted by hunger
                    r3 = np.random.rand(self.dim)
                    A = shrink * (2 * r3 - 1)
                    new_X = best + A * np.abs(best - X[i]) * W1
                else:
                    # Social interaction
                    j = np.random.randint(self.pop_size)
                    r4 = np.random.rand(self.dim)
                    A = shrink * (2 * r4 - 1)
                    if fitness[j] < fitness[i]:
                        new_X = X[i] + A * np.abs(X[j] - X[i]) * W1
                    else:
                        new_X = X[i] - A * np.abs(X[j] - X[i]) * W1

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
