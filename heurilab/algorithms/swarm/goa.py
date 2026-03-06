"""GOA — Grasshopper Optimization Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class GOA(_Base):
    def optimize(self):
        c_max, c_min = 1.0, 0.00004

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            c = c_max - t * (c_max - c_min) / self.max_iter

            new_X = np.zeros_like(X)
            for i in range(self.pop_size):
                S = np.zeros(self.dim)
                for j in range(self.pop_size):
                    if i == j:
                        continue
                    dist = np.abs(X[j] - X[i])
                    r = np.linalg.norm(X[j] - X[i]) + 1e-16
                    direction = (X[j] - X[i]) / r
                    # Social force: s(r) = f * exp(-r/l) - exp(-r)
                    s_val = 0.5 * np.exp(-dist / 1.5) - np.exp(-dist)
                    S += c * s_val * direction

                new_X[i] = self._clip(c * S + best)

            X = new_X
            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
