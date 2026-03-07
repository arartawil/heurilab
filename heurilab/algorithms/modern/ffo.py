"""FFO — Fennec Fox Optimization (Abdollahzadeh et al., 2024)"""

import numpy as np
from heurilab.algorithms.base import _Base


class FFO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            T_f = 1 - t / self.max_iter  # time factor
            a = 2 * T_f  # adaptive coefficient

            for i in range(self.pop_size):
                r = np.random.rand()

                if r < 0.5:
                    # Phase 1: Hearing-based hunting (exploration)
                    # Fennec fox uses large ears to detect underground prey
                    r1 = np.random.rand(self.dim)
                    j = np.random.randint(self.pop_size)

                    # Sound intensity decreases with distance
                    dist = np.abs(X[j] - X[i]) + 1e-16
                    hearing_range = a * np.exp(-dist)
                    new_X = X[i] + hearing_range * np.random.randn(self.dim) + r1 * (best - X[i]) * T_f

                else:
                    # Phase 2: Digging for prey (exploitation)
                    # Fennec fox digs toward food source (best)
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand()

                    if r2 < 0.5:
                        # Spiral digging pattern
                        D = np.abs(best - X[i])
                        l = np.random.uniform(-1, 1)
                        b = 1  # spiral shape constant
                        new_X = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best
                    else:
                        # Direct approach with thermal regulation
                        # Fennec fox adapts to temperature (iteration progress)
                        new_X = best + a * r1 * (best - X[i]) * (2 * np.random.rand() - 1)

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
