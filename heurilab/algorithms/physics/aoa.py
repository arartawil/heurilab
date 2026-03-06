"""AOA — Arithmetic Optimization Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class AOA(_Base):
    def optimize(self):
        MOP_min, MOP_max = 0.2, 1.0
        alpha_aoa, mu_aoa = 5, 0.499
        eps = 1e-16

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            MOP = 1 - ((t + 1) ** (1 / alpha_aoa)) / (self.max_iter ** (1 / alpha_aoa))
            MOA = MOP_min + (t + 1) * ((MOP_max - MOP_min) / self.max_iter)

            for i in range(self.pop_size):
                for j in range(self.dim):
                    r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                    if r1 > MOA:
                        # Exploration
                        if r2 > 0.5:
                            X[i, j] = best[j] / (MOP + eps) * ((self.ub[j] - self.lb[j]) * mu_aoa + self.lb[j])
                        else:
                            X[i, j] = best[j] * MOP * ((self.ub[j] - self.lb[j]) * mu_aoa + self.lb[j])
                    else:
                        # Exploitation
                        if r3 > 0.5:
                            X[i, j] = best[j] - MOP * ((self.ub[j] - self.lb[j]) * mu_aoa + self.lb[j])
                        else:
                            X[i, j] = best[j] + MOP * ((self.ub[j] - self.lb[j]) * mu_aoa + self.lb[j])

                X[i] = self._clip(X[i])

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
