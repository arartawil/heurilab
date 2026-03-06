"""HBA — Honey Badger Algorithm (Hashim et al., 2022)"""

import numpy as np
from heurilab.algorithms.base import _Base


class HBA(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]
        lb = np.array(self.lb)
        ub = np.array(self.ub)

        C = 2  # constant

        for t in range(self.max_iter):
            alpha = C * np.exp(-t / self.max_iter)  # decreasing factor

            for i in range(self.pop_size):
                r = np.random.rand()
                F_flag = 1 if np.random.rand() < 0.5 else -1

                # Intensity
                di = np.linalg.norm(X[i] - best) + 1e-16
                S_i = (X[i] - best) / (di + 1e-16)

                if r < 0.5:
                    # Digging phase (exploitation)
                    r3 = np.random.rand()
                    r4 = np.random.rand()
                    r5 = np.random.rand()
                    r6 = np.random.rand()
                    r7 = np.random.rand()

                    new_X = best + F_flag * C * alpha * S_i * (r3 * best - r4 * X[i])
                else:
                    # Honey phase (exploration)
                    r3 = np.random.rand()
                    r4 = np.random.rand()
                    r5 = np.random.rand()

                    new_X = best + F_flag * alpha * r3 * S_i * np.abs(np.cos(2 * np.pi * r4) * (1 - np.cos(2 * np.pi * r5)))

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
