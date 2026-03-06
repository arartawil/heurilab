"""WOA — Whale Optimization Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class WOA(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            a = 2 - 2 * t / self.max_iter
            b = 1
            l_param = np.random.uniform(-1, 1, self.pop_size)

            for i in range(self.pop_size):
                r = np.random.rand()
                A = 2 * a * np.random.rand(self.dim) - a
                C = 2 * np.random.rand(self.dim)

                if r < 0.5:
                    if np.abs(A).mean() < 1:
                        D = np.abs(C * best - X[i])
                        X[i] = best - A * D
                    else:
                        rand_idx = np.random.randint(self.pop_size)
                        D = np.abs(C * X[rand_idx] - X[i])
                        X[i] = X[rand_idx] - A * D
                else:
                    D = np.abs(best - X[i])
                    X[i] = D * np.exp(b * l_param[i]) * np.cos(2 * np.pi * l_param[i]) + best

                X[i] = self._clip(X[i])

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
