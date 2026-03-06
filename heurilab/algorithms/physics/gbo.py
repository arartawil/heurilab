"""GBO — Gradient-Based Optimizer (Ahmadianfar et al., 2020)"""

import numpy as np
from heurilab.algorithms.base import _Base


class GBO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        worst_idx = np.argmax(fitness)
        worst = X[worst_idx].copy()

        convergence = [best_fit]

        pr = 0.5  # probability parameter

        for t in range(self.max_iter):
            beta = 2 * (1 - ((t / self.max_iter) ** 3))  # decreasing coefficient
            alpha = abs(beta * np.sin(3 * np.pi / 2 + np.sin(beta * 3 * np.pi / 2)))

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2)
                eps = 5e-3 * t / self.max_iter

                # Gradient Search Rule (GSR)
                delta = 2 * r1 * (abs(np.random.rand() * best - X[i]) / (best - worst + 1e-16))
                step = (best - X[np.random.randint(self.pop_size)]) * r2

                # Direction of Movement (DM)
                GSR = alpha * delta * step

                # Local escaping operator
                if np.random.rand() < pr:
                    r3 = np.random.randint(self.pop_size)
                    r4 = np.random.randint(self.pop_size)
                    new_X = X[i] - GSR + np.random.rand() * (X[r3] - X[r4])
                else:
                    rnd = np.random.rand(self.dim)
                    new_X = best - GSR + rnd * (best - X[i])

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

            min_idx = np.argmin(fitness)
            max_idx = np.argmax(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]
            worst = X[max_idx].copy()

            convergence.append(best_fit)

        return best, best_fit, convergence
