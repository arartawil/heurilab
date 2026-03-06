"""DO — Dolphin Optimizer"""

import numpy as np
from heurilab.algorithms.base import _Base


class DO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            a = 2.0 * (1 - t / self.max_iter)  # Linearly decreasing

            for i in range(self.pop_size):
                r1 = np.random.rand()
                r2 = np.random.rand(self.dim)

                if r1 < 0.5:
                    # Echolocation: spiral movement toward prey (best)
                    l = np.random.uniform(-1, 1)
                    b = 1.0  # Spiral shape constant
                    D = np.abs(best - X[i])
                    new_X = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best
                else:
                    # Schooling: swim toward best with random perturbation
                    A = 2 * a * np.random.rand(self.dim) - a
                    C = 2 * r2
                    D = np.abs(C * best - X[i])
                    new_X = best - A * D

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

                if new_fit < best_fit:
                    best = new_X.copy()
                    best_fit = new_fit

            convergence.append(best_fit)

        return best, best_fit, convergence
