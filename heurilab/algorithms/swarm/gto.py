"""GTO — Gorilla Troops Optimizer (Abdollahzadeh et al., 2021)"""

import numpy as np
from heurilab.algorithms.base import _Base


class GTO(_Base):
    def optimize(self):
        p = 0.03   # Probability of migration
        beta_gto = 3.0
        w = 0.8

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            a = (np.cos(2 * np.random.rand()) + 1) * (1 - t / self.max_iter)
            C = a * (2 * np.random.rand() - 1)  # Controlling parameter

            for i in range(self.pop_size):
                r1 = np.random.rand()

                if np.abs(C) >= 1:
                    # Exploration
                    if r1 < p:
                        # Migration to unknown place
                        new_X = (self.ub - self.lb) * np.random.rand(self.dim) + self.lb
                    else:
                        # Move toward another gorilla
                        j = np.random.randint(self.pop_size)
                        r2 = np.random.rand()
                        new_X = (r2 - a) * X[j] + (1 - r2 + a) * X[i]
                else:
                    # Exploitation
                    if r1 >= 0.5:
                        # Follow the silverback
                        A = beta_gto * np.random.randn(self.dim)
                        new_X = best - np.abs(best - X[i]) * A * np.sign(np.random.rand(self.dim) - 0.5)
                    else:
                        # Competition for females
                        L = C * np.random.randn(self.dim)
                        r3 = np.random.rand()
                        new_X = X[i] - L * (L * (X[i] - best) + r3 * (X[i] - best))

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
