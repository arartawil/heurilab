"""MGO — Mountain Gazelle Optimizer (Abdollahzadeh et al., 2022)"""

import numpy as np
from heurilab.algorithms.base import _Base


class MGO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            a = -(1 + t / self.max_iter)  # coefficient vector
            M = np.zeros((self.pop_size, self.dim))

            for i in range(self.pop_size):
                r1 = np.random.rand()
                r2 = np.random.rand()
                idx_rand = np.random.randint(self.pop_size)

                # Territory selection coefficients
                coef = np.random.randn(self.dim)

                if r1 < 0.5:
                    # Exploration: herd interaction
                    mu = np.mean(X, axis=0)
                    M[i] = (best - mu) * coef
                else:
                    # Individual gazelle territory
                    M[i] = (best - X[idx_rand]) * coef

                if r2 < 0.5:
                    # Exploitation: grazing behavior
                    D = np.abs(best - X[i])
                    A = 2 * a * np.random.rand(self.dim) - a
                    X[i] = best - D * A + M[i]
                else:
                    # Running from predator
                    r3 = np.random.rand(self.dim)
                    X[i] = (best - np.abs(best - X[i]) * np.cos(2 * np.pi * r3) *
                            (1 - t / self.max_iter)) + M[i]

                X[i] = self._clip(X[i])

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
