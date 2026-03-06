"""MPA — Marine Predators Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class MPA(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]
        FADs = 0.2
        P = 0.5

        for t in range(self.max_iter):
            CF = (1 - t / self.max_iter) ** (2 * t / self.max_iter)

            for i in range(self.pop_size):
                R = np.random.rand(self.dim)

                if t < self.max_iter / 3:
                    # Phase 1: High velocity ratio — Brownian
                    stepsize = R * (best - R * X[i])
                    X[i] = X[i] + P * R * stepsize

                elif t < 2 * self.max_iter / 3:
                    # Phase 2: Unit velocity ratio
                    if i < self.pop_size / 2:
                        stepsize = R * (best - R * X[i])
                        X[i] = X[i] + P * R * stepsize
                    else:
                        stepsize = R * (R * best - X[i])
                        X[i] = best + P * CF * stepsize

                else:
                    # Phase 3: Low velocity ratio — Lévy
                    stepsize = R * (R * best - X[i])
                    X[i] = best + P * CF * stepsize

                X[i] = self._clip(X[i])

            # FADs effect
            if np.random.rand() < FADs:
                for i in range(self.pop_size):
                    if np.random.rand() < FADs:
                        u = np.random.rand(self.dim) < FADs
                        X[i] = X[i] + CF * (self.lb + np.random.rand(self.dim) * (self.ub - self.lb)) * u.astype(float)
                        X[i] = self._clip(X[i])

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
