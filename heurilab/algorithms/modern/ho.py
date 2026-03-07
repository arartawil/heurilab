"""HO — Hippopotamus Optimization Algorithm (Amiri et al., 2024)"""

import numpy as np
from heurilab.algorithms.base import _Base


class HO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            T = 1 - t / self.max_iter  # time factor

            for i in range(self.pop_size):
                r = np.random.rand()

                if r < 0.33:
                    # Phase 1: Position update in river/pond (exploitation)
                    # Hippo moves in water near the dominant hippo
                    r1 = np.random.rand(self.dim)
                    I = np.random.choice([1, 2])
                    new_X = X[i] + r1 * (best - I * X[i])

                elif r < 0.66:
                    # Phase 2: Defense against predators (exploration)
                    # Hippo defends territory aggressively
                    j = np.random.randint(self.pop_size)
                    r1 = np.random.rand(self.dim)

                    if fitness[j] < fitness[i]:
                        new_X = X[i] + r1 * (X[j] - X[i]) * T
                    else:
                        new_X = X[i] + r1 * (X[i] - X[j]) * T

                else:
                    # Phase 3: Escaping from predator (diversification)
                    # Hippo runs to random safe location
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    A = 2 * r1 * T - T  # adaptive coefficient
                    mu = np.mean(X, axis=0)
                    new_X = best + A * (best - mu) + r2 * np.random.randn(self.dim) * T

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
