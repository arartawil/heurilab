"""BOA — Butterfly Optimization Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class BOA(_Base):
    def optimize(self):
        a = 0.1      # Sensory modality
        c_val = 0.01 # Power exponent
        p = 0.8      # Switch probability

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            # Fragrance
            f_max = np.max(fitness)
            f_min = np.min(fitness)
            eps = 1e-16

            for i in range(self.pop_size):
                # Stimulus intensity
                I = fitness[i]
                fragrance = c_val * (I ** a)

                r = np.random.rand()
                if r < p:
                    # Global search: move toward best
                    new_X = X[i] + (np.random.rand(self.dim) ** 2) * fragrance * (best - X[i])
                else:
                    # Local search: move toward random neighbour
                    j, k = np.random.randint(0, self.pop_size, 2)
                    new_X = X[i] + (np.random.rand(self.dim) ** 2) * fragrance * (X[j] - X[k])

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

            # Update sensory modality
            a = a + 0.025 / (a * self.max_iter + eps)

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
