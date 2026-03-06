"""AEO — Artificial Ecosystem-based Optimization (Zhao et al., 2019)"""

import numpy as np
from heurilab.algorithms.base import _Base


class AEO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]
        lb = np.array(self.lb)
        ub = np.array(self.ub)

        for t in range(self.max_iter):
            r = (1 - t / self.max_iter)

            # Production (best individual = producer)
            # Herbivore/Omnivore/Carnivore operators
            for i in range(self.pop_size):
                rand = np.random.rand()

                if rand < 1 / 3:
                    # Production operator (energy from environment)
                    x1 = (1 - r) * lb + r * np.random.rand(self.dim) * (ub - lb)
                    new_X = best * (1 - r) + x1 * r
                elif rand < 2 / 3:
                    # Consumption operator: herbivore
                    j = np.random.randint(self.pop_size)
                    C = np.random.rand() * r
                    if fitness[i] < fitness[j]:
                        new_X = X[i] + C * (X[i] - X[j])
                    else:
                        new_X = X[i] + C * (X[j] - X[i])
                else:
                    # Consumption operator: carnivore
                    j = np.random.randint(self.pop_size)
                    C = np.random.rand() * r
                    new_X = X[i] + C * (best - X[j]) * r

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
