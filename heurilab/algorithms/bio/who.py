"""WHO — Wildebeest Herd Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class WHO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                r = np.random.rand()

                if r < 0.3:
                    # Grazing: local search around current position
                    scale = (self.ub - self.lb) * 0.05 * (1 - t / self.max_iter)
                    new_X = X[i] + np.random.randn(self.dim) * scale
                elif r < 0.6:
                    # Migration toward best pasture (best solution)
                    phi = np.random.rand(self.dim)
                    new_X = X[i] + phi * (best - X[i])
                elif r < 0.9:
                    # Herd interaction: move toward random herd member
                    j = np.random.randint(self.pop_size)
                    while j == i:
                        j = np.random.randint(self.pop_size)
                    r2 = np.random.rand(self.dim)
                    if fitness[j] < fitness[i]:
                        new_X = X[i] + r2 * (X[j] - X[i])
                    else:
                        new_X = X[i] + r2 * (X[i] - X[j])
                else:
                    # Predator escape: random jump
                    new_X = np.random.uniform(self.lb, self.ub, self.dim)

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
