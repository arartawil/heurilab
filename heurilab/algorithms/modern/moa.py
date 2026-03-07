"""MOA — Mother Optimization Algorithm (Matoušová et al., 2023)"""

import numpy as np
from heurilab.algorithms.base import _Base


class MOA(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            alpha = 1 - t / self.max_iter  # nurturing factor
            beta = 0.5 * (1 + t / self.max_iter)  # independence factor

            for i in range(self.pop_size):
                r = np.random.rand()

                if r < 0.33:
                    # Phase 1: Nurturing — mother guides child toward best
                    r1 = np.random.rand(self.dim)
                    new_X = X[i] + alpha * r1 * (best - X[i])
                elif r < 0.66:
                    # Phase 2: Education — learn from random members
                    j = np.random.randint(self.pop_size)
                    k = np.random.randint(self.pop_size)
                    r1 = np.random.rand(self.dim)
                    if fitness[j] < fitness[k]:
                        new_X = X[i] + r1 * (X[j] - X[k]) * alpha
                    else:
                        new_X = X[i] + r1 * (X[k] - X[j]) * alpha
                else:
                    # Phase 3: Independence — child explores on its own
                    r1 = np.random.rand(self.dim)
                    sigma = (self.ub - self.lb) * (1 - t / self.max_iter) * 0.1
                    new_X = X[i] + sigma * np.random.randn(self.dim) * beta
                    # With some probability move toward best
                    if np.random.rand() < 0.5:
                        new_X = new_X + r1 * (best - new_X) * (1 - alpha)

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
