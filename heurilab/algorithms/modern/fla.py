"""FLA — Fick's Law Algorithm (Hashim et al., 2023)"""

import numpy as np
from heurilab.algorithms.base import _Base


class FLA(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        C1, C2 = 0.5, 2.0  # diffusion coefficients
        n1 = self.pop_size // 2

        for t in range(self.max_iter):
            D = C1 * np.exp(-C2 * t / self.max_iter)  # diffusion factor
            TF = np.exp(-t / self.max_iter)  # transfer factor

            sorted_idx = np.argsort(fitness)

            for i in range(self.pop_size):
                if i < n1:
                    # First half: Fick's first law (exploitation)
                    # Diffusion from high to low concentration (toward best)
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand()
                    j = sorted_idx[np.random.randint(max(1, n1 // 2))]  # random good solution
                    grad = X[j] - X[i]
                    new_X = X[i] + D * r1 * grad + r2 * TF * (best - X[i])
                else:
                    # Second half: Fick's second law (exploration)
                    # Concentration change over time
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    j = np.random.randint(self.pop_size)
                    k = np.random.randint(self.pop_size)
                    new_X = X[i] + D * (r1 * (X[j] - X[k]) + r2 * (best - X[i])) * TF

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
