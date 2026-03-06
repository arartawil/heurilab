"""SMA — Slime Mould Algorithm (Li et al., 2020)"""

import numpy as np
from heurilab.algorithms.base import _Base


class SMA(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]
        lb = np.array(self.lb)
        ub = np.array(self.ub)

        z = 0.03  # probability threshold

        for t in range(self.max_iter):
            a = np.arctanh(1 - (t + 1) / self.max_iter)
            b = 1 - (t + 1) / self.max_iter

            # Sort by fitness and compute weights
            sorted_idx = np.argsort(fitness)
            W = np.zeros(self.pop_size)
            half = self.pop_size // 2
            worst_fit = fitness[sorted_idx[-1]]
            for i, si in enumerate(sorted_idx):
                if i < half:
                    W[si] = 1 + np.random.rand() * np.log10((best_fit - fitness[si]) / (worst_fit - best_fit + 1e-16) + 1)
                else:
                    W[si] = 1 - np.random.rand() * np.log10((best_fit - fitness[si]) / (worst_fit - best_fit + 1e-16) + 1)

            for i in range(self.pop_size):
                if np.random.rand() < z:
                    # Random exploration
                    new_X = lb + np.random.rand(self.dim) * (ub - lb)
                else:
                    p = np.tanh(abs(fitness[i] - best_fit))
                    vb = a * (2 * np.random.rand(self.dim) - 1)
                    vc = b * (2 * np.random.rand(self.dim) - 1)

                    r = np.random.rand()
                    rA, rB = np.random.randint(self.pop_size, size=2)

                    if r < p:
                        new_X = best + vb * (W[i] * X[rA] - X[rB])
                    else:
                        new_X = vc * X[i]

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
