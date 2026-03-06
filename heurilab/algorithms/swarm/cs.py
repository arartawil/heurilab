"""CS — Cuckoo Search"""

import numpy as np
from heurilab.algorithms.base import _Base


class CS(_Base):
    def optimize(self):
        pa = 0.25  # Discovery rate
        beta = 1.5  # Lévy exponent

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        # Precompute Lévy constants
        sigma_u = (
            np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
            / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                u = np.random.randn(self.dim) * sigma_u
                v = np.random.randn(self.dim)
                step = u / (np.abs(v) ** (1 / beta))

                new_X = self._clip(X[i] + 0.01 * step * (X[i] - best))
                new_fit = self._eval(new_X)

                j = np.random.randint(self.pop_size)
                if new_fit < fitness[j]:
                    X[j] = new_X
                    fitness[j] = new_fit

            # Abandon worst nests
            n_abandon = int(pa * self.pop_size)
            worst_idx = np.argsort(fitness)[-n_abandon:]
            for idx in worst_idx:
                r1, r2 = np.random.randint(0, self.pop_size, 2)
                step_size = np.random.rand() * (X[r1] - X[r2])
                X[idx] = self._clip(X[idx] + step_size)
                fitness[idx] = self._eval(X[idx])

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
