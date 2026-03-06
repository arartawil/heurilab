"""BA — Bat Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class BA(_Base):
    def optimize(self):
        f_min, f_max = 0.0, 2.0
        A = np.ones(self.pop_size)       # Loudness
        r0 = np.full(self.pop_size, 0.5) # Initial pulse rate
        r = r0.copy()
        alpha, gamma = 0.9, 0.9

        X = self._init_pop()
        V = np.zeros((self.pop_size, self.dim))
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                freq = f_min + (f_max - f_min) * np.random.rand()
                V[i] += (X[i] - best) * freq
                new_X = self._clip(X[i] + V[i])

                # Local search
                if np.random.rand() > r[i]:
                    new_X = self._clip(best + 0.01 * np.random.randn(self.dim) * np.mean(A))

                new_fit = self._eval(new_X)

                if new_fit < fitness[i] and np.random.rand() < A[i]:
                    X[i] = new_X
                    fitness[i] = new_fit
                    A[i] *= alpha
                    r[i] = r0[i] * (1 - np.exp(-gamma * (t + 1)))

                if new_fit < best_fit:
                    best = new_X.copy()
                    best_fit = new_fit

            convergence.append(best_fit)

        return best, best_fit, convergence
