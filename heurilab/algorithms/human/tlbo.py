"""TLBO — Teaching-Learning-Based Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class TLBO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            mean_X = np.mean(X, axis=0)

            # ── Teacher Phase ──
            teacher = X[np.argmin(fitness)].copy()
            for i in range(self.pop_size):
                TF = np.random.randint(1, 3)  # Teaching factor: 1 or 2
                r = np.random.rand(self.dim)
                new_X = X[i] + r * (teacher - TF * mean_X)
                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)
                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

            # ── Learner Phase ──
            for i in range(self.pop_size):
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                r = np.random.rand(self.dim)
                if fitness[i] < fitness[j]:
                    new_X = X[i] + r * (X[i] - X[j])
                else:
                    new_X = X[i] + r * (X[j] - X[i])

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
