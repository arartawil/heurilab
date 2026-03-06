"""SOS_H — Social Optimization Search (Human-based)"""

import numpy as np
from heurilab.algorithms.base import _Base


class SOS_H(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                # ── Imitation phase ──
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                r = np.random.rand(self.dim)
                if fitness[j] < fitness[i]:
                    new_X = X[i] + r * (X[j] - X[i])
                else:
                    new_X = X[i] + r * (X[i] - X[j])

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)
                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

                # ── Conversation phase ──
                j = np.random.randint(self.pop_size)
                k = np.random.randint(self.pop_size)
                while k == j:
                    k = np.random.randint(self.pop_size)

                mean_jk = (X[j] + X[k]) / 2.0
                r2 = np.random.rand(self.dim)
                new_X2 = X[i] + r2 * (mean_jk - X[i])
                new_X2 = self._clip(new_X2)
                new_fit2 = self._eval(new_X2)
                if new_fit2 < fitness[i]:
                    X[i] = new_X2
                    fitness[i] = new_fit2

                # ── Innovation phase ──
                r3 = np.random.rand(self.dim)
                new_X3 = X[i] + (2 * r3 - 1) * (self.ub - self.lb) * 0.01 * (1 - t / self.max_iter)
                new_X3 = self._clip(new_X3)
                new_fit3 = self._eval(new_X3)
                if new_fit3 < fitness[i]:
                    X[i] = new_X3
                    fitness[i] = new_fit3

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
