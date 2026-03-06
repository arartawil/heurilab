"""TLGO — Teaching-Learning-based Genetic Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class TLGO(_Base):
    def optimize(self):
        pc, pm = 0.8, 0.05

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            mean_X = np.mean(X, axis=0)
            teacher = X[np.argmin(fitness)].copy()

            # ── Teaching phase ──
            for i in range(self.pop_size):
                TF = np.random.randint(1, 3)
                r = np.random.rand(self.dim)
                new_X = X[i] + r * (teacher - TF * mean_X)
                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)
                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

            # ── Genetic crossover phase ──
            for i in range(0, self.pop_size - 1, 2):
                if np.random.rand() < pc:
                    alpha = np.random.rand(self.dim)
                    c1 = alpha * X[i] + (1 - alpha) * X[i + 1]
                    c2 = (1 - alpha) * X[i] + alpha * X[i + 1]
                    c1 = self._clip(c1)
                    c2 = self._clip(c2)
                    f1 = self._eval(c1)
                    f2 = self._eval(c2)
                    if f1 < fitness[i]:
                        X[i] = c1
                        fitness[i] = f1
                    if f2 < fitness[i + 1]:
                        X[i + 1] = c2
                        fitness[i + 1] = f2

            # ── Mutation phase ──
            for i in range(self.pop_size):
                if np.random.rand() < pm:
                    j = np.random.randint(self.dim)
                    X[i, j] = np.random.uniform(self.lb[j], self.ub[j])
                    fitness[i] = self._eval(X[i])

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
