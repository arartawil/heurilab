"""EDO — Exponential Distribution Optimizer (Abdel-Basset et al., 2023)"""

import numpy as np
from heurilab.algorithms.base import _Base


class EDO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]
        mu_guide = np.mean(X, axis=0)  # guiding mean

        for t in range(self.max_iter):
            alpha = 2 * (1 - t / self.max_iter)  # exploration factor
            mu_guide = np.mean(X, axis=0)

            for i in range(self.pop_size):
                r = np.random.rand()

                if r < 0.5:
                    # Exploitation: exponential distribution guided search
                    lam = 1 / (np.abs(best - X[i]) + 1e-16)
                    # Exponential random variate per dimension
                    exp_rand = -np.log(np.random.rand(self.dim) + 1e-16) / (lam + 1e-16)
                    # Clip exponential values to avoid overflow
                    exp_rand = np.minimum(exp_rand, self.ub - self.lb)
                    direction = np.sign(best - X[i])
                    new_X = X[i] + direction * exp_rand * (1 - t / self.max_iter)
                else:
                    # Exploration: guided by population mean and random walk
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    j = np.random.randint(self.pop_size)
                    new_X = X[i] + alpha * r1 * (mu_guide - X[i]) + r2 * (X[j] - X[i])

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
