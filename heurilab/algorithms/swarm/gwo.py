"""GWO — Grey Wolf Optimizer"""

import numpy as np
from heurilab.algorithms.base import _Base


class GWO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        sorted_idx = np.argsort(fitness)
        alpha, beta, delta = X[sorted_idx[0]].copy(), X[sorted_idx[1]].copy(), X[sorted_idx[2]].copy()
        alpha_fit = fitness[sorted_idx[0]]

        convergence = [alpha_fit]

        for t in range(self.max_iter):
            a = 2 - 2 * t / self.max_iter

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = np.abs(C1 * alpha - X[i])
                X1 = alpha - A1 * D_alpha

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = np.abs(C2 * beta - X[i])
                X2 = beta - A2 * D_beta

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = np.abs(C3 * delta - X[i])
                X3 = delta - A3 * D_delta

                X[i] = self._clip((X1 + X2 + X3) / 3)

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])
            sorted_idx = np.argsort(fitness)

            if fitness[sorted_idx[0]] < alpha_fit:
                alpha = X[sorted_idx[0]].copy()
                alpha_fit = fitness[sorted_idx[0]]
            beta = X[sorted_idx[1]].copy()
            delta = X[sorted_idx[2]].copy()

            convergence.append(alpha_fit)

        return alpha, alpha_fit, convergence
