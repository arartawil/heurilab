"""RUN — RUNge Kutta Optimizer (Ahmadianfar et al., 2021)"""

import numpy as np
from heurilab.algorithms.base import _Base


class RUN(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            f = 20 * np.exp(-(12 * t / self.max_iter))  # Adaptive factor
            SF = 2 * (0.5 - np.random.rand(self.pop_size)) * f  # Solution factor

            for i in range(self.pop_size):
                r1 = np.random.rand()
                r2 = np.random.rand()

                # Select random solutions
                idxs = list(range(self.pop_size))
                idxs.remove(i)
                rand1, rand2 = np.random.choice(idxs, 2, replace=False)

                x_r1 = X[rand1]
                x_r2 = X[rand2]

                delta_x = np.random.rand() * np.abs(best - x_r1)
                gamma = np.random.rand() * (X[i] - np.random.rand(self.dim) * (self.ub - self.lb)) * np.exp(-4 * t / self.max_iter)

                # RK slopes
                K1 = 0.5 * (np.random.rand() * x_r1 - X[i])
                K2 = 0.5 * (np.random.rand() * (x_r1 + K1 * delta_x / 2) - (X[i] + K1 / 2))
                K3 = 0.5 * (np.random.rand() * (x_r1 + K2 * delta_x / 2) - (X[i] + K2 / 2))
                K4 = 0.5 * (np.random.rand() * (x_r1 + K3 * delta_x) - (X[i] + K3))

                # RK update
                SM = (K1 + 2 * K2 + 2 * K3 + K4) / 6

                if r1 < 0.5:
                    new_X = best + np.random.randn(self.dim) * SM + gamma
                else:
                    new_X = x_r2 + np.random.randn(self.dim) * SM + gamma

                # Enhanced solution quality (ESQ)
                if r2 < 0.5:
                    r3 = np.random.rand()
                    EQ = r3 * best - r3 * x_r2
                    new_X = new_X + SF[i] * EQ

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
