"""AO — Aquila Optimizer (Abualigah et al., 2021)"""

import numpy as np
from heurilab.algorithms.base import _Base


class AO(_Base):
    def optimize(self):
        alpha, delta = 0.1, 0.1

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            t1 = t / self.max_iter
            G = 2 * (1 - t1)  # Linearly decreasing from 2 to 0

            mean_X = np.mean(X, axis=0)

            for i in range(self.pop_size):
                r1 = np.random.rand()

                if t1 <= 2 / 3:
                    if r1 < 0.5:
                        # X1: Expanded exploration (high soar with vertical stoop)
                        levy = self._levy_flight(self.dim)
                        new_X = best * (1 - t1) + (mean_X - best * r1) * np.random.rand()
                    else:
                        # X2: Narrowed exploration (contour flight with glide attack)
                        theta = -np.pi + 2 * np.pi * np.random.rand()
                        r_spiral = np.random.rand() * (self.ub - self.lb) * t1
                        new_X = best - mean_X * alpha + r_spiral * np.cos(theta)
                else:
                    if r1 < 0.5:
                        # X3: Expanded exploitation (low flight with gradual descent)
                        levy = self._levy_flight(self.dim)
                        QF = t1 ** 2  # Quality function
                        new_X = (best - mean_X) * alpha - np.random.rand() + ((self.ub - self.lb) * np.random.rand() + self.lb) * delta
                    else:
                        # X4: Narrowed exploitation (walk and grab)
                        QF = t1 ** 2
                        levy = self._levy_flight(self.dim)
                        new_X = QF * best - (G * np.random.rand() * X[i]) * np.abs(2 * np.random.rand() * best - X[i])

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

                if new_fit < best_fit:
                    best = new_X.copy()
                    best_fit = new_fit

            convergence.append(best_fit)

        return best, best_fit, convergence

    def _levy_flight(self, dim, beta=1.5):
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(dim) * sigma_u
        v = np.random.randn(dim)
        return u / (np.abs(v) ** (1 / beta))
