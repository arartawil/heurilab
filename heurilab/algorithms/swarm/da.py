"""DA — Dragonfly Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class DA(_Base):
    def optimize(self):
        X = self._init_pop()
        V = np.zeros((self.pop_size, self.dim))
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        worst_idx = np.argmax(fitness)
        worst = X[worst_idx].copy()

        convergence = [best_fit]

        for t in range(self.max_iter):
            w = 0.9 - t * (0.9 - 0.4) / self.max_iter
            ratio = t / self.max_iter
            s = 2 * np.random.rand() * (1 - ratio)  # Separation
            a = 2 * np.random.rand() * ratio         # Alignment
            c = 2 * np.random.rand() * ratio         # Cohesion
            f = 2 * np.random.rand()                  # Food attraction
            e = (1 - ratio) * np.random.rand()        # Enemy distraction

            for i in range(self.pop_size):
                # Find neighbours within radius
                dists = np.linalg.norm(X - X[i], axis=1)
                radius = (np.max(self.ub - self.lb)) * (1 - ratio) + 1e-16
                neighbours = np.where((dists < radius) & (dists > 0))[0]

                if len(neighbours) > 0:
                    S = -np.sum(X[neighbours] - X[i], axis=0)
                    A = np.mean(V[neighbours], axis=0)
                    C = np.mean(X[neighbours], axis=0) - X[i]
                else:
                    S = np.zeros(self.dim)
                    A = np.zeros(self.dim)
                    C = np.zeros(self.dim)

                F = best - X[i]
                E = worst - X[i]

                if len(neighbours) > 0:
                    V[i] = w * V[i] + s * S + a * A + c * C + f * F - e * E
                    X[i] = self._clip(X[i] + V[i])
                else:
                    # Lévy flight for isolated dragonflies
                    X[i] = self._clip(X[i] + np.random.randn(self.dim) * (self.ub - self.lb) * 0.01)

                fit = self._eval(X[i])
                fitness[i] = fit

                if fit < best_fit:
                    best = X[i].copy()
                    best_fit = fit

            worst_idx = np.argmax(fitness)
            worst = X[worst_idx].copy()

            convergence.append(best_fit)

        return best, best_fit, convergence
