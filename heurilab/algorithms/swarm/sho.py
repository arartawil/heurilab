"""SHO — Spotted Hyena Optimizer"""

import numpy as np
from heurilab.algorithms.base import _Base


class SHO(_Base):
    def optimize(self):
        h = 5.0  # Maximum value of h vector

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            h_val = h - t * (h / self.max_iter)

            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                B = 2 * r1  # Encircling coefficient
                E = 2 * h_val * r2 - h_val  # Updated coefficient

                D = np.abs(B * best - X[i])

                # Cluster formation: aggregate N best hyenas
                N = max(1, int(np.ceil(np.random.rand() * self.pop_size * 0.3)))
                sorted_idx = np.argsort(fitness)[:N]
                cluster_center = np.mean(X[sorted_idx], axis=0)

                if np.abs(np.mean(E)) >= 1:
                    # Exploration: search
                    X[i] = self._clip(cluster_center - E * D)
                else:
                    # Exploitation: attack
                    X[i] = self._clip(cluster_center - E * np.abs(D))

                fitness[i] = self._eval(X[i])

                if fitness[i] < best_fit:
                    best = X[i].copy()
                    best_fit = fitness[i]

            convergence.append(best_fit)

        return best, best_fit, convergence
