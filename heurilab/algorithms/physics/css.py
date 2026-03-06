"""CSS — Charged System Search"""

import numpy as np
from heurilab.algorithms.base import _Base


class CSS(_Base):
    def optimize(self):
        X = self._init_pop()
        V = np.zeros((self.pop_size, self.dim))
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            worst_fit = np.max(fitness)
            best_curr = np.min(fitness)
            eps = 1e-16

            # Charge magnitude
            q = (fitness - worst_fit) / (best_curr - worst_fit + eps)

            # Separation distance
            a_search = np.max(self.ub - self.lb)
            r_a = 0.1 * a_search * (1 - t / self.max_iter)

            force = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if i == j:
                        continue
                    r_ij = np.linalg.norm(X[i] - X[j]) + eps

                    # Decide if i is inside or outside
                    if r_ij < r_a:
                        force_mag = q[j] * (X[j] - X[i]) / (r_a ** 3 + eps)
                    else:
                        force_mag = q[j] * (X[j] - X[i]) / (r_ij ** 2 + eps)

                    p_ij = 1.0 if fitness[j] < fitness[i] else 0.0
                    force[i] += np.random.rand() * force_mag * p_ij

            ka = 0.5 * (1 - t / self.max_iter)
            kv = 0.5 * (1 + t / self.max_iter)

            V = kv * V + ka * force
            X = self._clip(X + V)
            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
