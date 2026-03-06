"""CFO — Central Force Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class CFO(_Base):
    def optimize(self):
        G0 = 2.0    # Initial gravitational constant
        alpha = 2.0  # Gravity exponent
        beta = 2.0   # Mass exponent

        X = self._init_pop()
        V = np.zeros((self.pop_size, self.dim))
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            G = G0 * (1 - t / self.max_iter)
            eps = 1e-16

            # Mass based on fitness
            worst = np.max(fitness)
            bst = np.min(fitness)
            if worst == bst:
                mass = np.ones(self.pop_size)
            else:
                mass = (worst - fitness) / (worst - bst + eps)
            mass = mass / (np.sum(mass) + eps)

            accel = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if i == j:
                        continue
                    if fitness[j] >= fitness[i]:
                        continue  # Only attracted to better probes

                    R = np.linalg.norm(X[j] - X[i]) + eps
                    direction = (X[j] - X[i]) / R
                    accel[i] += G * (mass[j] ** beta) * direction / (R ** alpha)

            # Update velocity and position
            dt = 1.0
            V = np.random.rand(self.pop_size, self.dim) * V + accel * dt
            X = self._clip(X + V * dt)

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
