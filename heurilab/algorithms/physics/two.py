"""TWO — Tug of War Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class TWO(_Base):
    def optimize(self):
        mu_s = 1.0  # Static friction coefficient

        X = self._init_pop()
        V = np.zeros((self.pop_size, self.dim))
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            eps = 1e-16
            worst = np.max(fitness)
            bst = np.min(fitness)

            # Weight (heavier = better fitness)
            if worst == bst:
                weight = np.ones(self.pop_size)
            else:
                weight = 1 + (worst - fitness) / (worst - bst + eps)

            delta = 1 - t / self.max_iter  # Decreasing factor

            for i in range(self.pop_size):
                force = np.zeros(self.dim)

                for j in range(self.pop_size):
                    if i == j:
                        continue

                    # Rope tension
                    if fitness[j] < fitness[i]:
                        direction = X[j] - X[i]
                        r = np.linalg.norm(direction) + eps
                        tension = weight[j] / (r + eps)
                        force += np.random.rand() * tension * direction / r

                # Acceleration = Force / mass
                accel = force / (weight[i] + eps)

                V[i] = delta * V[i] + accel
                X[i] = self._clip(X[i] + V[i])

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
