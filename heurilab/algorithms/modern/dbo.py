"""DBO — Dung Beetle Optimizer (Xue & Shen, 2023)"""

import numpy as np
from heurilab.algorithms.base import _Base


class DBO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]
        worst_idx = np.argmax(fitness)
        worst = X[worst_idx].copy()

        convergence = [best_fit]

        # Split into 4 behavior groups
        n1 = self.pop_size // 4  # ball-rolling
        n2 = self.pop_size // 4  # breeding
        n3 = self.pop_size // 4  # foraging (small)
        # remaining: thief

        for t in range(self.max_iter):
            r = 1 - t / self.max_iter

            # Ball-rolling dung beetles (exploration)
            for i in range(n1):
                alpha = self.rng.random()
                k = 0.1 * self.rng.standard_normal()
                delta = np.abs(X[i] - worst)
                if self.rng.random() < 0.9:
                    X[i] = X[i] + alpha * delta + k * (X[i] - best)
                else:
                    X[i] = X[i] + np.tan(self.rng.random() * np.pi / 6) * np.abs(X[i] - best)
                X[i] = self._clip(X[i])

            # Breeding dung beetles (exploitation near best)
            for i in range(n1, n1 + n2):
                R = self.rng.random(self.dim)
                b1 = best * (1 - r)
                b2 = best * (1 + r)
                lb_local = np.maximum(self.lb, b1)
                ub_local = np.minimum(self.ub, b2)
                X[i] = best + R * (lb_local + self.rng.random(self.dim) * (ub_local - lb_local) - X[i])
                X[i] = self._clip(X[i])

            # Small dung beetles (foraging)
            for i in range(n1 + n2, n1 + n2 + n3):
                C1 = self.rng.random(self.dim) * X[i]
                C2 = X[i] + self.rng.standard_normal(self.dim) * (X[i] - best) * r
                X[i] = X[i] + C1 * self.rng.standard_normal() + C2 * self.rng.standard_normal()
                X[i] = self._clip(X[i])

            # Thief dung beetles (stealing)
            for i in range(n1 + n2 + n3, self.pop_size):
                g = self.rng.standard_normal(self.dim)
                X[i] = best + g * (np.abs(X[i] - best) + np.abs(X[i] - worst)) / 2
                X[i] = self._clip(X[i])

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])
            min_idx = np.argmin(fitness)
            max_idx = np.argmax(fitness)

            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]
            worst = X[max_idx].copy()

            convergence.append(best_fit)

        return best, best_fit, convergence
