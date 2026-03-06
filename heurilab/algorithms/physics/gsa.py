"""GSA — Gravitational Search Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class GSA(_Base):
    def optimize(self):
        G0, alpha = 100, 20
        X = self._init_pop()
        V = np.zeros((self.pop_size, self.dim))
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            G = G0 * np.exp(-alpha * t / self.max_iter)

            worst = np.max(fitness)
            bst = np.min(fitness)
            eps = 1e-16

            if worst == bst:
                mass = np.ones(self.pop_size)
            else:
                mass = (worst - fitness) / (worst - bst + eps)
            mass = mass / (np.sum(mass) + eps)

            # Use K-best agents
            K = max(1, int(self.pop_size * (1 - t / self.max_iter)))
            sorted_idx = np.argsort(fitness)[:K]

            force = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):
                for j in sorted_idx:
                    if i != j:
                        R = np.linalg.norm(X[i] - X[j]) + eps
                        force[i] += np.random.rand(self.dim) * G * mass[j] * (X[j] - X[i]) / R

            acc = force / (mass.reshape(-1, 1) + eps)
            V = np.random.rand(self.pop_size, self.dim) * V + acc
            X = self._clip(X + V)

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
