"""ASO — Atom Search Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class ASO(_Base):
    def optimize(self):
        alpha_aso = 50   # Depth weight
        beta_aso = 0.2   # Multiplier of Kbest

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

            # Mass
            if worst == bst:
                mass = np.ones(self.pop_size)
            else:
                mass = np.exp(-(fitness - bst) / (worst - bst + eps))
            mass = mass / (np.sum(mass) + eps)

            K_best = max(2, int(self.pop_size * (1 - 0.9 * t / self.max_iter)))
            sorted_idx = np.argsort(fitness)[:K_best]

            # Interaction force
            force = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):
                for j in sorted_idx:
                    if i == j:
                        continue
                    r_ij = np.linalg.norm(X[i] - X[j]) + eps
                    sigma_ij = np.abs((self.ub - self.lb)).mean() * 0.1

                    # Lennard-Jones-like potential
                    h = sigma_ij / (r_ij + eps)
                    attract = -alpha_aso * (h ** 6)
                    repel = alpha_aso * (h ** 12) * 0.5

                    f_ij = (attract + repel) * np.random.rand()
                    direction = (X[j] - X[i]) / (r_ij + eps)
                    force[i] += f_ij * mass[j] * direction

            # Constraint force toward best
            G = np.exp(-20.0 * t / self.max_iter)

            accel = force / (mass.reshape(-1, 1) + eps) + G * (best - X)

            V = np.random.rand(self.pop_size, self.dim) * V + accel
            X = self._clip(X + V)

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
