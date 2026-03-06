"""SSOA — Sparrow Search Optimization Algorithm (Xue & Shen, 2020)"""

import numpy as np
from heurilab.algorithms.base import _Base


class SSOA(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]
        lb = np.array(self.lb)
        ub = np.array(self.ub)

        PD = 0.2  # proportion of producers/discoverers
        SD = 0.1  # proportion of sparrows aware of danger
        ST = 0.8  # safety threshold

        n_producers = max(1, int(self.pop_size * PD))
        n_danger = max(1, int(self.pop_size * SD))

        for t in range(self.max_iter):
            sorted_idx = np.argsort(fitness)
            X = X[sorted_idx]
            fitness = fitness[sorted_idx]

            R2 = np.random.rand()  # alarm value

            # Producers (discoverers) — best individuals
            for i in range(n_producers):
                if R2 < ST:
                    alpha = np.random.rand()
                    X[i] = X[i] * np.exp(-i / (alpha * self.max_iter + 1e-16))
                else:
                    Q = np.random.randn(self.dim)
                    X[i] = X[i] + Q

                X[i] = self._clip(X[i])

            # Scroungers (joiners) — follow producers
            worst_fit = fitness[-1]
            for i in range(n_producers, self.pop_size):
                if i > self.pop_size // 2:
                    # Worst half — go to random location
                    Q = np.random.randn(self.dim)
                    X[i] = Q * np.exp((fitness[-1] - fitness[i]) / (i * i + 1e-16))
                else:
                    # Follow the best producer
                    A = np.random.choice([-1, 1], size=(self.dim,))
                    A_plus = A / (A @ A + 1e-16)
                    X[i] = X[0] + np.abs(X[i] - X[0]) * A_plus

                X[i] = self._clip(X[i])

            # Danger awareness — scouts
            danger_idx = np.random.choice(self.pop_size, n_danger, replace=False)
            for idx in danger_idx:
                if fitness[idx] > np.mean(fitness):
                    beta = np.random.randn(self.dim)
                    X[idx] = best + beta * np.abs(X[idx] - best)
                elif fitness[idx] == best_fit:
                    eps_val = 1e-10
                    K = np.random.uniform(-1, 1)
                    X[idx] = X[idx] + K * (np.abs(X[idx] - best) / (fitness[idx] - best_fit + eps_val))

                X[idx] = self._clip(X[idx])

            # Re-evaluate
            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
