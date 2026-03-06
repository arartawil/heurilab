"""EO — Equilibrium Optimizer"""

import numpy as np
from heurilab.algorithms.base import _Base


class EO(_Base):
    def optimize(self):
        a1, a2 = 2.0, 1.0
        GP = 0.5  # Generation probability

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        sorted_idx = np.argsort(fitness)
        best = X[sorted_idx[0]].copy()
        best_fit = fitness[sorted_idx[0]]

        # Equilibrium pool: top 4 + average
        n_eq = min(4, self.pop_size)
        C_eq = [X[sorted_idx[k]].copy() for k in range(n_eq)]
        C_eq.append(np.mean(np.array(C_eq), axis=0))

        convergence = [best_fit]

        for t in range(self.max_iter):
            tt = (1 - t / self.max_iter) ** (a2 * t / self.max_iter)

            for i in range(self.pop_size):
                # Random equilibrium candidate
                eq = C_eq[np.random.randint(len(C_eq))]

                r = np.random.rand(self.dim)
                lam = np.random.rand(self.dim)
                F = a1 * np.sign(r - 0.5) * (np.exp(-lam * tt) - 1)

                # Generation rate
                if np.random.rand() < GP:
                    GCP = 0.5 * np.random.rand(self.dim)
                else:
                    GCP = np.zeros(self.dim)
                G = GCP * (eq - lam * X[i])

                new_X = self._clip(eq + (X[i] - eq) * F + G * (1 - F) / (lam + 1e-16))
                new_fit = self._eval(new_X)

                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

            # Update equilibrium pool
            sorted_idx = np.argsort(fitness)
            C_eq = [X[sorted_idx[k]].copy() for k in range(n_eq)]
            C_eq.append(np.mean(np.array(C_eq), axis=0))

            if fitness[sorted_idx[0]] < best_fit:
                best = X[sorted_idx[0]].copy()
                best_fit = fitness[sorted_idx[0]]

            convergence.append(best_fit)

        return best, best_fit, convergence
