"""GMO — Geometric Mean Optimizer (Rezaei et al., 2024)"""

import numpy as np
from heurilab.algorithms.base import _Base


class GMO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            T_f = 1 - t / self.max_iter
            a = 2 * T_f  # adaptive parameter

            # Compute geometric mean of population (per dimension)
            # Use log-space to avoid numerical issues
            X_positive = np.abs(X) + 1e-16
            geo_mean = np.exp(np.mean(np.log(X_positive), axis=0))
            # Restore sign using majority vote
            sign_vote = np.sign(np.mean(X, axis=0))
            sign_vote[sign_vote == 0] = 1
            geo_mean = geo_mean * sign_vote

            for i in range(self.pop_size):
                r = np.random.rand()

                if r < 0.5:
                    # Phase 1: Geometric mean attraction (exploitation)
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    new_X = X[i] + r1 * (geo_mean - X[i]) + r2 * a * (best - X[i])
                else:
                    # Phase 2: Exploration using GM-based perturbation
                    j = np.random.randint(self.pop_size)
                    k = np.random.randint(self.pop_size)
                    r1 = np.random.rand(self.dim)

                    # GM-based differential vector
                    diff = np.abs(X[j] - X[k]) + 1e-16
                    gm_diff = np.sqrt(diff * np.abs(best - X[i]) + 1e-16)
                    direction = np.sign(best - X[i])
                    new_X = X[i] + a * r1 * gm_diff * direction

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
