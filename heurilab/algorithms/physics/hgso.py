"""HGSO — Henry Gas Solubility Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class HGSO(_Base):
    def optimize(self):
        n_types = 5  # Number of gas types
        K = 1.0      # Henry's constant
        alpha_h = 1.0
        beta_h = 1.0
        T0, theta = 298.15, 1.0

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        # Assign gas types
        types = np.random.randint(0, n_types, self.pop_size)

        convergence = [best_fit]

        for t in range(self.max_iter):
            T = np.exp(-t / self.max_iter)  # Temperature decrease
            H = K * np.exp(-alpha_h * (1.0 / T - 1.0 / T0))  # Solubility

            # Find best in each cluster
            cluster_best = {}
            for c_type in range(n_types):
                mask = np.where(types == c_type)[0]
                if len(mask) > 0:
                    cb_idx = mask[np.argmin(fitness[mask])]
                    cluster_best[c_type] = cb_idx

            for i in range(self.pop_size):
                r1 = np.random.rand()
                gamma = beta_h * np.exp(-best_fit / (fitness[i] + 1e-16))

                if r1 > 0.5:
                    # Move toward best
                    new_X = X[i] + H * gamma * (best - X[i])
                else:
                    # Move toward cluster best
                    cb = cluster_best.get(types[i], best_idx)
                    new_X = X[i] + H * gamma * (X[cb] - X[i])

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

            # Worst agents re-initialize
            n_worst = max(1, int(0.1 * self.pop_size))
            worst_idx = np.argsort(fitness)[-n_worst:]
            for idx in worst_idx:
                X[idx] = np.random.uniform(self.lb, self.ub, self.dim)
                fitness[idx] = self._eval(X[idx])

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
