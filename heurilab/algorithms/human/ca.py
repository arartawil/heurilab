"""CA — Cultural Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class CA(_Base):
    def optimize(self):
        accept_rate = 0.2  # Fraction accepted into belief space

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        # Belief space: normative knowledge (ranges)
        bs_lb = self.lb.copy()
        bs_ub = self.ub.copy()

        convergence = [best_fit]

        for t in range(self.max_iter):
            # Influence function: use belief space to guide search
            new_X = X.copy()
            for i in range(self.pop_size):
                for j in range(self.dim):
                    if X[i, j] < bs_lb[j]:
                        new_X[i, j] = X[i, j] + np.random.rand() * (bs_lb[j] - X[i, j])
                    elif X[i, j] > bs_ub[j]:
                        new_X[i, j] = X[i, j] - np.random.rand() * (X[i, j] - bs_ub[j])
                    else:
                        new_X[i, j] = X[i, j] + np.random.randn() * (bs_ub[j] - bs_lb[j]) * 0.1

                new_X[i] = self._clip(new_X[i])

            new_fitness = np.array([self._eval(new_X[i]) for i in range(self.pop_size)])

            # Selection: keep better individuals
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    X[i] = new_X[i]
                    fitness[i] = new_fitness[i]

            # Update belief space with top individuals
            n_accept = max(1, int(accept_rate * self.pop_size))
            accepted = np.argsort(fitness)[:n_accept]
            acc_X = X[accepted]

            bs_lb = np.min(acc_X, axis=0)
            bs_ub = np.max(acc_X, axis=0)

            # Ensure belief space doesn't collapse completely
            collapsed = bs_ub - bs_lb < 1e-10
            bs_lb[collapsed] = self.lb[collapsed]
            bs_ub[collapsed] = self.ub[collapsed]

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
