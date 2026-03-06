"""RIME — Rime Optimization Algorithm (Su et al., 2022)"""

import numpy as np
from heurilab.algorithms.base import _Base


class RIME(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            E = (1 - t / self.max_iter) ** 5  # Rime factor

            for i in range(self.pop_size):
                new_X = X[i].copy()

                for j in range(self.dim):
                    # Soft-rime search
                    if np.random.rand() < E:
                        new_X[j] = best[j] + np.random.randn() * (best[j] - X[i][j]) * E
                    else:
                        # Hard-rime puncture
                        r1 = np.random.randint(self.pop_size)
                        new_X[j] = X[r1][j]

                new_X = self._clip(new_X)

                # Positive greedy selection
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
