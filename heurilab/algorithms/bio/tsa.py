"""TSA — Tree Seed Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class TSA(_Base):
    def optimize(self):
        ST = 0.1  # Search tendency (probability of using best tree)

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                # Generate seeds
                n_seeds = np.random.randint(1, max(2, self.pop_size // 5))

                for _ in range(n_seeds):
                    seed = X[i].copy()
                    k = np.random.randint(self.pop_size)
                    while k == i:
                        k = np.random.randint(self.pop_size)

                    for j in range(self.dim):
                        if np.random.rand() < ST:
                            # Use best tree
                            seed[j] = X[i, j] + np.random.uniform(-1, 1) * (best[j] - X[k, j])
                        else:
                            # Use random tree
                            seed[j] = X[i, j] + np.random.uniform(-1, 1) * (X[i, j] - X[k, j])

                    seed = self._clip(seed)
                    seed_fit = self._eval(seed)

                    if seed_fit < fitness[i]:
                        X[i] = seed
                        fitness[i] = seed_fit

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
