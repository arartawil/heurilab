"""SAO — Snow Ablation Optimizer (Deng & Liu, 2023)"""

import numpy as np
from heurilab.algorithms.base import _Base


class SAO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            T = np.exp(-t / self.max_iter)  # temperature factor
            N_elite = max(1, int(self.pop_size * T * 0.5))
            sorted_idx = np.argsort(fitness)

            for i in range(self.pop_size):
                if i in sorted_idx[:N_elite]:
                    # Elite snow (sublimation — exploitation)
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand()
                    new_X = best + r1 * (best - X[i]) * T * np.log(1 + r2)
                else:
                    # Non-elite snow (melting — exploration)
                    j = np.random.randint(self.pop_size)
                    k = np.random.randint(self.pop_size)
                    r = np.random.rand(self.dim)

                    if fitness[j] < fitness[i]:
                        new_X = X[i] + r * (X[j] - X[k]) * (1 - t / self.max_iter)
                    else:
                        # Random ablation
                        new_X = X[i] + np.random.randn(self.dim) * (self.ub - self.lb) * T * 0.1

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
