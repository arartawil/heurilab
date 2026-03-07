"""COA — Coati Optimization Algorithm (Dehghani et al., 2023)"""

import numpy as np
from heurilab.algorithms.base import _Base


class COA(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            half = self.pop_size // 2

            # Phase 1: Hunting strategy (iguana hunting) — first half
            for i in range(half):
                r = np.random.rand(self.dim)
                I = np.random.choice([1, 2])

                # Iguana position (random in tree = upper search space)
                iguana = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
                iguana_fit = self._eval(iguana)

                if iguana_fit < fitness[i]:
                    new_X = X[i] + r * (iguana - I * X[i])
                else:
                    new_X = X[i] + r * (X[i] - iguana)

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)
                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

            # Phase 2: Escaping predators — second half
            for i in range(half, self.pop_size):
                r = np.random.rand(self.dim)
                # Coati escaping to safe area near the best
                new_X = X[i] + (1 - 2 * r) * (best - I * X[i]) * (1 - t / self.max_iter)

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
