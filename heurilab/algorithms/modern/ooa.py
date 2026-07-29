"""OOA — Osprey Optimization Algorithm (Dehghani et al., 2023)"""

import numpy as np
from heurilab.algorithms.base import _Base


class OOA(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                # Phase 1: Position identification and fish hunting (exploration)
                # Osprey identifies fish position
                fish_pos = self.lb + self.rng.random(self.dim) * (self.ub - self.lb)
                r = self.rng.random(self.dim)
                I = self.rng.choice([1, 2])

                fish_fit = self._eval(fish_pos)
                if fish_fit < fitness[i]:
                    new_X = X[i] + r * (fish_pos - I * X[i])
                else:
                    new_X = X[i] + r * (X[i] - fish_pos)

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)
                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

                # Phase 2: Carrying fish to suitable position (exploitation)
                j = self.rng.integers(self.pop_size)
                while j == i:
                    j = self.rng.integers(self.pop_size)

                r = self.rng.random(self.dim)
                sf = (1 - t / self.max_iter)  # shrinking factor

                if fitness[i] < fitness[j]:
                    new_X = X[i] + r * sf * (X[i] - X[j])
                else:
                    new_X = X[i] + r * sf * (X[j] - X[i])

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
