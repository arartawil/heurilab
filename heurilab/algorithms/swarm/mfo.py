"""MFO — Moth-Flame Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class MFO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        sorted_idx = np.argsort(fitness)
        flames = X[sorted_idx].copy()
        flame_fit = fitness[sorted_idx].copy()

        best = flames[0].copy()
        best_fit = flame_fit[0]

        convergence = [best_fit]

        for t in range(self.max_iter):
            n_flames = max(1, int(self.pop_size - t * (self.pop_size - 1) / self.max_iter))
            b = 1

            for i in range(self.pop_size):
                flame_idx = min(i, n_flames - 1)
                D = np.abs(flames[flame_idx] - X[i])
                tt = np.random.uniform(-1, 1, self.dim)
                X[i] = D * np.exp(b * tt) * np.cos(2 * np.pi * tt) + flames[flame_idx]
                X[i] = self._clip(X[i])

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

            # Merge and sort
            combined = np.vstack([flames, X])
            combined_fit = np.hstack([flame_fit, fitness])
            sorted_idx = np.argsort(combined_fit)
            flames = combined[sorted_idx[:self.pop_size]].copy()
            flame_fit = combined_fit[sorted_idx[:self.pop_size]].copy()

            if flame_fit[0] < best_fit:
                best = flames[0].copy()
                best_fit = flame_fit[0]

            convergence.append(best_fit)

        return best, best_fit, convergence
