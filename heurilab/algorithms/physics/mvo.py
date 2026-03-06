"""MVO — Multi-Verse Optimizer"""

import numpy as np
from heurilab.algorithms.base import _Base


class MVO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        sorted_idx = np.argsort(fitness)
        best = X[sorted_idx[0]].copy()
        best_fit = fitness[sorted_idx[0]]

        convergence = [best_fit]

        for t in range(self.max_iter):
            WEP = 0.2 + (1.0 - 0.2) * (t / self.max_iter)
            TDR = 1 - ((t) ** (1 / 6)) / (self.max_iter ** (1 / 6))

            # Normalize fitness for roulette
            sorted_idx = np.argsort(fitness)
            norm_fit = (fitness - fitness[sorted_idx[0]]) + 1e-16
            norm_fit = norm_fit / np.sum(norm_fit)

            for i in range(self.pop_size):
                for j in range(self.dim):
                    r1 = np.random.rand()
                    if r1 < norm_fit[i]:
                        # White hole: select random universe by roulette
                        white_idx = sorted_idx[np.random.randint(self.pop_size)]
                        X[i, j] = X[white_idx, j]

                    r2 = np.random.rand()
                    if r2 < WEP:
                        r3, r4 = np.random.rand(), np.random.rand()
                        if r3 < 0.5:
                            X[i, j] = best[j] + TDR * ((self.ub[j] - self.lb[j]) * r4 + self.lb[j])
                        else:
                            X[i, j] = best[j] - TDR * ((self.ub[j] - self.lb[j]) * r4 + self.lb[j])

                X[i] = self._clip(X[i])

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
