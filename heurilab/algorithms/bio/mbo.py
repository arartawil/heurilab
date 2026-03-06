"""MBO — Monarch Butterfly Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class MBO(_Base):
    def optimize(self):
        p_mbo = 5.0 / 12.0  # Ratio of Land 1 population
        BAR = 5.0 / 12.0     # Butterfly adjusting rate
        peri = 1.2            # Migration period

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        NP1 = max(1, int(np.ceil(p_mbo * self.pop_size)))
        NP2 = self.pop_size - NP1

        convergence = [best_fit]

        for t in range(self.max_iter):
            sorted_idx = np.argsort(fitness)
            land1_idx = sorted_idx[:NP1]  # Best NP1 in Land 1
            land2_idx = sorted_idx[NP1:]  # Rest in Land 2

            new_X = X.copy()

            # ── Migration operator (Land 1 butterflies) ──
            for i in land1_idx:
                for j in range(self.dim):
                    r1 = np.random.rand() * peri
                    if r1 <= p_mbo:
                        r_idx = np.random.choice(land1_idx)
                        new_X[i, j] = X[r_idx, j]
                    else:
                        r_idx = np.random.choice(land2_idx) if len(land2_idx) > 0 else np.random.choice(land1_idx)
                        new_X[i, j] = X[r_idx, j]

            # ── Butterfly adjusting operator (Land 2 butterflies) ──
            for i in land2_idx:
                scale = (self.ub - self.lb) / (t + 1) ** 2
                for j in range(self.dim):
                    if np.random.rand() >= BAR:
                        new_X[i, j] = best[j]
                    else:
                        r_idx = np.random.choice(land2_idx) if len(land2_idx) > 1 else np.random.choice(self.pop_size)
                        new_X[i, j] = X[r_idx, j]
                        if np.random.rand() > BAR:
                            new_X[i, j] += scale[j] * (2 * np.random.rand() - 1)

            new_X = np.array([self._clip(new_X[i]) for i in range(self.pop_size)])
            new_fitness = np.array([self._eval(new_X[i]) for i in range(self.pop_size)])

            # Greedy selection
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    X[i] = new_X[i]
                    fitness[i] = new_fitness[i]

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
