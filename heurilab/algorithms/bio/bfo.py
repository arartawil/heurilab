"""BFO — Bacterial Foraging Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class BFO(_Base):
    def optimize(self):
        N_c = 4       # Chemotaxis steps
        N_s = 4       # Swim steps
        d_attract = 0.1
        w_attract = 0.2
        h_repel = 0.1
        w_repel = 10.0
        p_ed = 0.25   # Elimination-dispersal probability

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        iters_per_gen = max(1, self.max_iter // N_c)

        for t in range(self.max_iter):
            # ── Chemotaxis ──
            for i in range(self.pop_size):
                # Random tumble direction
                delta = np.random.randn(self.dim)
                delta = delta / (np.linalg.norm(delta) + 1e-16)

                step_size = 0.1 * (self.ub - self.lb) * (1 - t / self.max_iter)
                new_X = self._clip(X[i] + step_size * delta)
                new_fit = self._eval(new_X)

                # Swim in same direction if improved
                for _ in range(N_s):
                    if new_fit < fitness[i]:
                        X[i] = new_X
                        fitness[i] = new_fit
                        new_X = self._clip(X[i] + step_size * delta)
                        new_fit = self._eval(new_X)
                    else:
                        break

                if fitness[i] < best_fit:
                    best = X[i].copy()
                    best_fit = fitness[i]

            # ── Reproduction ──
            sorted_idx = np.argsort(fitness)
            half = self.pop_size // 2
            for i in range(half):
                src = sorted_idx[i]
                dst = sorted_idx[self.pop_size - 1 - i]
                X[dst] = X[src].copy()
                fitness[dst] = fitness[src]

            # ── Elimination-Dispersal ──
            for i in range(self.pop_size):
                if np.random.rand() < p_ed:
                    X[i] = np.random.uniform(self.lb, self.ub, self.dim)
                    fitness[i] = self._eval(X[i])
                    if fitness[i] < best_fit:
                        best = X[i].copy()
                        best_fit = fitness[i]

            convergence.append(best_fit)

        return best, best_fit, convergence
