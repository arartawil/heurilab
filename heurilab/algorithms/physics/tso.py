"""TSO — Transient Search Optimization (Qais et al., 2020)"""

import numpy as np
from heurilab.algorithms.base import _Base


class TSO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]
        lb = np.array(self.lb)
        ub = np.array(self.ub)

        for t in range(self.max_iter):
            T = 1 - t / self.max_iter  # Transient factor (decreases)

            for i in range(self.pop_size):
                r1 = np.random.rand()
                r2 = np.random.rand()

                if r1 < 0.5:
                    # Source current (exploitation) — damped oscillation toward best
                    omega = 2 * np.pi * np.random.rand()
                    damping = np.exp(-T * np.random.rand())
                    new_X = best + damping * np.cos(omega) * (X[i] - best)
                else:
                    # Transient response (exploration)
                    if r2 < 0.5:
                        # Over-damped
                        r3 = np.random.randint(self.pop_size)
                        new_X = X[r3] + T * np.random.randn(self.dim) * (ub - lb) * 0.1
                    else:
                        # Under-damped with oscillation
                        alpha_t = 2 * T * np.random.rand()
                        new_X = X[i] + alpha_t * (best - X[i]) * np.sin(2 * np.pi * np.random.rand())

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
