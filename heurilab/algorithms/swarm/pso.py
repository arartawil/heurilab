"""PSO — Particle Swarm Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class PSO(_Base):
    def optimize(self):
        w_max, w_min = 0.9, 0.2
        c1, c2 = 2.0, 2.0

        X = self._init_pop()
        V = np.zeros((self.pop_size, self.dim))
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        pBest = X.copy()
        pBest_fit = fitness.copy()

        gBest_idx = np.argmin(fitness)
        gBest = X[gBest_idx].copy()
        gBest_fit = fitness[gBest_idx]

        convergence = [gBest_fit]

        for t in range(self.max_iter):
            w = w_max - (w_max - w_min) * t / self.max_iter
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)

            V = w * V + c1 * r1 * (pBest - X) + c2 * r2 * (gBest - X)
            X = self._clip(X + V)

            for i in range(self.pop_size):
                fit = self._eval(X[i])
                if fit < pBest_fit[i]:
                    pBest[i] = X[i].copy()
                    pBest_fit[i] = fit
                if fit < gBest_fit:
                    gBest = X[i].copy()
                    gBest_fit = fit

            convergence.append(gBest_fit)

        return gBest, gBest_fit, convergence
