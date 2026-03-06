"""WDO — Wind Driven Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class WDO(_Base):
    def optimize(self):
        alpha = 0.4   # Friction coefficient
        g = 0.2       # Gravitational constant
        RT = 3.0      # RT constant
        c = 0.4       # Coriolis effect

        X = self._init_pop()
        V = np.zeros((self.pop_size, self.dim))
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            rank = np.argsort(fitness)
            pressure = np.argsort(np.argsort(fitness)).astype(float)
            pressure = pressure / (self.pop_size - 1) if self.pop_size > 1 else pressure

            for i in range(self.pop_size):
                # Pick a random other air parcel
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                # Coriolis-like cross-dimensional effect
                other_dims = np.roll(V[i], 1)

                V[i] = ((1 - alpha) * V[i]
                         - g * X[i]
                         + np.abs(1.0 / (rank[i] + 1) - 1.0 / (rank[j] + 1)) * RT * (X[j] - X[i])
                         + c * other_dims / (rank[i] + 1))

                V[i] = np.clip(V[i], -(self.ub - self.lb), self.ub - self.lb)
                X[i] = self._clip(X[i] + V[i])
                fitness[i] = self._eval(X[i])

                if fitness[i] < best_fit:
                    best = X[i].copy()
                    best_fit = fitness[i]

            convergence.append(best_fit)

        return best, best_fit, convergence
