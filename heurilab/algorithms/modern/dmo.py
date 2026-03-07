"""DMO — Dwarf Mongoose Optimization (Agushaka et al., 2022)"""

import numpy as np
from heurilab.algorithms.base import _Base


class DMO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]
        peep = 2  # peeping intensity

        # Split population: alpha group, scouts, babysitters
        n_alpha = max(1, self.pop_size // 3)
        n_scout = max(1, self.pop_size // 3)

        tau = np.zeros(self.pop_size)  # sleeping mound counter

        for t in range(self.max_iter):
            C = 1 - t / self.max_iter  # control parameter
            phi = 0.5 + 0.5 * (1 - t / self.max_iter)  # foraging probability

            sorted_idx = np.argsort(fitness)

            # Alpha group — foraging near the best
            for idx in sorted_idx[:n_alpha]:
                new_X = X[idx] + C * peep * np.random.randn(self.dim) * (best - X[idx])
                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)
                if new_fit < fitness[idx]:
                    X[idx] = new_X
                    fitness[idx] = new_fit
                    tau[idx] = 0
                else:
                    tau[idx] += 1

            # Scout group — exploration
            for idx in sorted_idx[n_alpha:n_alpha + n_scout]:
                if np.random.rand() < phi:
                    # Exploration near random member
                    r = np.random.randint(self.pop_size)
                    new_X = X[idx] + C * np.random.randn(self.dim) * (X[r] - X[idx])
                else:
                    # Random exploration
                    new_X = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)
                if new_fit < fitness[idx]:
                    X[idx] = new_X
                    fitness[idx] = new_fit
                    tau[idx] = 0
                else:
                    tau[idx] += 1

            # Babysitter group — sleeping mound exchange
            for idx in sorted_idx[n_alpha + n_scout:]:
                if tau[idx] > 3:
                    # Exchange sleeping mound
                    X[idx] = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
                    fitness[idx] = self._eval(X[idx])
                    tau[idx] = 0
                else:
                    new_X = X[idx] + C * np.random.randn(self.dim) * (best - X[idx]) * phi
                    new_X = self._clip(new_X)
                    new_fit = self._eval(new_X)
                    if new_fit < fitness[idx]:
                        X[idx] = new_X
                        fitness[idx] = new_fit
                        tau[idx] = 0
                    else:
                        tau[idx] += 1

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
