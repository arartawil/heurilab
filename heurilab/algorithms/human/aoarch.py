"""AOArch — Archimedes Optimization Algorithm (Hashim et al., 2021)"""

import numpy as np
from heurilab.algorithms.base import _Base


class AOArch(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]
        lb = np.array(self.lb)
        ub = np.array(self.ub)

        # Density and volume
        den = np.random.rand(self.pop_size)
        vol = np.random.rand(self.pop_size)

        for t in range(self.max_iter):
            TF = np.exp((t - self.max_iter) / self.max_iter)  # Transfer factor
            d_flag = np.exp((t - self.max_iter) / self.max_iter) - (t / self.max_iter)

            den_new = den.copy()
            vol_new = vol.copy()

            for i in range(self.pop_size):
                # Update density and volume
                r_idx = np.random.randint(self.pop_size)
                den_new[i] = den[i] + np.random.rand() * (den[r_idx] - den[i])
                vol_new[i] = vol[i] + np.random.rand() * (vol[r_idx] - vol[i])

                if TF <= 0.5:
                    # Exploration phase
                    r1 = np.random.randint(self.pop_size)
                    new_X = X[i] + np.random.rand(self.dim) * (X[r1] - X[i]) * d_flag
                else:
                    # Exploitation phase
                    acc = (den_new[best_idx] + vol_new[best_idx] * np.random.rand(self.dim)) / \
                          (den_new[i] * vol_new[i] + 1e-16)
                    f = 2 * np.random.rand() - 1  # direction flag
                    new_X = best + f * TF * acc * (best - X[i])

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

            den = den_new
            vol = vol_new

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]
                best_idx = min_idx

            convergence.append(best_fit)

        return best, best_fit, convergence
