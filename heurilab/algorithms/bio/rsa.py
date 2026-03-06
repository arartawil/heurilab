"""RSA — Reptile Search Algorithm (Abualigah et al., 2022)"""

import numpy as np
from heurilab.algorithms.base import _Base


class RSA(_Base):
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
            alpha = 2 * (1 - t / self.max_iter)  # exploration factor
            beta = 2 * (t / self.max_iter)  # exploitation factor

            ES = 2 * np.random.rand() - 1  # evolutionary sense

            for i in range(self.pop_size):
                r1 = np.random.rand()
                r_idx = np.random.randint(self.pop_size)

                if t < self.max_iter / 4:
                    # Phase 1: High walking
                    new_X = best * (1 - t / self.max_iter) + \
                            np.random.rand(self.dim) * (X[r_idx] - best) * np.random.rand()
                elif t < self.max_iter / 2:
                    # Phase 2: Belly walking
                    new_X = best * X[r_idx] * ES * np.random.rand(self.dim)
                elif t < 3 * self.max_iter / 4:
                    # Phase 3: Hunting coordination
                    new_X = best * alpha + X[r_idx] * np.random.rand() * beta
                else:
                    # Phase 4: Hunting cooperation
                    new_X = best * X[r_idx] * alpha * np.random.rand(self.dim)

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
