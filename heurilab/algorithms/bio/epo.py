"""EPO — Emperor Penguin Optimizer"""

import numpy as np
from heurilab.algorithms.base import _Base


class EPO(_Base):
    def optimize(self):
        M = 2  # Movement parameter
        f_param = 2  # Controls convergence

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            T = 0 if t < self.max_iter / 2 else 1  # Temperature
            ratio = t / self.max_iter

            # Huddle boundary
            R = np.random.rand()
            if T == 0:
                # Exploration: large radius
                P = np.random.uniform(2, 3)
            else:
                # Exploitation: small radius
                P = np.random.uniform(0, 2)

            for i in range(self.pop_size):
                r = np.random.rand(self.dim)
                theta = np.random.rand() * 2 * np.pi
                A = (M * (T + P) * r - T) * np.cos(theta)

                # Social forces
                S = np.abs(f_param * np.exp(-ratio) - np.exp(-1))

                # Distance to emperor (best)
                D = np.abs(S * best - X[i])

                new_X = best - A * D

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
