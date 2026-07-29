"""HHO — Harris Hawks Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class HHO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        rabbit = X[best_idx].copy()
        rabbit_fit = fitness[best_idx]

        convergence = [rabbit_fit]

        for t in range(self.max_iter):
            E0 = 2 * self.rng.random() - 1
            J_factor = 2 * (1 - t / self.max_iter)
            E = E0 * J_factor

            for i in range(self.pop_size):
                q = self.rng.random()

                if abs(E) >= 1:  # Exploration
                    if q >= 0.5:
                        rand_idx = self.rng.integers(self.pop_size)
                        X[i] = X[rand_idx] - self.rng.random(self.dim) * np.abs(X[rand_idx] - 2 * self.rng.random(self.dim) * X[i])
                    else:
                        X[i] = rabbit - np.mean(X, axis=0) - self.rng.random(self.dim) * (self.ub - self.lb) * self.rng.random() + self.lb
                else:  # Exploitation
                    if q >= 0.5 and abs(E) >= 0.5:
                        D = np.abs(rabbit - X[i])
                        X[i] = rabbit - E * D
                    elif q >= 0.5 and abs(E) < 0.5:
                        D = np.abs(rabbit - X[i])
                        X[i] = rabbit - E * D
                    elif q < 0.5 and abs(E) >= 0.5:
                        J = 2 * (1 - self.rng.random())
                        Y = rabbit - E * np.abs(J * rabbit - X[i])
                        fit_Y = self._eval(self._clip(Y))
                        S = Y + self.rng.standard_normal(self.dim) * (self.ub - self.lb) * 0.01
                        fit_S = self._eval(self._clip(S))
                        X[i] = Y if fit_Y < fit_S else S
                    else:
                        J = 2 * (1 - self.rng.random())
                        Y = rabbit - E * np.abs(J * rabbit - np.mean(X, axis=0))
                        fit_Y = self._eval(self._clip(Y))
                        S = Y + self.rng.standard_normal(self.dim) * (self.ub - self.lb) * 0.01
                        fit_S = self._eval(self._clip(S))
                        X[i] = Y if fit_Y < fit_S else S

                X[i] = self._clip(X[i])

            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < rabbit_fit:
                rabbit = X[min_idx].copy()
                rabbit_fit = fitness[min_idx]

            convergence.append(rabbit_fit)

        return rabbit, rabbit_fit, convergence
