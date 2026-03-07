"""EVO — Energy Valley Optimizer (Azizi et al., 2023)"""

import numpy as np
from heurilab.algorithms.base import _Base


class EVO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            # Decay rate controls exploration vs exploitation
            decay = np.exp(-4 * (t / self.max_iter) ** 2)
            a = 2 * (1 - t / self.max_iter)  # linearly decreasing

            for i in range(self.pop_size):
                r = np.random.rand()

                if r < 0.5:
                    # Generation phase: particles fall into energy valley
                    # Potential energy drives movement toward best
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    distance = np.abs(best - X[i])
                    # Kinetic energy component
                    KE = 0.5 * decay * distance ** 2
                    new_X = X[i] + r1 * KE * np.sign(best - X[i]) + r2 * a * (best - X[i])
                else:
                    if np.random.rand() < decay:
                        # Transition phase: electron tunneling (exploration)
                        j = np.random.randint(self.pop_size)
                        sigma = np.abs(X[j] - X[i]) * decay + 1e-16
                        new_X = best + sigma * np.random.randn(self.dim)
                    else:
                        # Absorption phase: settling into valley (exploitation)
                        r1 = np.random.rand(self.dim)
                        C = 2 * r1 - 1
                        new_X = best + C * decay * (self.ub - self.lb) * 0.01

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
