"""SBOA — Secretary Bird Optimization Algorithm (Fu et al., 2024)"""

import numpy as np
from heurilab.algorithms.base import _Base


class SBOA(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            RB = np.random.randn(self.dim)  # random Brownian
            T_f = 1 - t / self.max_iter

            for i in range(self.pop_size):
                r = np.random.rand()

                if r < 0.5:
                    # Phase 1: Hunting (exploration)
                    # Secretary bird searches for prey (snakes)
                    j = np.random.randint(self.pop_size)
                    r1 = np.random.rand(self.dim)

                    # Attack strategy: stomp on prey
                    if fitness[j] < fitness[i]:
                        new_X = X[i] + r1 * (X[j] - 2 * X[i]) * T_f
                    else:
                        new_X = X[i] + r1 * (X[i] - X[j]) * T_f
                else:
                    # Phase 2: Escape from predators (exploitation)
                    r1 = np.random.rand(self.dim)

                    if np.random.rand() < 0.5:
                        # Flee toward best known safe position
                        K = np.random.choice([1, 2])
                        new_X = best + RB * (best - K * X[i]) * T_f
                    else:
                        # Evasive flight maneuver
                        step = np.random.rand() * 2 * np.pi
                        new_X = X[i] + np.cos(step) * r1 * (best - X[i]) * T_f

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
