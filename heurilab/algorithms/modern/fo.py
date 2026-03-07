"""FO — Fox Optimizer (Mohammed & Rashid, 2024)"""

import numpy as np
from heurilab.algorithms.base import _Base


class FO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            a = 2 * (1 - t / self.max_iter)  # decreasing parameter
            p_explore = 1 - t / self.max_iter

            for i in range(self.pop_size):
                r = np.random.rand()

                if r < p_explore:
                    # Phase 1: Exploration — fox searching for prey
                    if np.random.rand() < 0.5:
                        # Walking around territory
                        j = np.random.randint(self.pop_size)
                        r1 = np.random.rand(self.dim)
                        new_X = X[i] + a * r1 * (X[j] - X[i])
                    else:
                        # Hearing-based exploration
                        dist = np.abs(best - X[i])
                        angle = 2 * np.pi * np.random.rand(self.dim)
                        new_X = X[i] + dist * np.cos(angle) * a
                else:
                    # Phase 2: Exploitation — fox stalking and pouncing
                    if np.random.rand() < 0.5:
                        # Stalking: slowly approach prey (best)
                        r1 = np.random.rand(self.dim)
                        jump_strength = 2 * np.random.rand() * (1 - t / self.max_iter)
                        new_X = best - jump_strength * r1 * (best - X[i])
                    else:
                        # Pouncing: leap toward prey
                        tt = np.random.rand()
                        sp = best / (X[i] + 1e-16)
                        sp = np.clip(sp, -10, 10)
                        new_X = best * tt + (1 - tt) * sp * X[i] * (1 - t / self.max_iter)

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
