"""CSA — Crow Search Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class CSA(_Base):
    def optimize(self):
        AP = 0.1  # Awareness probability
        fl = 2.0  # Flight length

        X = self._init_pop()
        memory = X.copy()  # Each crow's hidden food position
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])
        mem_fitness = fitness.copy()

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                j = np.random.randint(self.pop_size)  # Random crow to follow

                if np.random.rand() >= AP:
                    # Crow j doesn't notice: follow its memory
                    r = np.random.rand()
                    new_X = X[i] + r * fl * (memory[j] - X[i])
                else:
                    # Crow j notices: go to random position
                    new_X = np.random.uniform(self.lb, self.ub, self.dim)

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

                    # Update memory if new position is better
                    if new_fit < mem_fitness[i]:
                        memory[i] = new_X.copy()
                        mem_fitness[i] = new_fit

            min_idx = np.argmin(mem_fitness)
            if mem_fitness[min_idx] < best_fit:
                best = memory[min_idx].copy()
                best_fit = mem_fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
