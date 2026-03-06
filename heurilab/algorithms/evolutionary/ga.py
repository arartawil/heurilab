"""GA — Genetic Algorithm (real-coded)"""

import numpy as np
from heurilab.algorithms.base import _Base


class GA(_Base):
    def optimize(self):
        pc, pm = 0.9, 0.1
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            # Tournament selection
            new_pop = np.empty_like(X)
            for i in range(self.pop_size):
                a, b = np.random.randint(0, self.pop_size, 2)
                new_pop[i] = X[a].copy() if fitness[a] < fitness[b] else X[b].copy()

            # SBX crossover
            for i in range(0, self.pop_size - 1, 2):
                if np.random.rand() < pc:
                    eta = 20
                    u = np.random.rand(self.dim)
                    beta_q = np.where(u <= 0.5,
                                      (2 * u) ** (1 / (eta + 1)),
                                      (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
                    c1 = 0.5 * ((1 + beta_q) * new_pop[i] + (1 - beta_q) * new_pop[i + 1])
                    c2 = 0.5 * ((1 - beta_q) * new_pop[i] + (1 + beta_q) * new_pop[i + 1])
                    new_pop[i] = self._clip(c1)
                    new_pop[i + 1] = self._clip(c2)

            # Polynomial mutation
            for i in range(self.pop_size):
                for j in range(self.dim):
                    if np.random.rand() < pm:
                        eta_m = 20
                        r = np.random.rand()
                        delta = (2 * r) ** (1 / (eta_m + 1)) - 1 if r < 0.5 else 1 - (2 * (1 - r)) ** (1 / (eta_m + 1))
                        new_pop[i, j] += delta * (self.ub[j] - self.lb[j])
                        new_pop[i, j] = np.clip(new_pop[i, j], self.lb[j], self.ub[j])

            X = new_pop
            fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            # Elitism: replace worst with best
            worst_idx = np.argmax(fitness)
            X[worst_idx] = best.copy()
            fitness[worst_idx] = best_fit

            convergence.append(best_fit)

        return best, best_fit, convergence
