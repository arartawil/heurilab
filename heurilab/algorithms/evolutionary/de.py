"""DE — Differential Evolution (DE/rand/1/bin)"""

import numpy as np
from heurilab.algorithms.base import _Base


class DE(_Base):
    def optimize(self):
        F, CR = 0.8, 0.9
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                # Select 3 distinct individuals != i
                idxs = list(range(self.pop_size))
                idxs.remove(i)
                a, b, c = np.random.choice(idxs, 3, replace=False)

                # Mutation
                mutant = self._clip(X[a] + F * (X[b] - X[c]))

                # Crossover
                trial = X[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                trial = self._clip(trial)
                trial_fit = self._eval(trial)

                if trial_fit <= fitness[i]:
                    X[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_fit:
                        best = trial.copy()
                        best_fit = trial_fit

            convergence.append(best_fit)

        return best, best_fit, convergence
