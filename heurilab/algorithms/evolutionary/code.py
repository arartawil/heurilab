"""CoDE — Composite Differential Evolution"""

import numpy as np
from heurilab.algorithms.base import _Base


class CoDE(_Base):
    def optimize(self):
        # Three trial vector generation strategies with parameter pools
        F_pool = [1.0, 1.0, 0.8]
        CR_pool = [0.1, 0.9, 0.2]

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                trials = []
                trial_fits = []

                for s in range(3):
                    F = F_pool[s] * np.random.rand()
                    CR = CR_pool[s]

                    idxs = list(range(self.pop_size))
                    idxs.remove(i)

                    if s == 0:
                        # rand/1/bin
                        r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                        mutant = X[r1] + F * (X[r2] - X[r3])
                    elif s == 1:
                        # rand/2/bin
                        r1, r2, r3, r4, r5 = np.random.choice(idxs, 5, replace=False)
                        mutant = X[r1] + F * (X[r2] - X[r3]) + F * (X[r4] - X[r5])
                    else:
                        # current-to-rand/1
                        r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                        K = np.random.rand()
                        mutant = X[i] + K * (X[r1] - X[i]) + F * (X[r2] - X[r3])

                    mutant = self._clip(mutant)

                    # Crossover
                    trial = X[i].copy()
                    j_rand = np.random.randint(self.dim)
                    for j in range(self.dim):
                        if np.random.rand() < CR or j == j_rand:
                            trial[j] = mutant[j]

                    trial_fit = self._eval(trial)
                    trials.append(trial)
                    trial_fits.append(trial_fit)

                # Select the best trial vector
                best_trial = np.argmin(trial_fits)
                if trial_fits[best_trial] < fitness[i]:
                    X[i] = trials[best_trial]
                    fitness[i] = trial_fits[best_trial]

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
