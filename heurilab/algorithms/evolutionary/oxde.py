"""OXDE — Opposition-based Learning Differential Evolution"""

import numpy as np
from heurilab.algorithms.base import _Base


class OXDE(_Base):
    def optimize(self):
        F, CR = 0.5, 0.9
        Jr = 0.3  # Jumping rate

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            # Opposition-based population jump
            if np.random.rand() < Jr:
                a = np.min(X, axis=0)
                b = np.max(X, axis=0)
                opp_X = a + b - X
                opp_X = np.array([self._clip(opp_X[i]) for i in range(self.pop_size)])
                opp_fitness = np.array([self._eval(opp_X[i]) for i in range(self.pop_size)])

                # Merge and select best pop_size
                combined_X = np.vstack([X, opp_X])
                combined_fit = np.concatenate([fitness, opp_fitness])
                sel = np.argsort(combined_fit)[:self.pop_size]
                X = combined_X[sel]
                fitness = combined_fit[sel]

            # Standard DE/rand/1/bin
            for i in range(self.pop_size):
                idxs = list(range(self.pop_size))
                idxs.remove(i)
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

                mutant = self._clip(X[r1] + F * (X[r2] - X[r3]))

                trial = X[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                trial_fit = self._eval(trial)
                if trial_fit <= fitness[i]:
                    X[i] = trial
                    fitness[i] = trial_fit

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
