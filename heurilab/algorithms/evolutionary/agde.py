"""AGDE — Adaptive Guided Differential Evolution (Mohamed et al., 2019)"""

import numpy as np
from heurilab.algorithms.base import _Base


class AGDE(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            sorted_idx = np.argsort(fitness)
            n_sup = max(1, self.pop_size // 4)  # Superior group
            n_inf = max(1, self.pop_size // 4)  # Inferior group

            sup_idx = sorted_idx[:n_sup]
            inf_idx = sorted_idx[-n_inf:]
            mid_idx = sorted_idx[n_sup:-n_inf] if n_sup + n_inf < self.pop_size else sorted_idx[n_sup:]

            for i in range(self.pop_size):
                # Adaptive F and CR based on generation
                ratio = t / self.max_iter
                F = 0.1 + 0.9 * np.random.rand() * (1 - ratio)
                CR = 0.1 + 0.8 * np.random.rand()

                # Select from superior, middle, and inferior groups
                r_sup = np.random.choice(sup_idx)
                r_inf = np.random.choice(inf_idx)
                if len(mid_idx) > 0:
                    r_mid = np.random.choice(mid_idx)
                else:
                    r_mid = np.random.choice(sorted_idx)

                # Guided mutation: bias toward superior solutions
                mutant = X[i] + F * (X[r_sup] - X[r_inf]) + F * (X[r_sup] - X[r_mid])
                mutant = self._clip(mutant)

                # Binomial crossover
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
