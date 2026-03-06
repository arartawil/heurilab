"""SaDE — Self-adaptive Differential Evolution"""

import numpy as np
from heurilab.algorithms.base import _Base


class SaDE(_Base):
    def optimize(self):
        LP = 50  # Learning period

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        # Strategy probability and CR memory
        n_strat = 2
        ns = np.zeros(n_strat)  # Success counts
        nf = np.zeros(n_strat)  # Failure counts
        p_strat = np.ones(n_strat) / n_strat

        CR_memory = [[] for _ in range(n_strat)]
        CR_mean = np.full(n_strat, 0.5)

        convergence = [best_fit]

        for t in range(self.max_iter):
            if t >= LP and np.sum(ns + nf) > 0:
                p_strat = (ns + 0.01) / (ns + nf + 0.02)
                p_strat = p_strat / np.sum(p_strat)
                for s in range(n_strat):
                    if len(CR_memory[s]) > 0:
                        CR_mean[s] = np.median(CR_memory[s][-LP:])
                ns[:] = 0
                nf[:] = 0

            for i in range(self.pop_size):
                strat = np.random.choice(n_strat, p=p_strat)
                CR = np.clip(np.random.normal(CR_mean[strat], 0.1), 0, 1)
                F = np.clip(np.random.normal(0.5, 0.3), 0.01, 2.0)

                idxs = list(range(self.pop_size))
                idxs.remove(i)

                if strat == 0:
                    # rand/1/bin
                    r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                    mutant = X[r1] + F * (X[r2] - X[r3])
                else:
                    # current-to-best/1/bin
                    r1, r2 = np.random.choice(idxs, 2, replace=False)
                    mutant = X[i] + F * (best - X[i]) + F * (X[r1] - X[r2])

                mutant = self._clip(mutant)

                trial = X[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                trial_fit = self._eval(trial)

                if trial_fit <= fitness[i]:
                    X[i] = trial
                    fitness[i] = trial_fit
                    ns[strat] += 1
                    CR_memory[strat].append(CR)
                else:
                    nf[strat] += 1

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
