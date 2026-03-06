"""IMODE — Improved Multi-Operator Differential Evolution (Sallam et al., 2020)"""

import numpy as np
from heurilab.algorithms.base import _Base


class IMODE(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        # Operator counts and rewards
        n_ops = 3
        ns = np.ones(n_ops)
        nf = np.ones(n_ops)

        F = 0.5
        CR = 0.8

        for t in range(self.max_iter):
            # Adaptive probabilities based on success rates
            p_total = ns / (ns + nf + 1e-16)
            p_total = p_total / (np.sum(p_total) + 1e-16)

            ns_new = np.zeros(n_ops)
            nf_new = np.zeros(n_ops)

            for i in range(self.pop_size):
                # Select operator
                op = np.random.choice(n_ops, p=p_total)

                F_i = 0.1 + 0.9 * np.random.rand()
                CR_i = np.random.rand()

                idxs = list(range(self.pop_size))
                idxs.remove(i)

                if op == 0:
                    # DE/rand/1
                    r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                    mutant = X[r1] + F_i * (X[r2] - X[r3])
                elif op == 1:
                    # DE/current-to-best/1
                    r1, r2 = np.random.choice(idxs, 2, replace=False)
                    mutant = X[i] + F_i * (best - X[i]) + F_i * (X[r1] - X[r2])
                else:
                    # DE/rand-to-best/1
                    r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                    mutant = X[r1] + F_i * (best - X[r1]) + F_i * (X[r2] - X[r3])

                mutant = self._clip(mutant)

                trial = X[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR_i or j == j_rand:
                        trial[j] = mutant[j]

                trial_fit = self._eval(trial)

                if trial_fit <= fitness[i]:
                    X[i] = trial
                    fitness[i] = trial_fit
                    ns_new[op] += 1
                else:
                    nf_new[op] += 1

            # Weighted update
            ns = 0.5 * ns + 0.5 * ns_new + 1e-2
            nf = 0.5 * nf + 0.5 * nf_new + 1e-2

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
