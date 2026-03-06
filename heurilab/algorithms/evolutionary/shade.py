"""SHADE — Success-History based Adaptive Differential Evolution"""

import numpy as np
from heurilab.algorithms.base import _Base


class SHADE(_Base):
    def optimize(self):
        H = self.pop_size  # History size
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.5)
        k = 0  # History index

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            S_CR, S_F, delta_f = [], [], []

            for i in range(self.pop_size):
                ri = np.random.randint(H)
                CR_i = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                F_i = min(1.0, max(0.01, M_F[ri] + 0.1 * np.random.standard_cauchy()))

                # current-to-pbest/1
                p = max(2, int(0.1 * self.pop_size))
                p_best_idx = np.argsort(fitness)[:p]
                x_pbest = X[np.random.choice(p_best_idx)]

                idxs = list(range(self.pop_size))
                idxs.remove(i)
                r1, r2 = np.random.choice(idxs, 2, replace=False)

                mutant = X[i] + F_i * (x_pbest - X[i]) + F_i * (X[r1] - X[r2])
                mutant = self._clip(mutant)

                # Binomial crossover
                trial = X[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR_i or j == j_rand:
                        trial[j] = mutant[j]

                trial_fit = self._eval(trial)

                if trial_fit <= fitness[i]:
                    if trial_fit < fitness[i]:
                        S_CR.append(CR_i)
                        S_F.append(F_i)
                        delta_f.append(abs(fitness[i] - trial_fit))
                    X[i] = trial
                    fitness[i] = trial_fit

            # Update history
            if len(S_CR) > 0:
                w = np.array(delta_f)
                w = w / (np.sum(w) + 1e-16)
                M_CR[k] = np.sum(w * np.array(S_CR))
                sf = np.array(S_F)
                M_F[k] = np.sum(w * sf ** 2) / (np.sum(w * sf) + 1e-16)
                k = (k + 1) % H

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
