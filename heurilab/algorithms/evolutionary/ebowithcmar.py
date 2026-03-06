"""EBOwithCMAR — Enhanced BO with Covariance Matrix Adaptation Restart (Kumar et al., 2019)"""

import numpy as np
from heurilab.algorithms.base import _Base


class EBOwithCMAR(_Base):
    def optimize(self):
        H = 5
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.5)
        k = 0

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        # Covariance matrix init
        C = np.eye(self.dim) * 0.01

        for t in range(self.max_iter):
            S_CR, S_F, delta_f = [], [], []

            # Adaptive exploitation probability
            p_exploit = 0.5 * (1 + t / self.max_iter)

            for i in range(self.pop_size):
                ri = np.random.randint(H)
                CR_i = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                F_i = min(1.0, max(0.01, M_F[ri] + 0.1 * np.random.standard_cauchy()))

                if np.random.rand() < p_exploit:
                    # CMA-based perturbation
                    try:
                        z = np.random.multivariate_normal(np.zeros(self.dim), C)
                    except np.linalg.LinAlgError:
                        z = np.random.randn(self.dim) * 0.1
                    mutant = best + F_i * z
                else:
                    # DE/current-to-best/1
                    idxs = list(range(self.pop_size))
                    idxs.remove(i)
                    r1, r2 = np.random.choice(idxs, 2, replace=False)
                    mutant = X[i] + F_i * (best - X[i]) + F_i * (X[r1] - X[r2])

                mutant = self._clip(mutant)

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

            # Update parameter memory
            if len(S_CR) > 0:
                w = np.array(delta_f)
                w = w / (np.sum(w) + 1e-16)
                M_CR[k] = np.sum(w * np.array(S_CR))
                sf = np.array(S_F)
                M_F[k] = np.sum(w * sf ** 2) / (np.sum(w * sf) + 1e-16)
                k = (k + 1) % H

            # Update covariance from elite
            n_elite = max(2, self.pop_size // 5)
            elite_idx = np.argsort(fitness)[:n_elite]
            elite = X[elite_idx]
            mean_elite = np.mean(elite, axis=0)
            diff = elite - mean_elite
            C = (diff.T @ diff) / (n_elite - 1 + 1e-16) + 1e-6 * np.eye(self.dim)

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
