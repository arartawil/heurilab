"""CMA — Covariance Matrix Adaptation Evolution Strategy (CMA-ES)"""

import numpy as np
from heurilab.algorithms.base import _Base


class CMA(_Base):
    def optimize(self):
        n = self.dim
        lam = self.pop_size
        mu = lam // 2

        # Recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size control
        sigma = 0.3 * np.mean(self.ub - self.lb)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
        E_N = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Covariance matrix adaptation
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # State
        mean = np.random.uniform(self.lb, self.ub)
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)

        best = mean.copy()
        best_fit = self._eval(mean)
        convergence = [best_fit]

        for t in range(self.max_iter):
            # Sample population
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(n)
                sqrt_C = np.eye(n)

            Z = np.random.randn(lam, n)
            X = np.array([self._clip(mean + sigma * sqrt_C @ z) for z in Z])
            fitness = np.array([self._eval(X[i]) for i in range(lam)])

            # Sort by fitness
            order = np.argsort(fitness)
            X = X[order]
            Z = Z[order]
            fitness = fitness[order]

            if fitness[0] < best_fit:
                best = X[0].copy()
                best_fit = fitness[0]

            # Recombination
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * X[:mu], axis=0)

            # Cumulation
            invsqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * invsqrt_C @ (mean - old_mean) / sigma
            hs = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (t + 1))) < (1.4 + 2 / (n + 1)) * E_N else 0

            pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma

            # Covariance matrix update
            artmp = (X[:mu] - old_mean) / sigma
            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C)
                 + cmu * (weights[:, None] * artmp).T @ artmp)

            # Step size update
            sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / E_N - 1))

            convergence.append(best_fit)

        return best, best_fit, convergence
