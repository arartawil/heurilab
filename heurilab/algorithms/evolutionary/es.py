"""ES — Evolution Strategy (mu,lambda)-ES"""

import numpy as np
from heurilab.algorithms.base import _Base


class ES(_Base):
    def optimize(self):
        mu = self.pop_size
        lam = mu * 7  # lambda = 7*mu

        X = self._init_pop()
        sigma = np.full((mu, self.dim), 0.3 * (self.ub - self.lb))
        fitness = np.array([self._eval(X[i]) for i in range(mu)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]
        tau = 1 / np.sqrt(2 * self.dim)
        tau_prime = 1 / np.sqrt(2 * np.sqrt(self.dim))

        for t in range(self.max_iter):
            offspring_X = []
            offspring_sigma = []
            offspring_fit = []

            for _ in range(lam):
                parent = np.random.randint(mu)
                s_new = sigma[parent] * np.exp(tau_prime * np.random.randn() + tau * np.random.randn(self.dim))
                x_new = self._clip(X[parent] + s_new * np.random.randn(self.dim))
                offspring_X.append(x_new)
                offspring_sigma.append(s_new)
                offspring_fit.append(self._eval(x_new))

            offspring_X = np.array(offspring_X)
            offspring_sigma = np.array(offspring_sigma)
            offspring_fit = np.array(offspring_fit)

            # (mu,lambda) selection
            sorted_idx = np.argsort(offspring_fit)[:mu]
            X = offspring_X[sorted_idx]
            sigma = offspring_sigma[sorted_idx]
            fitness = offspring_fit[sorted_idx]

            if fitness[0] < best_fit:
                best = X[0].copy()
                best_fit = fitness[0]

            convergence.append(best_fit)

        return best, best_fit, convergence
