"""BBO — Biogeography-Based Optimization"""

import numpy as np
from heurilab.algorithms.base import _Base


class BBO(_Base):
    def optimize(self):
        mutation_rate = 0.005

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        # Immigration and emigration rates
        rank = np.argsort(np.argsort(fitness))
        lambdas = rank / (self.pop_size - 1) if self.pop_size > 1 else np.zeros(self.pop_size)
        mus = 1 - lambdas

        for t in range(self.max_iter):
            new_X = X.copy()

            for i in range(self.pop_size):
                for j in range(self.dim):
                    # Immigration
                    if np.random.rand() < lambdas[i]:
                        probs = mus / (np.sum(mus) + 1e-16)
                        source = np.random.choice(self.pop_size, p=probs)
                        new_X[i, j] = X[source, j]

                    # Mutation
                    if np.random.rand() < mutation_rate:
                        new_X[i, j] = np.random.uniform(self.lb[j], self.ub[j])

            new_X = np.array([self._clip(new_X[i]) for i in range(self.pop_size)])
            new_fitness = np.array([self._eval(new_X[i]) for i in range(self.pop_size)])

            # Elitism: keep the best from old and new
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    X[i] = new_X[i]
                    fitness[i] = new_fitness[i]

            # Update ranks
            rank = np.argsort(np.argsort(fitness))
            lambdas = rank / (self.pop_size - 1) if self.pop_size > 1 else np.zeros(self.pop_size)
            mus = 1 - lambdas

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
