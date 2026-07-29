"""SBO — Satin Bowerbird Optimizer"""

import numpy as np
from heurilab.algorithms.base import _Base, roulette_probabilities


class SBO(_Base):
    def optimize(self):
        z = 0.02  # Mutation probability
        alpha_sbo = 0.94  # Step size parameter

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            # Probability of each bower based on fitness
            max_fit = np.max(fitness)
            probs = roulette_probabilities(max_fit - fitness)

            for i in range(self.pop_size):
                new_X = X[i].copy()

                for j in range(self.dim):
                    if self.rng.random() < z:
                        # Mutation: random value
                        new_X[j] = self.rng.uniform(self.lb[j], self.ub[j])
                    else:
                        # Select an elite bower via roulette
                        k = self.rng.choice(self.pop_size, p=probs)
                        lam = alpha_sbo / (1 + probs[k])
                        new_X[j] = X[i, j] + lam * ((X[k, j] + best[j]) / 2.0 - X[i, j])

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
