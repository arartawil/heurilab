"""QLA — Q-Learning-based Optimization Algorithm"""

import numpy as np
from heurilab.algorithms.base import _Base


class QLA(_Base):
    def optimize(self):
        alpha_q = 0.1   # Learning rate
        gamma_q = 0.9   # Discount factor
        epsilon = 0.3   # Exploration rate
        n_actions = 4   # Number of actions: exploit, explore, random, restart

        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        # Q-table: one entry per individual, per action
        Q = np.zeros((self.pop_size, n_actions))

        convergence = [best_fit]

        for t in range(self.max_iter):
            eps_decay = epsilon * (1 - t / self.max_iter)

            for i in range(self.pop_size):
                # Select action (epsilon-greedy)
                if np.random.rand() < eps_decay:
                    action = np.random.randint(n_actions)
                else:
                    action = np.argmax(Q[i])

                if action == 0:
                    # Exploit: move toward best
                    r = np.random.rand(self.dim)
                    new_X = X[i] + r * (best - X[i])
                elif action == 1:
                    # Explore: Lévy-like jump
                    step = np.random.randn(self.dim) * (self.ub - self.lb) * 0.1
                    new_X = X[i] + step
                elif action == 2:
                    # Random neighbor interaction
                    j = np.random.randint(self.pop_size)
                    new_X = X[i] + np.random.rand(self.dim) * (X[j] - X[i])
                else:
                    # Restart from random position
                    new_X = np.random.uniform(self.lb, self.ub, self.dim)

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                # Reward
                reward = fitness[i] - new_fit  # Positive = improvement

                # Q-learning update
                Q[i, action] = Q[i, action] + alpha_q * (
                    reward + gamma_q * np.max(Q[i]) - Q[i, action]
                )

                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
