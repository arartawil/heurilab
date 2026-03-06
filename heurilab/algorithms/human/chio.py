"""CHIO — Coronavirus Herd Immunity Optimizer (Al-Betar et al., 2020)"""

import numpy as np
from heurilab.algorithms.base import _Base


class CHIO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        # Status: 0=susceptible, 1=infected, 2=immune
        status = np.zeros(self.pop_size, dtype=int)
        # Initially infect some
        n_infected = max(1, self.pop_size // 5)
        infected_idx = np.random.choice(self.pop_size, n_infected, replace=False)
        status[infected_idx] = 1

        HI_rate = 0.7  # herd immunity rate
        BR_max = 0.01  # max spreading rate

        for t in range(self.max_iter):
            spreading_rate = BR_max * (1 - t / self.max_iter)

            for i in range(self.pop_size):
                if status[i] == 2:  # immune — skip
                    continue

                new_X = X[i].copy()

                for j in range(self.dim):
                    r = np.random.rand()
                    r_idx = np.random.randint(self.pop_size)

                    if status[r_idx] == 1:  # infected neighbor
                        new_X[j] = X[i][j] + r * (X[i][j] - X[r_idx][j])
                    elif status[r_idx] == 2:  # immune neighbor
                        new_X[j] = X[i][j] + r * (X[r_idx][j] - X[i][j])
                    else:  # susceptible
                        new_X[j] = X[i][j] + r * (best[j] - X[i][j])

                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit
                    status[i] = 2  # recovered/immune
                else:
                    if np.random.rand() < spreading_rate:
                        status[i] = 1  # becomes infected

            # Check herd immunity
            immune_ratio = np.sum(status == 2) / self.pop_size
            if immune_ratio >= HI_rate:
                # Reset some immune to susceptible
                immune_idxs = np.where(status == 2)[0]
                n_reset = max(1, len(immune_idxs) // 3)
                reset_idxs = np.random.choice(immune_idxs, n_reset, replace=False)
                status[reset_idxs] = 0

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fit:
                best = X[min_idx].copy()
                best_fit = fitness[min_idx]

            convergence.append(best_fit)

        return best, best_fit, convergence
