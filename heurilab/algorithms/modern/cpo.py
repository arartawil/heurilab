"""CPO — Crested Porcupine Optimizer (Abdel-Basset et al., 2024)"""

import numpy as np
from heurilab.algorithms.base import _Base


class CPO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            U = np.random.rand()  # threat level
            T_f = (1 - t / self.max_iter)  # time factor
            S = 2 * np.random.rand() - 1  # defense strategy factor

            for i in range(self.pop_size):
                r = np.random.rand()

                if U < 0.3:
                    # Defense mode 1: Sight (exploration — quill erection display)
                    r1 = np.random.rand(self.dim)
                    j = np.random.randint(self.pop_size)
                    delta = np.abs(X[j] - X[i])
                    new_X = X[i] + r1 * delta * np.sign(np.random.rand(self.dim) - 0.5) * T_f

                elif U < 0.6:
                    # Defense mode 2: Sound (exploitation — quill vibration)
                    r1 = np.random.rand(self.dim)
                    F = 2 * T_f * np.random.randn(self.dim)
                    new_X = best + F * (best - X[i]) * r1

                else:
                    # Defense mode 3: Physical attack (strong exploitation — quill release)
                    r1 = np.random.rand(self.dim)
                    if r < 0.5:
                        # Direct quill shooting toward predator
                        new_X = best - S * r1 * (best - X[i]) * T_f ** 2
                    else:
                        # Cyclic defense pattern
                        theta = 2 * np.pi * np.random.rand(self.dim)
                        new_X = best + np.abs(best - X[i]) * np.cos(theta) * T_f

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
