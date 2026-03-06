"""GJO — Golden Jackal Optimization (Chopra & Ansari, 2022)"""

import numpy as np
from heurilab.algorithms.base import _Base


class GJO(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        sorted_idx = np.argsort(fitness)
        male = X[sorted_idx[0]].copy()   # Best — male jackal
        male_fit = fitness[sorted_idx[0]]
        female = X[sorted_idx[1]].copy()  # Second best — female jackal
        female_fit = fitness[sorted_idx[1]]

        best = male.copy()
        best_fit = male_fit

        convergence = [best_fit]
        lb = np.array(self.lb)
        ub = np.array(self.ub)

        for t in range(self.max_iter):
            E1 = 1.5 * (1 - t / self.max_iter)  # Escaping energy

            for i in range(self.pop_size):
                E0 = 2 * np.random.rand() - 1
                E = E1 * E0  # Escaping factor

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                if abs(E) >= 1:
                    # Exploration: search for prey
                    D_male = abs(r1 * male - X[i])
                    D_female = abs(r2 * female - X[i])
                    X1 = male - E * D_male
                    X2 = female - E * D_female
                else:
                    # Exploitation: attack prey
                    D_male = abs(male - X[i])
                    D_female = abs(female - X[i])
                    X1 = male - E * D_male
                    X2 = female - E * D_female

                new_X = (X1 + X2) / 2
                new_X = self._clip(new_X)
                new_fit = self._eval(new_X)

                if new_fit < fitness[i]:
                    X[i] = new_X
                    fitness[i] = new_fit

            # Update male and female
            for i in range(self.pop_size):
                if fitness[i] < male_fit:
                    female = male.copy()
                    female_fit = male_fit
                    male = X[i].copy()
                    male_fit = fitness[i]
                elif fitness[i] < female_fit:
                    female = X[i].copy()
                    female_fit = fitness[i]

            best = male.copy()
            best_fit = male_fit

            convergence.append(best_fit)

        return best, best_fit, convergence
