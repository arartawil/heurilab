"""POA — Political Optimizer Algorithm (Askari et al., 2020)"""

import numpy as np
from heurilab.algorithms.base import _Base


class POA(_Base):
    def optimize(self):
        n_parties = 5
        members_per_party = self.pop_size // n_parties
        total = members_per_party * n_parties

        X = self._init_pop()
        if total < self.pop_size:
            X = X[:total]
        fitness = np.array([self._eval(X[i]) for i in range(total)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]
        lb = np.array(self.lb)
        ub = np.array(self.ub)

        for t in range(self.max_iter):
            # Assign parties
            parties = [list(range(p * members_per_party, (p + 1) * members_per_party)) for p in range(n_parties)]

            # Find party leaders (best in each party)
            leaders = []
            for party in parties:
                party_fit = [fitness[m] for m in party]
                leader = party[np.argmin(party_fit)]
                leaders.append(leader)

            # Phase 1: Constituency allocation operator
            for p, party in enumerate(parties):
                leader = leaders[p]
                for m in party:
                    if m == leader:
                        continue
                    r = np.random.rand()
                    new_X = X[m] + r * (X[leader] - X[m])
                    new_X = self._clip(new_X)
                    new_fit = self._eval(new_X)
                    if new_fit < fitness[m]:
                        X[m] = new_X
                        fitness[m] = new_fit

            # Phase 2: Party switching
            for p, party in enumerate(parties):
                for m in party:
                    if np.random.rand() < 0.2 * (1 - t / self.max_iter):
                        other_leader = leaders[np.random.randint(n_parties)]
                        r = np.random.rand()
                        new_X = X[m] + r * (X[other_leader] - X[m])
                        new_X = self._clip(new_X)
                        new_fit = self._eval(new_X)
                        if new_fit < fitness[m]:
                            X[m] = new_X
                            fitness[m] = new_fit

            # Phase 3: Election campaign — move toward global best
            for i in range(total):
                r = np.random.rand()
                new_X = X[i] + r * (best - X[i])
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
