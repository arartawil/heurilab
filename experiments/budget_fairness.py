"""
Equal-iteration versus equal-evaluation-budget comparison.

Runs the same ten algorithms twice over the CEC 2017 suite: once with the
iteration count held constant, as comparative studies conventionally do, and
once with the number of objective evaluations held constant instead. If the two
rankings differ, the conventional protocol has been reporting the budget rather
than the algorithm.

The ten algorithms span the whole cost range of the registry, from harmony
search at one evaluation per iteration to the firefly algorithm at about four
hundred and thirty-five.

Usage
-----
    python experiments/budget_fairness.py

    # options
    python experiments/budget_fairness.py --runs 30        # thorough version
    python experiments/budget_fairness.py --dim 10         # cheaper
    python experiments/budget_fairness.py --jobs 4         # cap the workers

Progress is written to results_budget_fairness.csv after every function, so the
script can be stopped and restarted without losing completed work.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heurilab.algorithms import ALL_ALGORITHMS                 # noqa: E402
from heurilab.core.budget import calibrate_iterations          # noqa: E402
from heurilab.core.opfunu_suites import get_cec2017_opfunu_suite  # noqa: E402

#: Chosen to span the registry's evaluation-cost range, not by expected performance.
ALGORITHMS = ["HS", "LSHADE", "PSO", "GWO", "HHO", "BFO", "TSA", "SOS", "ES", "FA"]

OUT = "results_budget_fairness.csv"
COLUMNS = ["protocol", "function", "algorithm", "run", "best", "n_fes",
           "max_iter", "seconds"]


def one_run(name, cls, bench, dim, pop, max_iter, max_fes, seed):
    """A single optimisation. Returns the row that describes it."""
    algo = cls(pop_size=pop, dim=dim, lb=bench.lb, ub=bench.ub,
               max_iter=max_iter, obj_func=bench.obj_func,
               seed=seed, max_fes=max_fes)
    t0 = time.perf_counter()
    _, best, _ = algo.optimize()
    return float(best), int(algo.n_fes), int(max_iter), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=30)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--iters", type=int, default=500, help="equal-iteration budget")
    ap.add_argument("--fes", type=int, default=60000, help="equal-evaluation budget")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    registry = dict(ALL_ALGORITHMS)
    missing = [a for a in ALGORITHMS if a not in registry]
    if missing:
        sys.exit(f"not in the registry: {missing}")

    suite = get_cec2017_opfunu_suite(ndim=args.dim)
    benches = list(suite.benchmarks)

    # ── calibrate once: each algorithm's iteration count for the FE budget ──
    print(f"calibrating {len(ALGORITHMS)} algorithms for a {args.fes:,}-evaluation budget")
    calibrated = {}
    for name in ALGORITHMS:
        iters, per_iter = calibrate_iterations(registry[name], args.fes,
                                               pop_size=args.pop, dim=args.dim)
        calibrated[name] = iters
        print(f"   {name:7s} {per_iter:7.1f} evals/iter  ->  max_iter = {iters}")

    # ── resume: skip any (protocol, function) block already on disk ──
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.protocol, prev.function))
        print(f"\nresuming: {len(done)} of {2 * len(benches)} blocks already complete")
    else:
        pd.DataFrame(columns=COLUMNS).to_csv(OUT, index=False)

    try:
        from joblib import Parallel, delayed
        runner = Parallel(n_jobs=args.jobs, prefer="processes")
    except ImportError:
        runner = None
        print("joblib not installed - running serially (pip install joblib to speed up)")

    total = 2 * len(benches)
    block = 0
    t_start = time.time()

    for protocol in ("equal_iterations", "equal_evaluations"):
        for bench in benches:
            block += 1
            if (protocol, bench.name) in done:
                continue
            jobs = []
            for name in ALGORITHMS:
                cls = registry[name]
                if protocol == "equal_iterations":
                    max_iter, max_fes = args.iters, None
                else:
                    max_iter, max_fes = calibrated[name], args.fes
                for r in range(args.runs):
                    seed = int(np.random.SeedSequence(
                        [args.seed, abs(hash(bench.name)) % (2 ** 31),
                         abs(hash(name)) % (2 ** 31), r]).generate_state(1)[0])
                    jobs.append((name, cls, max_iter, max_fes, seed, r))

            t0 = time.time()
            if runner is not None:
                out = runner(delayed(one_run)(n, c, bench, args.dim, args.pop,
                                              mi, mf, s)
                             for (n, c, mi, mf, s, _) in jobs)
            else:
                out = [one_run(n, c, bench, args.dim, args.pop, mi, mf, s)
                       for (n, c, mi, mf, s, _) in jobs]

            rows = [{"protocol": protocol, "function": bench.name, "algorithm": n,
                     "run": r, "best": b, "n_fes": f, "max_iter": mi,
                     "seconds": sec}
                    for (n, _, _, _, _, r), (b, f, mi, sec) in zip(jobs, out)]
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=False, index=False)

            elapsed = time.time() - t_start
            print(f"[{block:3d}/{total}] {protocol:17s} {bench.name:6s} "
                  f"{time.time() - t0:6.1f}s   elapsed {elapsed / 60:5.1f} min")

    print(f"\ndone in {(time.time() - t_start) / 60:.1f} minutes -> {OUT}")
    summarise()


def summarise():
    """Friedman ranks under each protocol, and the movement between them."""
    from scipy.stats import rankdata, spearmanr
    df = pd.read_csv(OUT)
    print("\n" + "=" * 66)
    ranks = {}
    for protocol, sub in df.groupby("protocol"):
        piv = sub.pivot_table(index="function", columns="algorithm",
                              values="best", aggfunc="mean")
        r = pd.DataFrame(np.apply_along_axis(rankdata, 1, piv.values),
                         index=piv.index, columns=piv.columns).mean()
        ranks[protocol] = r.sort_values()
        print(f"\n{protocol}  (mean Friedman rank, lower is better)")
        for i, (a, v) in enumerate(ranks[protocol].items(), 1):
            print(f"   {i:2d}. {a:7s} {v:5.2f}")

    if len(ranks) == 2:
        a, b = "equal_iterations", "equal_evaluations"
        oa = {n: i for i, n in enumerate(ranks[a].index, 1)}
        ob = {n: i for i, n in enumerate(ranks[b].index, 1)}
        common = sorted(oa, key=lambda n: oa[n])
        rho, p = spearmanr([ranks[a][n] for n in common],
                           [ranks[b][n] for n in common])
        print(f"\n{'algorithm':10s} {'equal-iter':>11s} {'equal-FE':>10s}   move")
        for n in common:
            print(f"{n:10s} {oa[n]:11d} {ob[n]:10d}   {ob[n] - oa[n]:+d}")
        print(f"\nSpearman rho between the two protocols: {rho:.3f}  (p = {p:.2e})")
        worst = max(common, key=lambda n: abs(ob[n] - oa[n]))
        print(f"largest movement: {worst}  {oa[worst]} -> {ob[worst]}")


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summarise()
    else:
        main()
