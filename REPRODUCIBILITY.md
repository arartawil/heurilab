# Reproducibility demonstration — HeuriLab v2.3.0

Evidence for the reproducibility claim, to accompany the main results.

## Setup

| | |
|---|---|
| Algorithms | PSO, GWO, DE, WOA |
| Benchmarks | CEC 2017 F1, F3, F4, F5, F6 (opfunu), d = 10 |
| Population | 20 |
| Budget | 10,000 function evaluations per run |
| Independent runs | 10 per algorithm–function pair (200 runs per campaign) |
| Campaign seed | 42 (7 for the control) |

Four campaigns were executed: the same seeded configuration twice, the same
configuration again across four parallel workers, and a fourth under a
different seed as a control.

## Result

SHA-256 over the recorded results — best fitness, per-run seed, evaluations
consumed and the full convergence trace:

| campaign | digest |
|---|---|
| A — seed 42, serial | `3eaadeddc45f7a88` |
| B — seed 42, serial (repeat) | `3eaadeddc45f7a88` |
| C — seed 42, `n_jobs=4` | `3eaadeddc45f7a88` |
| D — seed 7, serial | `2176ca858ad2cabc` |

- **A ≡ B.** Re-executing an identical configuration reproduces every number.
- **A ≡ C.** Parallelism changes wall-clock only. Each run's seed is derived
  from `(campaign seed, function, algorithm, run index)` before any work is
  dispatched, so execution order cannot influence the outcome.
- **A ≠ D.** A different campaign seed produces different results, confirming
  the agreement above is genuine reproducibility rather than a degenerate
  search that ignores its random stream.

Wall-clock time (`Time_s`) is excluded from the digest: it varies between
executions by nature and is not a reproducible quantity.

## Budget adherence

All 200 runs of campaign A consumed exactly **10,000 evaluations** — minimum
10,000, maximum 10,000 — against a budget of 10,000. Every algorithm received
identical search effort despite differing per-iteration costs.

## Per-run provenance

`raw_runs.csv` records the seed and the evaluations consumed for each run, so
any single run can be re-executed in isolation:

| Benchmark | Algorithm | Run | Seed | FEs | BestFitness |
|---|---|---|---|---|---|
| F1 | PSO | 1 | 2675464099 | 10000 | 2.012990e+09 |
| F1 | PSO | 2 | 382534785 | 10000 | 1.736664e+09 |
| F1 | PSO | 3 | 2374083657 | 10000 | 2.012990e+09 |
| F1 | PSO | 4 | 1818095901 | 10000 | 4.826026e+03 |
| F1 | PSO | 5 | 2616313381 | 10000 | 1.205119e+09 |
| F1 | PSO | 6 | 2915525317 | 10000 | 5.631975e+03 |

All 200 seeds within a campaign are distinct, so the runs are independent
rather than repetitions of one trajectory.

## Reproducing this table

```python
from heurilab import run_experiment, get_opfunu_suite
from heurilab.algorithms import PSO, GWO, DE, WOA

run_experiment(
    algorithms=[("PSO", PSO), ("GWO", GWO), ("DE", DE), ("WOA", WOA)],
    benchmark_suites=[get_opfunu_suite("2017", ndim=10, functions=[1, 3, 4, 5, 6])],
    output_dir="repro", pop_size=20, dim=10, n_runs=10,
    max_fes=10_000, seed=42, run_engineering=False,
)
```
