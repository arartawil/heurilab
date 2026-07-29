# HeuriLab v2.2.0 — What's in the Package

Generated from the source tree on 28 July 2026. Every number below was read out
of the code, not the README.

---

## 1. At a glance

| | Count |
|---|---|
| Optimization algorithms | **102** |
| Benchmark functions (built in, no extra install) | **62** |
| Benchmark functions (with `opfunu`) | **196** additional, across 11 CEC editions |
| Engineering design problems | **12** |
| Statistical tests | **3** (Wilcoxon, Friedman, Nemenyi) |
| Output formats | CSV, Excel, PNG |
| Tests | **410** passing |
| Lines of Python | ~11,500 (+624 test) |

---

## 2. Algorithms — 102 in 6 categories

| Category | n | Algorithms |
|---|---|---|
| **Swarm intelligence** | 20 | PSO, GWO, WOA, MFO, SSA, HHO, MPA, BA, CS, FPA, DA, GOA, ALO, SHO, DO⚠, EHO, AO, HGS, GTO, RUN |
| **Evolutionary** | 15 | GA, DE, ES, EP, CMA, BBO, SHADE, TLGO⚠, CoDE, SaDE, OXDE, AGDE, LSHADE, EBOwithCMAR, IMODE |
| **Physics-based** | 16 | GSA, MVO, SCA, AOA, SA, EO, WDO, HGSO, CSS, CFO, TWO, ASO, RIME, AEO, GBO, TSO |
| **Human / social** | 16 | TLBO, JA, HS, ICA, CA, BSO, SOS_H⚠, QLA⚠, INFO, HBO, AOArch, CHIO, SSOA, POA, ED, AMO⚠ |
| **Bio-inspired** | 15 | ABC, FA, SOS, BFO, CSA, BOA, TSA, WHO, SBO, MBO, EPO, SMA, HBA, RSA, GJO |
| **Modern 2022–2025** | 20 | AVOA, DMO, MGO, DBO, COA, OOA, NOA, SAO, FLA, EVO, EDO, MOA, CPO, PO, FO, HO, KOA, SBOA, GMO, FFO |

⚠ = provenance issue, see §7. Full citations with DOIs are in `CITATIONS.md`.

Every algorithm subclasses `_Base` and implements
`optimize() -> (best_solution, best_fitness, convergence)`. Minimization.
Accepts `seed=` for reproducible runs.

---

## 3. Benchmark functions

### Built in (62, no extra install)

| Suite | Functions | Accessor |
|---|---|---|
| Classical F1–F23 (Yao et al.) | 23 | `get_classical_suite()` |
| — unimodal (F1–F7) | 7 | `get_unimodal_suite()` |
| — multimodal (F8–F13) | 6 | `get_multimodal_suite()` |
| — fixed-dimension (F14–F23) | 10 | `get_fixeddim_suite()` |
| CEC 2017 (F1, F3–F30) | 29 | `get_cec2017_suite()` |
| — unimodal / multimodal / hybrid / composition | 2 / 7 / 10 / 10 | `get_cec2017_*_suite()` |
| CEC 2020 | 10 | `get_cec2020_suite()` |
| — unimodal / multimodal / hybrid / composition | 1 / 3 / 3 / 3 | `get_cec2020_*_suite()` |

### Via `opfunu` (196 more, `pip install heurilab[cec]`)

| CEC edition | Functions | Supported dimensions |
|---|---|---|
| CEC 2005 | 25 | 10, 30, 50 |
| CEC 2008 | 7 | 100, 500, 1000 |
| CEC 2010 | 20 | 1000 |
| CEC 2013 | 28 | 2, 5, 10, 20, 30, 40, 50 |
| CEC 2014 | 30 | 10, 20, 30, 50, 100 |
| CEC 2015 | 15 | 10, 30 |
| CEC 2017 | 29 | 2, 10, 20, 30, 50, 100 |
| CEC 2019 | 10 | 9, 10, 16, 18 |
| CEC 2020 | 10 | 2, 5, 10, 15, 20, 30, 50, 100 |
| CEC 2021 | 10 | 10, 20 |
| CEC 2022 | 12 | 2, 10, 20 |
| **Total** | **196** | |

`get_opfunu_suite(year, ndim)`. Function numbers follow the **official CEC
numbering**, not opfunu's internal numbering — this matters for CEC 2017, where
opfunu packs F1/F3–F30 into F1..F29.

---

## 4. Engineering design problems — 12

All verified: an optimiser reaches the published optimum feasibly (tested).

| # | Problem | Vars | Constraints | Best known | Reference |
|---|---|---|---|---|---|
| 1 | Pressure Vessel Design | 4 | 4 | 5885.3328 | Kannan & Kramer (1994) |
| 2 | Welded Beam Design | 4 | 7 | 1.724852 | Rao (1996); Coello (2000) |
| 3 | Tension/Compression Spring | 3 | 4 | 0.012665 | Belegundu (1982); Arora (1989) |
| 4 | Speed Reducer Design | 7 | 11 | 2996.3482 | Golinski (1970) |
| 5 | Three-Bar Truss Design | 2 | 3 | 263.8958 | Ray & Saini (2001) |
| 6 | Cantilever Beam Design | 5 | 1 | 1.339956 | Chickermane & Gea (1996) |
| 7 | Gear Train Design | 4 | 0 | 2.7009e-12 | Sandgren (1990) |
| 8 | I-Beam Vertical Deflection | 4 | 1 | 0.0130741 | Gold & Krishnamurty (1997) |
| 9 | Tubular Column Design | 2 | 6 | 26.4995 | Rao (1996); Hsu & Liu (2007) |
| 10 | Multi-Disc Clutch Brake | 5 | 8 | 0.235242 | Osyczka (2002) |
| 11 | Corrugated Bulkhead Design | 4 | 6 | 6.842958 | Ravindran et al. (2006) |
| 12 | Himmelblau Nonlinear Design (G04) | 5 | 6 | −30665.539 | Himmelblau (1972) |

Objectives and constraints are declared separately, so constraint handling is
swappable and `problem.is_feasible(x)` works. Default handling is a static
penalty.

---

## 5. Analysis and output

| Component | What it provides |
|---|---|
| **Statistical tests** (`heurilab.stats`) | Wilcoxon rank-sum (pairwise vs proposed), Friedman (mean ranks + χ²), Nemenyi post-hoc with critical difference |
| **CSV export** | `raw_runs.csv` (per-run fitness, seed, wall-clock, full convergence trace), `results.csv` (mean/std/best/worst/median), `convergence.csv` |
| **Excel export** | `Results.xlsx` (+ ranking sheet), `Wilcoxon.xlsx`, `Friedman.xlsx` |
| **Plots** | Convergence curve + box plot per benchmark function, auto-saved PNG |
| **Enhancement Advisor** (`enhance()`) | 6 diagnostic problems, exploration/exploitation/stability/convergence-speed scores 0–100, stagnation detection, weakness detection mapped to 10 improvement techniques with code snippets, radar + diversity + score plots |

The Enhancement Advisor is the feature no competing library has.

---

## 6. Runner

`run_experiment()` — one call runs the whole algorithm × function × run matrix
and emits every artifact above.

| Parameter | Default | Notes |
|---|---|---|
| `algorithms` | — | `[(name, Class), ...]`; first is treated as the proposed method |
| `benchmark_suites` | — | any mix of built-in and opfunu suites |
| `pop_size`, `max_iter`, `dim`, `n_runs` | 50, 300, 30, 30 | |
| `seed` | `None` | per-run sub-seeds derived from (seed, function, algorithm, run) and recorded in `raw_runs.csv` |
| `n_jobs` | `1` | `-1` for all cores; results are identical to serial when seeded |
| `run_engineering` | `True` | plus `engineering_pop_size` / `_max_iter` / `_n_runs` |

---

## 7. Known issues to resolve before publication

| Item | Problem |
|---|---|
| `DO` | Labelled "Dolphin Optimizer" but implements WOA's spiral + encircling equations. No metaheuristic named "Dolphin Optimizer" exists in the literature. Counting it separately overstates the registry. |
| `AMO`, `QLA`, `SOS_H` | No traceable original publication. AMO appears to be your own design — claim it explicitly. |
| `TLGO` vs `TLBO` | Both resolve to Rao et al. (2011). Either distinguish or deduplicate. |
| `LSHADE`, `FO` | Wrong years in docstrings (2020→2014, 2024→2023). |
| `SSOA`, `AOArch` | Miscategorised: Sparrow Search is swarm, Archimedes is physics. |
| Constraint handling | Static penalty only; adaptive / ε-constrained / feasibility rules are the modern standard. |
| Engineering comparison | Runs the proposed algorithm only, so no cross-algorithm engineering table can be produced. |
| Docs | README not yet updated for opfunu suites, the 12 engineering problems, or `n_jobs`. |

---

## 8. Dependencies

| | Packages |
|---|---|
| Required | numpy, scipy, matplotlib, openpyxl, tqdm |
| `heurilab[cec]` | opfunu — the 196 extra CEC functions |
| `heurilab[parallel]` | joblib — `n_jobs` |
| `heurilab[all]` | both |
| `heurilab[dev]` | pytest, build, twine |
