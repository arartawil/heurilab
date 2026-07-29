# HeuriLab v2.2.0 vs the Python Optimization Ecosystem

Updated 28 July 2026, after adding opfunu-backed CEC suites, 12 engineering
design problems, per-run seeding and parallelism. Supersedes the §2 table of the
earlier competitive analysis.

Competitor facts were verified by installing each package and introspecting its
source (algorithm counts, `grep` for statistical tests / box plots / seeding),
plus the PyPI JSON API for versions. HeuriLab's own numbers were read from its
source tree.

Legend: ✅ built in and automatic · ◐ present but manual or partial · ❌ absent

---

## 1. Master comparison

| | **HeuriLab** | MEALPY | pymoo | NiaPy | jMetalPy | pygmo | Nevergrad | Opytimizer | DEAP | PySwarms | scikit-opt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Algorithms** | 102 | **233** | 33 | 39–77 | 23 | 25 | 542¹ | 114 | 4 drivers² | 4 | 14 |
| Single-objective focus | ✅ | ✅ | ◐ | ✅ | ❌ MO | ◐ | ✅ | ✅ | ◐ | ✅ | ✅ |
| Multi-objective | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ◐ | ◐ | ✅ | ❌ | ❌ |
| **Classical F1–F23 as a named suite** | **✅** | ❌³ | ❌ | ❌³ | ❌ | ❌ | ❌³ | ❌³ | ❌ | ❌ | ❌ |
| **CEC coverage** | **✅ 258 fns, 11 editions** | ◐ 197 via opfunu, unwrapped | ❌ | ❌ | ◐ CEC2009 | ✅ 4 editions | ◐ LSGO | ✅ 67 fns | ❌ | ❌ | ❌ |
| **Engineering design problems** | **✅ 12, optima verified** | ❌ | ✅ G1–G24 + 5 | ❌ | ✅ RE 16 + RWA 10 (MO) | ❌ | ◐ applied | ❌ | ❌ | ❌ | ❌ |
| **Wilcoxon / Friedman / Nemenyi** | **✅** | ❌ | ❌ | ❌ | ✅⁴ | ❌ | ❌⁵ | ❌ | ❌ | ❌ | ❌ |
| **Box plots auto-generated** | **✅** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Convergence curves auto-saved | ✅ | ◐ per run | ◐ manual | ◐ display only | ❌ | ❌ | ✅ | ◐ broken⁶ | ❌ | ◐ | ❌ |
| CSV export | ✅ real-time | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Excel export, styled** | **✅** | ❌ | ❌ | ◐ raw | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **One-call full experiment** | **✅** | ◐ raw numbers | ❌ | ◐ raw numbers | ◐ 4–5 calls | ◐ | ◐ preset plans | ❌ | ❌ | ❌ | ❌ |
| **Reproducible *campaign*** | **✅ per-run seeds** | ❌⁷ | n/a⁸ | ❌⁷ | ❌ | n/a⁸ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Parallelism** | **✅ `n_jobs`** | ✅ | ◐ | ◐ | ✅ | ✅ | ✅ | ❌ | ◐ | ❌ | ❌ |
| **Algorithm diagnostics + improvement advice** | **✅ unique** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Latest release | 2.2.0 | 3.0.3 (2025-08) | 0.6.2 (2026-06) | 2.7.0 (2026-02) | 1.9.0 (2025-10) | 2.19.8 (2026-04) | 1.0.12 (2025-04) | 4.1.0 (2026-04) | 1.4.4 (2026-04) | 1.3.0 (2021-01) | 0.6.6 (2022-01) |
| Maintained | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ | ✅ | ✅ | ❌ dormant | ❌ dormant |
| Software paper | *(pending)* | JSA 2023 | IEEE Access 2020 | JOSS 2018 | SwEvo 2019 | JOSS 2020 | SIGEVOlution 2021 | arXiv only | JMLR 2012 | JOSS 2018 | none |

**Notes**
1. Nevergrad's 542 are largely parameter variants of a much smaller set of families; its own docs decline to give a count.
2. DEAP is a toolkit: 4 evolutionary drivers plus 15 selection / 14 crossover / 8 mutation operators to compose.
3. The individual functions exist (opfunu 125 named, NiaPy 47, nevergrad 38, opytimark 162) but none packages them as the named Yao et al. F1–F23 set.
4. jMetalPy has *more* tests than HeuriLab (Quade, Holm, Hochberg, Bayesian, …). Do not claim to be the only library with statistical testing.
5. Nevergrad compares via win-rate "fight" matrices, not significance tests.
6. `opytimizer==4.1.0`'s `visualization` subpackage is unimportable as published — `_core/` is missing from wheel and sdist.
7. MEALPY and NiaPy accept a seed on a *single* run, but neither passes one through their batch runner (`Multitask`, `Runner`), so a 30-run campaign is not reproducible.
8. pymoo and pygmo seed cleanly per run but have no batch runner, so "campaign reproducibility" does not apply.

---

## 2. What changed since v2.1

Three cells flipped, and they were the three that a reviewer would have attacked.

| Capability | Before | Now | Effect on the comparison |
|---|---|---|---|
| CEC coverage | 62 functions (2017 + 2020) | 258 reachable, 11 editions | Was behind pygmo and Opytimizer; now ahead of everyone |
| Engineering problems | 3 | 12, each verified to reach its published optimum | Was behind pymoo and jMetalPy; now the largest *single-objective* set, and the only verified one |
| Parallelism | none | `n_jobs`, results identical to serial | Removes the "does this scale?" objection |
| Campaign reproducibility | none | per-run seeds recorded in `raw_runs.csv` | Moves from worst-in-class to joint-best |

---

## 3. Claims you can defend

Ranked by how hard they are to attack.

**1. The only library with automated algorithm diagnosis and improvement advice.**
`enhance()` instruments the search trajectory, scores exploration / exploitation /
stability / convergence speed, detects stagnation, and maps weaknesses to ten
improvement techniques with runnable code. I searched all ten competitors; there
is no equivalent. Unattackable.

**2. The only library where a full comparison campaign is reproducible *and*
statistically tested *and* plotted.** Nevergrad also has reproducible campaigns
but no significance tests and no box plots. jMetalPy has richer statistics but no
seeding at all and is multi-objective. Nobody else has both halves.

**3. The only library shipping the classical F1–F23 suite as a named entity.**
Trivial to implement, universally required in single-objective metaheuristic
papers, and nobody has done it.

**4. The largest verified set of single-objective engineering design problems.**
pymoo ships more constrained problems (G1–G24) and jMetalPy ships the 16-problem
RE suite, but jMetalPy's are multi-objective and neither project verifies its
formulations against published optima. HeuriLab's twelve are each checked in CI:
an optimiser must reach the literature value feasibly. Word this as
"single-objective" and "verified" — drop either qualifier and it becomes false.

**5. Broadest CEC coverage.** 258 functions across 11 editions, natively wrapped.
MEALPY depends on opfunu but never wraps it, so its users assemble suites by
hand.

---

## 4. Claims that will get you caught

| Tempting claim | Why it fails |
|---|---|
| "Most algorithms" | MEALPY 233, Opytimizer 114, Nevergrad 542 names vs your 102. Never lead with the count. |
| "Only library with statistical tests" | jMetalPy has more of them. Say "the only *single-objective*…". |
| "Only library with engineering problems" | pymoo and jMetalPy both ship them. Your differentiators are *single-objective* and *verified*. |
| "102 distinct algorithms" | `DO` is WOA under another name. Fix or relabel before this number appears in print. |
| "Complete metaheuristic library" | No multi-objective support at all. pymoo and jMetalPy own that space. |

---

## 5. Honest remaining weaknesses

| | Detail |
|---|---|
| Algorithm count | 102 vs MEALPY's 233. Frame around workflow, not inventory. |
| No multi-objective | Structural. Say so plainly and cite pymoo/jMetalPy as complementary. |
| Constraint handling | Static penalty only; adaptive, ε-constrained and feasibility rules are the modern standard. |
| Engineering comparison | The engineering runner still evaluates only the proposed algorithm, so no cross-algorithm engineering table can be produced. |
| Provenance | `DO` = WOA; `AMO`, `QLA`, `SOS_H` uncited; `TLGO` duplicates `TLBO`; two wrong docstring years. See `CITATIONS.md`. |
| Documentation | README still describes the old feature set; no hosted API docs. |

---

## 6. Revised statement of need

> Proposing a new single-objective metaheuristic carries a fixed empirical
> burden: evaluation on the classical F1–F23 suite and a CEC benchmark set, 30
> independent runs per pairing, mean/std/best/worst tables, convergence curves,
> box plots, Wilcoxon rank-sum tests against each competitor, a Friedman test
> with Nemenyi post-hoc analysis, and validation on constrained engineering
> design problems. Existing Python libraries supply the algorithms but not this
> pipeline. MEALPY (233 algorithms) and NiaPy provide batch runners that emit
> raw fitness values and stop there, and neither propagates a seed through the
> batch, so a completed campaign cannot be reproduced. pymoo, the richest problem
> library, has no batch runner. jMetalPy implements the statistical machinery but
> targets multi-objective optimisation, requires the experiment matrix to be
> assembled by hand, and has no seeding at all. Every research group therefore
> reimplements the same several hundred lines of orchestration, aggregation,
> plotting and statistics — code that is rarely published and therefore rarely
> audited.
>
> HeuriLab closes this gap. A single `run_experiment()` call executes the full
> algorithm × function × run matrix, in parallel when asked, and emits the
> complete artifact set above, writing incrementally so long campaigns survive
> interruption. Each run receives a deterministic sub-seed derived from the
> campaign seed and recorded alongside its result, so a campaign reproduces
> exactly and any individual run can be re-executed in isolation. The library
> ships 102 algorithms, 62 built-in benchmark functions with a further 196 CEC
> functions available through opfunu, and 12 constrained engineering design
> problems whose formulations are continuously verified against their published
> optima. Beyond benchmarking, HeuriLab introduces the Enhancement Advisor: an
> automated diagnostic that quantifies an algorithm's exploration, exploitation,
> stability, convergence speed and stagnation, then maps detected weaknesses onto
> a curated knowledge base of improvement techniques, each returned with
> applicable code. To our knowledge no existing optimisation library offers
> automated algorithmic diagnosis and improvement guidance.
