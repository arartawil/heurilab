# HeuriLab

<p align="center">
  <img src="logo.png" alt="HeuriLab Logo" width="400">
</p>

A Python package with **100 built-in metaheuristic optimization algorithms** and automated experiment infrastructure: runner, CSV outputs, convergence/box plots, and statistical Excel analysis.

## Installation

```bash
pip install heurilab
```

Or install from source:

```bash
pip install -e .
```

## Quick Start

```python
from heurilab import run_experiment, get_classical_suite, get_cec2017_suite
from heurilab.algorithms import PSO, GWO, WOA, DE, CPO, DBO, HO

algorithms = [
    ("PSO", PSO),       # First = proposed (highlighted in plots & stats)
    ("GWO", GWO),
    ("WOA", WOA),
    ("DE", DE),
    ("CPO", CPO),       # Modern (2024)
    ("DBO", DBO),       # Modern (2023)
    ("HO", HO),         # Modern (2024)
]

run_experiment(
    algorithms=algorithms,
    benchmark_suites=[get_classical_suite(), get_cec2017_suite()],
    output_dir="output",
    pop_size=50, max_iter=300, dim=30, n_runs=30,
)
```

## 100 Built-in Algorithms

### Swarm Intelligence (20)
PSO, GWO, WOA, MFO, SSA, HHO, MPA, BA, CS, FPA, DA, GOA, ALO, SHO, DO, EHO, AO, HGS, GTO, RUN

### Evolutionary (15)
GA, DE, ES, EP, CMA, BBO, SHADE, TLGO, CoDE, SaDE, OXDE, AGDE, LSHADE, EBOwithCMAR, IMODE

### Physics-based (16)
GSA, MVO, SCA, AOA, SA, EO, WDO, HGSO, CSS, CFO, TWO, ASO, RIME, AEO, GBO, TSO

### Human/Social (14)
TLBO, JA, HS, ICA, CA, BSO, SOS_H, QLA, INFO, HBO, AOArch, CHIO, SSOA, POA

### Bio-inspired (15)
ABC, FA, SOS, BFO, CSA, BOA, TSA, WHO, SBO, MBO, EPO, SMA, HBA, RSA, GJO

### Modern 2022–2025 (20)
| Abbr | Algorithm | Year |
|------|-----------|------|
| AVOA | African Vultures Optimization | 2022 |
| DMO | Dwarf Mongoose Optimization | 2022 |
| MGO | Mountain Gazelle Optimizer | 2022 |
| DBO | Dung Beetle Optimizer | 2023 |
| COA | Coati Optimization Algorithm | 2023 |
| OOA | Osprey Optimization Algorithm | 2023 |
| NOA | Nutcracker Optimization Algorithm | 2023 |
| SAO | Snow Ablation Optimizer | 2023 |
| FLA | Fick's Law Algorithm | 2023 |
| EVO | Energy Valley Optimizer | 2023 |
| EDO | Exponential Distribution Optimizer | 2023 |
| MOA | Mother Optimization Algorithm | 2023 |
| CPO | Crested Porcupine Optimizer | 2024 |
| PO | Parrot Optimizer | 2024 |
| FO | Fox Optimizer | 2024 |
| HO | Hippopotamus Optimization Algorithm | 2024 |
| KOA | Kepler Optimization Algorithm | 2024 |
| SBOA | Secretary Bird Optimization Algorithm | 2024 |
| GMO | Geometric Mean Optimizer | 2024 |
| FFO | Fennec Fox Optimization | 2024 |

## Algorithm Interface

All algorithms must implement:

```python
class Algorithm:
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        ...

    def optimize(self):
        # Returns: (best_solution, best_fitness, convergence_list)
        # convergence_list has max_iter+1 entries (index 0 = initial)
        return best_solution, best_fitness, convergence_list
```

## Benchmark Suites

- **Classical (F1–F23):** 7 unimodal + 6 multimodal + 10 fixed-dimension
- **CEC 2017 (29 functions):** F1, F3–F30 with shifted optima

```python
from heurilab import (
    get_classical_suite,    # All 23
    get_unimodal_suite,     # F1–F7
    get_multimodal_suite,   # F8–F13
    get_fixeddim_suite,     # F14–F23
    get_cec2017_suite,      # All 29
)
```

## Outputs

```
output/
├── CSV Data/
│   ├── results.csv          # Summary stats per algo × function
│   ├── raw_runs.csv         # Per-run data with convergence
│   └── convergence.csv      # Mean convergence per algo × function
├── Convergence Curves/      # One PNG per benchmark function
├── Box Plots/               # One PNG per benchmark function
└── Excel Files/
    ├── Results.xlsx          # Styled summary with rankings
    ├── Wilcoxon.xlsx         # Wilcoxon rank-sum tests
    ├── Friedman.xlsx         # Friedman + Nemenyi post-hoc
    └── Engineering.xlsx      # Engineering design problems
```

## Dependencies

- numpy
- scipy
- matplotlib
- openpyxl
