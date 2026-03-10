<div align="center">

# HeuriLab

**The Complete Metaheuristic Optimization Laboratory**

*100 ready-to-use algorithms, 52 benchmark functions, automated experiments, statistical analysis & publication-ready outputs — all in one package*

[![PyPI version](https://badge.fury.io/py/heurilab.svg)](https://badge.fury.io/py/heurilab)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/heurilab)](https://pepy.tech/project/heurilab)
[![GitHub stars](https://img.shields.io/github/stars/arartawil/heurilab.svg?style=social&label=Star)](https://github.com/arartawil/heurilab)

[📚 Algorithms](#-100-built-in-algorithms) • [🚀 Quick Start](#-quick-start) • [📊 Benchmarks](#-benchmark-suites) • [💡 Examples](#-complete-examples) • [🤝 Contributing](#-contributing)

</div>

<p align="center">
  <img src="logo.png" alt="HeuriLab Logo" width="400">
</p>

---

## ✨ Highlights

🧠 **100 algorithms** across 6 categories — from classics (PSO, GA, DE) to cutting-edge 2024 optimizers  
📐 **52 benchmarks** — 23 classical (F1–F23) + 29 CEC 2017 (F1, F3–F30)  
⚡ **One-command experiments** — run, export CSVs, plot, and generate Excel stats automatically  
📊 **Publication-ready outputs** — convergence curves, box plots, Wilcoxon & Friedman tests  
🔧 **Real-time CSV saving** — data is written after every single run, never lost  
📈 **Live progress bar** — tqdm-powered experiment tracking  
🔬 **Enhancement Advisor** — automated diagnostics, scoring (0–100), weakness detection & enhancement suggestions  

**[Get Started in 30 seconds →](#-installation)**

---

## 🎯 HeuriLab vs Others

| Feature | HeuriLab | MEALPY | PySwarms | SciPy |
|---------|----------|--------|----------|-------|
| **Algorithms** | ✅ 100 | ~200 | PSO only | Few |
| **Benchmarks** | ✅ 52 (Classical + CEC 2017) | Limited | None | None |
| **Automated Runner** | ✅ One function call | ❌ Manual | ❌ Manual | ❌ Manual |
| **CSV + Excel Export** | ✅ Real-time | ❌ | ❌ | ❌ |
| **Statistical Tests** | ✅ Wilcoxon + Friedman + Nemenyi | ❌ | ❌ | ❌ |
| **Convergence Plots** | ✅ Auto-generated | Manual | Manual | ❌ |
| **Box Plots** | ✅ Auto-generated | ❌ | ❌ | ❌ |
| **Enhancement Advisor** | ✅ Built-in | ❌ | ❌ | ❌ |
| **Engineering Problems** | ✅ Built-in | ❌ | ❌ | ❌ |
| **Progress Bar** | ✅ tqdm | ❌ | ❌ | ❌ |
| **Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 📋 Table of Contents

- [Highlights](#-highlights)
- [HeuriLab vs Others](#-heurilab-vs-others)
- [Installation](#-installation)
- [30-Second Quickstart](#-30-second-quickstart)
- [Quick Start](#-quick-start)
- [Architecture Overview](#-architecture-overview)
- [Core Module](#-core-module)
  - [BenchmarkConfig & BenchmarkSuite](#benchmarkconfig--benchmarksuite)
  - [run_experiment() — Full API Reference](#run_experiment--full-api-reference)
- [100 Built-in Algorithms](#-100-built-in-algorithms)
  - [Base Class (_Base)](#base-class-_base)
  - [Swarm Intelligence (20)](#swarm-intelligence-20)
  - [Evolutionary (15)](#evolutionary-15)
  - [Physics-based (16)](#physics-based-16)
  - [Human/Social (14)](#humansocial-14)
  - [Bio-inspired (15)](#bio-inspired-15)
  - [Modern 2022–2025 (20)](#modern-20222025-20)
  - [Convenience Algorithm Lists](#convenience-algorithm-lists)
- [Benchmark Suites](#-benchmark-suites)
  - [Classical Functions (F1–F23)](#classical-benchmark-functions-f1f23)
  - [CEC 2017 Functions (29)](#cec-2017-benchmark-functions-29-functions)
- [Enhancement Advisor (enhance)](#-enhancement-advisor)
  - [enhance()](#enhance-1)
  - [cec2017_benchmarks()](#cec2017_benchmarks)
  - [Diagnostic Metrics (0–100)](#diagnostic-metrics-scored-0100)
  - [Enhancement Knowledge Base](#enhancement-knowledge-base-10-techniques)
- [Exporters Module](#-exporters-module)
  - [CSV Export](#csv-export)
  - [Plots](#plots)
  - [Excel Export](#excel-export)
- [Stats Module](#-stats-module)
  - [Wilcoxon Rank-Sum Test](#wilcoxon-rank-sum-test)
  - [Friedman Test](#friedman-test)
  - [Nemenyi Post-Hoc Test](#nemenyi-post-hoc-test)
- [Engineering Module](#-engineering-module)
- [Output Structure](#-output-structure)
- [Complete Examples](#-complete-examples)
- [Creating Custom Algorithms](#-creating-custom-algorithms)
- [Dependencies](#-dependencies)
- [Contributing](#-contributing)
- [Community](#-community)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Star History](#-star-history)

---

## 📦 Installation

**Quick install:**
```bash
pip install heurilab
```

**From source (latest features):**
```bash
git clone https://github.com/arartawil/heurilab.git
cd heurilab
pip install -e .
```

**Requirements:** Python ≥3.8, NumPy, SciPy, Matplotlib, openpyxl, tqdm

**[View on PyPI](https://pypi.org/project/heurilab/) • [GitHub](https://github.com/arartawil/heurilab)**

---

## ⚡ 30-Second Quickstart

```python
from heurilab import run_experiment, get_classical_suite
from heurilab.algorithms import PSO, GWO, WOA

run_experiment(
    algorithms=[("PSO", PSO), ("GWO", GWO), ("WOA", WOA)],
    benchmark_suites=[get_classical_suite()],
    output_dir="output",
)
```

**That's it!** 🎉 CSVs, plots, and Excel files are generated automatically.

---

## 🎯 Quick Start

### Full Experiment with Classical + CEC 2017

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
    pop_size=50,
    max_iter=300,
    dim=30,
    n_runs=30,
    run_engineering=True,
)
```

**Output:**
```
============================================================
  Metaheuristic Experiment
  Algorithms: PSO, GWO, WOA, DE, CPO, DBO, HO
  Proposed: PSO
  Pop=50, MaxIter=300, Dim=30, Runs=30
============================================================

Experiment: 100%|██████████████████████████████| 364/364 [12:34<00:00]

CSV files written to: output\CSV Data
Plots saved to: output
Excel files saved to: output\Excel Files

============================================================
  Experiment complete!
  All outputs in: C:\...\output
============================================================
```

### Using Specific Benchmark Subsets

```python
from heurilab import (
    get_classical_suite,    # All 23 classical (F1–F23)
    get_unimodal_suite,     # F1–F7  (unimodal)
    get_multimodal_suite,   # F8–F13 (multimodal)
    get_fixeddim_suite,     # F14–F23 (fixed-dimension)
    get_cec2017_suite,      # All 29 CEC 2017
)

# Quick test with unimodal only
run_experiment(
    algorithms=[("PSO", PSO), ("GWO", GWO)],
    benchmark_suites=[get_unimodal_suite()],
    output_dir="output",
    pop_size=30, max_iter=100, dim=10, n_runs=5,
)
```

---

## 🏛️ Architecture Overview

```
heurilab/
├── __init__.py              # Public API re-exports
├── algorithms/              # 100 metaheuristic algorithms (6 categories)
│   ├── base.py              # _Base class — shared interface
│   ├── swarm/               # 20 swarm intelligence algorithms
│   ├── evolutionary/        # 15 evolutionary algorithms
│   ├── physics/             # 16 physics-based algorithms
│   ├── human/               # 14 human/social algorithms
│   ├── bio/                 # 15 bio-inspired algorithms
│   └── modern/              # 20 modern (2022–2025) algorithms
├── core/
│   ├── benchmarks.py        # BenchmarkConfig, BenchmarkSuite
│   ├── functions.py         # 23 classical benchmark functions
│   ├── cec2017.py           # 29 CEC 2017 functions
│   └── runner.py            # run_experiment() orchestrator
├── analyzer/
│   ├── enhance.py           # Enhancement Advisor
│   └── _cec2017_tests.py    # CEC 2017 benchmark dict for enhance()
├── exporters/
│   ├── csv_export.py        # Real-time CSV output
│   ├── plots.py             # Convergence curves & box plots
│   └── excel_export.py      # Results.xlsx, Wilcoxon.xlsx, Friedman.xlsx
├── stats/
│   └── tests.py             # Wilcoxon, Friedman, Nemenyi statistical tests
└── engineering/
    ├── problems.py          # 3 constrained engineering design problems
    └── runner.py            # Engineering problem runner
```

**Pipeline Flow:**

```
run_experiment()
  ├── Phase 1: Run all (algorithm × benchmark × n_runs) → CSV files (real-time)
  ├── Phase 2: Generate convergence curve + box plot PNGs
  ├── Phase 3: Generate Excel files (Results, Wilcoxon, Friedman)
  └── Phase 4: (optional) Run engineering design problems
```

---

## 🧩 Core Module

### BenchmarkConfig & BenchmarkSuite

```python
from heurilab import BenchmarkConfig, BenchmarkSuite
```

**`BenchmarkConfig`** — A single benchmark function:

| Field      | Type       | Default | Description                   |
|------------|------------|---------|-------------------------------|
| `name`     | `str`      | —       | Display name                  |
| `obj_func` | `Callable` | —       | Objective function `f(x) → float` |
| `lb`       | `float`    | —       | Lower bound                   |
| `ub`       | `float`    | —       | Upper bound                   |
| `dim`      | `int`      | `30`    | Dimensionality                |

**`BenchmarkSuite`** — A named collection of benchmarks:

| Field        | Type                     | Description              |
|-------------|--------------------------|--------------------------|
| `category`   | `str`                    | Category name (e.g., "Classical") |
| `benchmarks` | `List[BenchmarkConfig]`  | List of benchmark configs |

**Methods:**

| Method                               | Description                        |
|--------------------------------------|------------------------------------|
| `.add(name, obj_func, lb, ub, dim)`  | Add a benchmark (chainable)        |
| `len(suite)`                         | Number of benchmarks               |
| `for bench in suite:`                | Iterate over benchmarks            |

**Example — Custom Suite:**

```python
import numpy as np
from heurilab import BenchmarkSuite

suite = BenchmarkSuite(category="My Functions")
suite.add("Sphere", lambda x: sum(x**2), -100, 100, 30)
suite.add("Rastrigin", lambda x: 10*len(x) + sum(x**2 - 10*np.cos(2*np.pi*x)), -5.12, 5.12, 30)
```

---

### run_experiment() — Full API Reference

```python
from heurilab import run_experiment
```

**Signature:**

```python
run_experiment(
    algorithms: List[Tuple[str, Type]],
    benchmark_suites: List[BenchmarkSuite],
    output_dir: str = "output",
    pop_size: int = 50,
    max_iter: int = 300,
    dim: int = 30,
    n_runs: int = 30,
    run_engineering: bool = True,
    engineering_pop_size: int = 50,
    engineering_max_iter: int = 500,
    engineering_n_runs: int = 30,
)
```

**Parameters:**

| Parameter               | Type                         | Default    | Description                                           |
|------------------------|------------------------------|------------|-------------------------------------------------------|
| `algorithms`            | `List[Tuple[str, Type]]`     | —          | List of `(name, AlgorithmClass)` tuples. **First = proposed** |
| `benchmark_suites`      | `List[BenchmarkSuite]`       | —          | Benchmark suites to run                              |
| `output_dir`            | `str`                        | `"output"` | Root output directory                                |
| `pop_size`              | `int`                        | `50`       | Population size                                       |
| `max_iter`              | `int`                        | `300`      | Maximum iterations                                    |
| `dim`                   | `int`                        | `30`       | Default dimensionality (overridden by benchmark config) |
| `n_runs`                | `int`                        | `30`       | Number of independent runs                            |
| `run_engineering`       | `bool`                       | `True`     | Whether to run engineering design problems             |
| `engineering_pop_size`   | `int`                       | `50`       | Pop size for engineering problems                     |
| `engineering_max_iter`   | `int`                       | `500`      | Max iterations for engineering problems               |
| `engineering_n_runs`     | `int`                       | `30`       | Number of runs for engineering problems               |

**Pipeline Phases:**

1. **Phase 1** — Run all algorithm × benchmark × n_runs combinations. CSV files are written in real-time after every single run (`raw_runs.csv`) and after every algorithm-function combo (`results.csv`, `convergence.csv`).
2. **Phase 2** — Generate convergence curve and box plot PNG images for every benchmark function.
3. **Phase 3** — Generate styled Excel workbooks: `Results.xlsx`, `Wilcoxon.xlsx`, `Friedman.xlsx`.
4. **Phase 4** — (optional) Run 3 engineering design problems using only the proposed algorithm.

> ⚠️ **Important:** The **first algorithm** in the list is treated as the "proposed" algorithm and is:
> - Highlighted with thicker lines in convergence plots
> - Highlighted with red border in box plots
> - Used as the reference in Wilcoxon pairwise tests
> - Highlighted with a distinct fill color in Excel sheets
> - The only algorithm used for engineering problems

---

## 🧠 100 Built-in Algorithms

```python
from heurilab.algorithms import PSO, GWO, WOA, ...
from heurilab.algorithms import ALL_ALGORITHMS
```

### Base Class (_Base)

Every algorithm inherits from `_Base`:

```python
class _Base:
    def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
        self.pop_size = pop_size           # int
        self.dim = dim                     # int
        self.lb = np.array(lb)             # np.ndarray (broadcast to dim)
        self.ub = np.array(ub)             # np.ndarray (broadcast to dim)
        self.max_iter = max_iter           # int
        self.obj_func = obj_func           # Callable
        self._progress_callback = None

    def _init_pop(self):
        """Random uniform initialization in [lb, ub]."""
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

    def _clip(self, x):
        """Clip solution to [lb, ub]."""
        return np.clip(x, self.lb, self.ub)

    def _eval(self, x):
        """Evaluate objective function (with optional progress callback)."""
        result = self.obj_func(x)
        if self._progress_callback is not None:
            self._progress_callback(result)
        return result
```

**All algorithms must implement:**

```python
def optimize(self) -> Tuple[np.ndarray, float, list]:
    """
    Returns:
        best_solution: np.ndarray of shape (dim,)
        best_fitness: float
        convergence: list of best fitness per iteration (length = max_iter + 1)
    """
```

---

### Swarm Intelligence (20)

| Abbr | Algorithm                        | Author(s), Year              |
|------|----------------------------------|------------------------------|
| PSO  | Particle Swarm Optimization      | Kennedy & Eberhart, 1995     |
| GWO  | Grey Wolf Optimizer              | Mirjalili et al., 2014       |
| WOA  | Whale Optimization Algorithm     | Mirjalili & Lewis, 2016      |
| MFO  | Moth-Flame Optimization          | Mirjalili, 2015              |
| SSA  | Salp Swarm Algorithm             | Mirjalili et al., 2017       |
| HHO  | Harris Hawks Optimization        | Heidari et al., 2019         |
| MPA  | Marine Predators Algorithm       | Faramarzi et al., 2020       |
| BA   | Bat Algorithm                    | Yang, 2010                   |
| CS   | Cuckoo Search                    | Yang & Deb, 2009             |
| FPA  | Flower Pollination Algorithm     | Yang, 2012                   |
| DA   | Dragonfly Algorithm              | Mirjalili, 2016              |
| GOA  | Grasshopper Optimization Algorithm | Saremi et al., 2017        |
| ALO  | Ant Lion Optimizer               | Mirjalili, 2015              |
| SHO  | Spotted Hyena Optimizer          | Dhiman & Kumar, 2017         |
| DO   | Dolphin Optimizer                | Shaqfa & Beyer, 2023         |
| EHO  | Elephant Herding Optimization    | Wang et al., 2015            |
| AO   | Aquila Optimizer                 | Abualigah et al., 2021       |
| HGS  | Hunger Games Search              | Yang et al., 2021            |
| GTO  | Gorilla Troops Optimizer         | Abdollahzadeh et al., 2021   |
| RUN  | RUNge Kutta Optimizer            | Ahmadianfar et al., 2021     |

### Evolutionary (15)

| Abbr          | Algorithm                             | Author(s), Year              |
|---------------|---------------------------------------|------------------------------|
| GA            | Genetic Algorithm                     | Holland, 1975                |
| DE            | Differential Evolution                | Storn & Price, 1997          |
| ES            | Evolution Strategy                    | Rechenberg, 1973             |
| EP            | Evolutionary Programming             | Fogel, 1966                  |
| CMA           | CMA-ES                               | Hansen & Ostermeier, 2001    |
| BBO           | Biogeography-Based Optimization       | Simon, 2008                  |
| SHADE         | Success-History Adaptation DE         | Tanabe & Fukunaga, 2013      |
| TLGO          | Teaching-Learning-GO                  | Rao et al., 2011             |
| CoDE          | Composite DE                         | Wang et al., 2011            |
| SaDE          | Self-adaptive DE                      | Qin et al., 2009             |
| OXDE          | Orthogonal Crossover DE              | Wang & Cai, 2012             |
| AGDE          | Adaptive Guided DE                    | Mohamed et al., 2019         |
| LSHADE        | L-SHADE                              | Tanabe & Fukunaga, 2014      |
| EBOwithCMAR   | EBO with CMAR                         | Kumar et al., 2017           |
| IMODE         | Improved Multi-Op DE                  | Sallam et al., 2020          |

### Physics-based (16)

| Abbr | Algorithm                        | Author(s), Year              |
|------|----------------------------------|------------------------------|
| GSA  | Gravitational Search Algorithm   | Rashedi et al., 2009         |
| MVO  | Multi-Verse Optimizer            | Mirjalili et al., 2016       |
| SCA  | Sine Cosine Algorithm            | Mirjalili, 2016              |
| AOA  | Arithmetic Optimization Algorithm | Abualigah et al., 2021      |
| SA   | Simulated Annealing              | Kirkpatrick et al., 1983     |
| EO   | Equilibrium Optimizer            | Faramarzi et al., 2020       |
| WDO  | Wind Driven Optimization         | Bayraktar et al., 2010       |
| HGSO | Henry Gas Solubility Optimization | Hashim et al., 2019         |
| CSS  | Charged System Search            | Kaveh & Talatahari, 2010     |
| CFO  | Central Force Optimization       | Formato, 2007                |
| TWO  | Tug of War Optimization          | Kaveh & Zolghadr, 2016       |
| ASO  | Atom Search Optimization         | Zhao et al., 2019            |
| RIME | RIME Optimization                | Su et al., 2023              |
| AEO  | Artificial Ecosystem Optimizer   | Zhao et al., 2020            |
| GBO  | Gradient-Based Optimizer         | Ahmadianfar et al., 2020     |
| TSO  | Transient Search Optimization    | Qais et al., 2020            |

### Human/Social (14)

| Abbr   | Algorithm                         | Author(s), Year              |
|--------|-----------------------------------|------------------------------|
| TLBO   | Teaching-Learning-Based Optimization | Rao et al., 2011          |
| JA     | Jaya Algorithm                    | Rao, 2016                    |
| HS     | Harmony Search                    | Geem et al., 2001            |
| ICA    | Imperialist Competitive Algorithm | Atashpaz-Gargari & Lucas, 2007 |
| CA     | Cultural Algorithm                | Reynolds, 1994               |
| BSO    | Brain Storm Optimization          | Shi, 2011                    |
| SOS_H  | Symbiotic Organisms Search        | Cheng & Prayogo, 2014        |
| QLA    | Q-Learning Algorithm              | Watkins, 1989                |
| INFO   | Weighted Mean of Vectors (INFO)   | Ahmadianfar et al., 2022     |
| HBO    | Heap-Based Optimizer              | Askari et al., 2020          |
| AOArch | Archimedes Optimization           | Hashim et al., 2021          |
| CHIO   | Coronavirus Herd Immunity Optimizer | Al-Betar et al., 2021      |
| SSOA   | Social Spider Optimization        | Cuevas et al., 2013          |
| POA    | Pelican Optimization Algorithm    | Trojovský & Dehghani, 2022   |

### Bio-inspired (15)

| Abbr | Algorithm                        | Author(s), Year              |
|------|----------------------------------|------------------------------|
| ABC  | Artificial Bee Colony            | Karaboga, 2005               |
| FA   | Firefly Algorithm                | Yang, 2008                   |
| SOS  | Symbiotic Organisms Search       | Cheng & Prayogo, 2014        |
| BFO  | Bacterial Foraging Optimization  | Passino, 2002                |
| CSA  | Crow Search Algorithm            | Askarzadeh, 2016             |
| BOA  | Butterfly Optimization Algorithm | Arora & Singh, 2019          |
| TSA  | Tunicate Swarm Algorithm         | Kaur et al., 2020            |
| WHO  | Wild Horse Optimizer             | Naruei & Keynia, 2022        |
| SBO  | Satin Bowerbird Optimizer        | Moosavi & Bardsiri, 2017     |
| MBO  | Monarch Butterfly Optimization   | Wang et al., 2019            |
| EPO  | Emperor Penguin Optimizer        | Dhiman & Kumar, 2018         |
| SMA  | Slime Mould Algorithm            | Li et al., 2020              |
| HBA  | Honey Badger Algorithm           | Hashim et al., 2022          |
| RSA  | Reptile Search Algorithm         | Abualigah et al., 2022       |
| GJO  | Golden Jackal Optimization       | Chopra & Ansari, 2022        |

### Modern 2022–2025 (20)

| Abbr | Algorithm                        | Author(s), Year              |
|------|----------------------------------|------------------------------|
| AVOA | African Vultures Optimization    | Abdollahzadeh et al., 2021   |
| DMO  | Dwarf Mongoose Optimizer         | Agushaka et al., 2022        |
| MGO  | Mountain Gazelle Optimizer       | Abdollahzadeh et al., 2022   |
| DBO  | Dung Beetle Optimizer            | Xue & Shen, 2023            |
| COA  | Coati Optimization Algorithm     | Dehghani et al., 2023        |
| OOA  | Osprey Optimization Algorithm    | Dehghani & Trojovský, 2023   |
| NOA  | Nutcracker Optimization Algorithm | Abdel-Basset et al., 2023   |
| SAO  | Snow Ablation Optimizer          | Deng & Liu, 2023             |
| FLA  | Fick's Law Algorithm             | Hashim et al., 2023          |
| EVO  | Energy Valley Optimizer          | Azizi et al., 2023           |
| EDO  | Exponential Distribution Optimizer | Abdel-Basset et al., 2023  |
| MOA  | Mud Ring Algorithm               | Wang & Zhang, 2023           |
| CPO  | Crested Porcupine Optimizer      | Abdel-Basset et al., 2024   |
| PO   | Pufferfish Optimization          | Mohammadi et al., 2024       |
| FO   | Fox Optimizer                    | Mohammed & Rashid, 2023      |
| HO   | Hippopotamus Optimization        | Amiri et al., 2024           |
| KOA  | Komodo Algorithm                 | Suyanto et al., 2024         |
| SBOA | Secretary Bird Optimization      | Wang et al., 2024            |
| GMO  | Geometric Mean Optimizer         | Rezaei et al., 2023          |
| FFO  | Fennec Fox Optimization          | Trojovská et al., 2024       |

### Convenience Algorithm Lists

```python
from heurilab.algorithms import (
    SWARM_ALGORITHMS,        # 20 (name, class) tuples
    EVOLUTIONARY_ALGORITHMS, # 15
    PHYSICS_ALGORITHMS,      # 16
    HUMAN_ALGORITHMS,        # 14
    BIO_ALGORITHMS,          # 15
    MODERN_ALGORITHMS,       # 20
    ALL_ALGORITHMS,          # 100 (all combined)
)

# Run all 100 algorithms
run_experiment(
    algorithms=ALL_ALGORITHMS,
    benchmark_suites=[get_classical_suite()],
    ...
)
```

---

## 📐 Benchmark Suites

### Classical Benchmark Functions (F1–F23)

Standard test suite from **Yao et al. (1999)**, widely used in metaheuristic literature.

```python
from heurilab import F1, F2, F3, ..., F23
from heurilab import get_classical_suite, get_unimodal_suite, get_multimodal_suite, get_fixeddim_suite
```

#### Unimodal Functions (F1–F7)

| Function | Name            | Bounds         | Dim | Optimum |
|----------|-----------------|----------------|-----|---------|
| `F1`     | Sphere          | [-100, 100]    | 30  | 0       |
| `F2`     | Schwefel 2.22   | [-10, 10]      | 30  | 0       |
| `F3`     | Schwefel 1.2    | [-100, 100]    | 30  | 0       |
| `F4`     | Schwefel 2.21   | [-100, 100]    | 30  | 0       |
| `F5`     | Rosenbrock      | [-30, 30]      | 30  | 0       |
| `F6`     | Step            | [-100, 100]    | 30  | 0       |
| `F7`     | Quartic (Noise) | [-1.28, 1.28]  | 30  | 0       |

#### Multimodal Functions (F8–F13)

| Function | Name          | Bounds       | Dim | Optimum          |
|----------|---------------|-------------|-----|------------------|
| `F8`     | Schwefel 2.26 | [-500, 500] | 30  | -12569.5         |
| `F9`     | Rastrigin     | [-5.12, 5.12] | 30 | 0               |
| `F10`    | Ackley        | [-32, 32]   | 30  | 0                |
| `F11`    | Griewank      | [-600, 600] | 30  | 0                |
| `F12`    | Penalized 1   | [-50, 50]   | 30  | 0                |
| `F13`    | Penalized 2   | [-50, 50]   | 30  | 0                |

#### Fixed-Dimension Multimodal Functions (F14–F23)

| Function | Name             | Bounds             | Dim |
|----------|------------------|--------------------|-----|
| `F14`    | Shekel Foxholes  | [-65.536, 65.536]  | 2   |
| `F15`    | Kowalik          | [-5, 5]            | 4   |
| `F16`    | Six-Hump Camel   | [-5, 5]            | 2   |
| `F17`    | Branin           | [-5, 15]           | 2   |
| `F18`    | Goldstein-Price  | [-2, 2]            | 2   |
| `F19`    | Hartmann 3-D     | [0, 1]             | 3   |
| `F20`    | Hartmann 6-D     | [0, 1]             | 6   |
| `F21`    | Shekel 5         | [0, 10]            | 4   |
| `F22`    | Shekel 7         | [0, 10]            | 4   |
| `F23`    | Shekel 10        | [0, 10]            | 4   |

#### Pre-built Suite Constructors

| Function                | Returns                       |
|------------------------|-------------------------------|
| `get_classical_suite()` | All 23 functions (F1–F23)     |
| `get_unimodal_suite()`  | Unimodal only (F1–F7)        |
| `get_multimodal_suite()` | Multimodal only (F8–F13)     |
| `get_fixeddim_suite()`   | Fixed-dim only (F14–F23)     |

---

### CEC 2017 Benchmark Functions (29 Functions)

Based on the **CEC 2017** competition (Awad et al., 2016). F2 is excluded per official spec. All functions use **seeded shift vectors** — the global optimum is NOT at the origin. Search range: **[-100, 100]^D**.

```python
from heurilab import CEC17_F1, CEC17_F3, ..., CEC17_F30
from heurilab import get_cec2017_suite
```

#### Unimodal (2 functions)

| Function    | Name            | F_min |
|-------------|-----------------|-------|
| `CEC17_F1`  | Bent Cigar      | 100   |
| `CEC17_F3`  | Zakharov        | 300   |

#### Simple Multimodal (7 functions)

| Function     | Name                    | F_min |
|-------------|-------------------------|-------|
| `CEC17_F4`  | Rosenbrock              | 400   |
| `CEC17_F5`  | Rastrigin               | 500   |
| `CEC17_F6`  | Expanded Scaffer's F6   | 600   |
| `CEC17_F7`  | Lunacek Bi-Rastrigin    | 700   |
| `CEC17_F8`  | Non-Cont. Rastrigin     | 800   |
| `CEC17_F9`  | Levy                    | 900   |
| `CEC17_F10` | Schwefel                | 1000  |

#### Hybrid (10 functions)

| Function     | Name      | Sub-functions                                          | F_min |
|-------------|-----------|--------------------------------------------------------|-------|
| `CEC17_F11` | Hybrid 1  | Zakharov, Rosenbrock, Rastrigin                        | 1100  |
| `CEC17_F12` | Hybrid 2  | Elliptic, Schwefel, Bent Cigar                         | 1200  |
| `CEC17_F13` | Hybrid 3  | Bent Cigar, Rosenbrock, Lunacek                        | 1300  |
| `CEC17_F14` | Hybrid 4  | Elliptic, Ackley, Schwefel, Rastrigin                  | 1400  |
| `CEC17_F15` | Hybrid 5  | Bent Cigar, HGBat, Rastrigin, Rosenbrock               | 1500  |
| `CEC17_F16` | Hybrid 6  | Exp. Scaffer, HGBat, Rosenbrock, Schwefel              | 1600  |
| `CEC17_F17` | Hybrid 7  | Katsuura, Ackley, Griewank, Schwefel, Rastrigin        | 1700  |
| `CEC17_F18` | Hybrid 8  | Elliptic, Ackley, Schwefel, Rastrigin, Bent Cigar      | 1800  |
| `CEC17_F19` | Hybrid 9  | HappyCat, Katsuura, Ackley, Rastrigin, Schwefel, Rosen | 1900  |
| `CEC17_F20` | Hybrid 10 | HGBat, Katsuura, Ackley, Rastrigin, Schwefel, Scaffer  | 2000  |

#### Composition (10 functions)

| Function     | Name           | Components | F_min |
|-------------|----------------|------------|-------|
| `CEC17_F21` | Composition 1  | 3          | 2100  |
| `CEC17_F22` | Composition 2  | 3          | 2200  |
| `CEC17_F23` | Composition 3  | 4          | 2300  |
| `CEC17_F24` | Composition 4  | 4          | 2400  |
| `CEC17_F25` | Composition 5  | 5          | 2500  |
| `CEC17_F26` | Composition 6  | 5          | 2600  |
| `CEC17_F27` | Composition 7  | 6          | 2700  |
| `CEC17_F28` | Composition 8  | 6          | 2800  |
| `CEC17_F29` | Composition 9  | 7          | 2900  |
| `CEC17_F30` | Composition 10 | 8          | 3000  |

#### Pre-built CEC 2017 Suite Constructors

| Function                          | Returns                         |
|----------------------------------|---------------------------------|
| `get_cec2017_suite()`             | All 29 functions                |
| `get_cec2017_unimodal_suite()`    | F1, F3 (2 functions)            |
| `get_cec2017_multimodal_suite()`  | F4–F10 (7 functions)            |
| `get_cec2017_hybrid_suite()`      | F11–F20 (10 functions)          |
| `get_cec2017_composition_suite()` | F21–F30 (10 functions)          |

---

## 🔬 Enhancement Advisor

The enhancement advisor runs diagnostic benchmarks on any algorithm and provides performance scores, detected weaknesses, ranked enhancement suggestions with code snippets, and diagnostic plots.

```python
from heurilab.analyzer import enhance, cec2017_benchmarks
```

### enhance()

```python
result = enhance(
    algorithm=("MyAlgo", MyAlgo),
    output_dir="enhance_report",
    pop_size=50,
    max_iter=200,
    n_runs=10,
    benchmarks=None,
)
```

**Parameters:**

| Parameter    | Type                    | Default            | Description                                              |
|-------------|-------------------------|--------------------|----------------------------------------------------------|
| `algorithm`  | `Tuple[str, Type]`      | —                  | `(name, AlgorithmClass)` to analyze                      |
| `output_dir` | `str`                   | `"enhance_report"` | Directory to save report and plots                       |
| `pop_size`   | `int`                   | `50`               | Population size for diagnostic runs                      |
| `max_iter`   | `int`                   | `200`              | Max iterations for diagnostic runs                       |
| `n_runs`     | `int`                   | `10`               | Number of runs per diagnostic test                       |
| `benchmarks` | `dict` or `None`        | `None`             | Custom benchmarks. If `None`, uses built-in 6-test suite |

**Returns:** `dict` with keys:
- `"scores"` — Performance scores (0–100) for each metric
- `"weaknesses"` — List of detected weaknesses
- `"suggestions"` — Ranked enhancement suggestions
- `"report_text"` — Full formatted text report

**Built-in Diagnostic Suite** (6 tests, used when `benchmarks=None`):

| Test Key      | Function           | Dim | Purpose                      |
|---------------|--------------------|-----|------------------------------|
| `unimodal`    | F1 Sphere          | 30  | Exploitation quality         |
| `rosenbrock`  | F5 Rosenbrock      | 30  | Valley navigation            |
| `multimodal`  | F9 Rastrigin       | 30  | Local optima escape          |
| `ackley`      | F10 Ackley         | 30  | Multimodal performance       |
| `griewank`    | F11 Griewank       | 30  | Multimodal (lower difficulty) |
| `highdim`     | F1 Sphere          | 100 | Scalability                  |

**Custom Benchmarks:**

```python
benchmarks = {
    "test1": {
        "name": "My Function",
        "func": my_func,       # Callable: f(x) → float
        "lb": -100,
        "ub": 100,
        "dim": 30,
        "optimum": 0.0,
    },
}
result = enhance(("MyAlgo", MyAlgo), benchmarks=benchmarks)
```

**Output Files:**

| File                            | Description                    |
|---------------------------------|--------------------------------|
| `{AlgoName}_report.txt`        | Full text report with scores, weaknesses, suggestions |
| `{AlgoName}_radar.png`         | Radar chart of performance profile |
| `{AlgoName}_scores.png`        | Horizontal bar chart of scores |
| `{AlgoName}_convergence.png`   | Convergence curves for all diagnostic tests |
| `{AlgoName}_diversity.png`     | Population diversity curves    |

---

### cec2017_benchmarks()

```python
from heurilab.analyzer import cec2017_benchmarks

cec_tests = cec2017_benchmarks(dim=30)  # Returns dict for all 29 CEC 2017 functions
result = enhance(("MyAlgo", MyAlgo), benchmarks=cec_tests)
```

Returns a dict formatted for `enhance(benchmarks=...)` with all 29 CEC 2017 functions, including their known optima.

---

### Diagnostic Metrics (Scored 0–100)

| Metric                | Weight | Description                                             |
|-----------------------|--------|---------------------------------------------------------|
| Exploitation          | 20%    | How close to known optimum on unimodal (F1)            |
| Exploration           | 10%    | Initial population diversity maintenance                |
| Local Optima Escape   | 20%    | Performance on multimodal (F9 Rastrigin)               |
| Convergence Speed     | 10%    | Fraction of improvement in first 25% of iterations     |
| Stability             | 10%    | Inverse coefficient of variation across runs            |
| Stagnation Resistance | 10%    | How late stagnation begins relative to max_iter         |
| Scalability           | 10%    | Performance on high-dim (F1, d=100)                    |
| Multimodal            | 5%     | Average across Rastrigin, Ackley, Griewank             |
| Valley Navigation     | 5%     | Performance on Rosenbrock                               |
| **Overall**           | —      | Weighted sum of all above                               |

**Score Interpretation:**

| Score Range | Label    |
|------------|----------|
| ≥ 70       | Strong   |
| 50–69      | OK       |
| 35–49      | Weak     |
| < 35       | Poor     |

---

### Enhancement Knowledge Base (10 Techniques)

When weaknesses are detected (score < threshold), the advisor suggests applicable enhancements ranked by impact:

| # | Enhancement                          | Fixes                        | Impact |
|---|--------------------------------------|------------------------------|--------|
| 1 | Chaotic Inertia / Step-Size Decay    | exploitation, convergence    | ★★★★★  |
| 2 | Lévy Flight Mutation                 | local_optima_escape, multimodal | ★★★★★ |
| 3 | Opposition-Based Learning (OBL)      | exploration, local_optima    | ★★★★   |
| 4 | Stagnation Escape (Pop Restart)      | stagnation, local_optima     | ★★★★   |
| 5 | Gaussian Local Search                | exploitation, convergence    | ★★★★   |
| 6 | Adaptive Parameter Control (SHADE)   | exploration, exploitation    | ★★★    |
| 7 | Elite Preservation with Perturbation | stability, exploitation      | ★★★    |
| 8 | Dimensional Learning                 | highdim, scalability         | ★★★    |
| 9 | Boundary Reflection                  | exploration, boundary_bias   | ★★     |
| 10| Cauchy Mutation for Best Solution     | local_optima, exploitation   | ★★★    |

Each suggestion includes a description and ready-to-use Python code snippet.

---

### How to Use the Enhancement Advisor — Step by Step

#### Step 1: Run a Quick Diagnostic on Your Algorithm

```python
from heurilab.analyzer import enhance
from heurilab.algorithms import PSO   # or your own algorithm

result = enhance(
    algorithm=("PSO", PSO),
    output_dir="enhance_report_pso",
    pop_size=50,
    max_iter=200,
    n_runs=10,
)
```

This runs 6 built-in diagnostic tests (Sphere, Rosenbrock, Rastrigin, Ackley, Griewank, High-Dim Sphere) and generates a full report.

#### Step 2: Read the Scores

```python
scores = result["scores"]
print(f"Overall:            {scores['overall']:.1f}/100")
print(f"Exploitation:       {scores['exploitation']:.1f}/100")
print(f"Exploration:        {scores['exploration']:.1f}/100")
print(f"Local Optima Escape:{scores['local_optima_escape']:.1f}/100")
print(f"Convergence Speed:  {scores['convergence_speed']:.1f}/100")
print(f"Stability:          {scores['stability']:.1f}/100")
print(f"Stagnation:         {scores['stagnation']:.1f}/100")
print(f"Scalability:        {scores['scalability']:.1f}/100")
print(f"Multimodal:         {scores['multimodal']:.1f}/100")
print(f"Valley Navigation:  {scores['valley_navigation']:.1f}/100")
```

#### Step 3: Check Detected Weaknesses

```python
if result["weaknesses"]:
    print("⚠️  Weaknesses detected:")
    for w in result["weaknesses"]:
        print(f"  - {w}")
else:
    print("✅ No major weaknesses found!")
```

#### Step 4: Read Enhancement Suggestions

```python
for i, suggestion in enumerate(result["suggestions"], 1):
    print(f"\n{'='*60}")
    print(f"Suggestion #{i}: {suggestion['name']}")
    print(f"Impact: {'★' * suggestion['impact']}")
    print(f"Description: {suggestion['description']}")
    print(f"Code:\n{suggestion['code']}")
```

#### Step 5: Check the Generated Files

After running, your output directory will contain:

```
enhance_report_pso/
├── PSO_report.txt          ← Full text report (human-readable)
├── PSO_radar.png           ← Radar chart showing strengths/weaknesses
├── PSO_scores.png          ← Bar chart of all metric scores
├── PSO_convergence.png     ← Convergence curves across diagnostic tests
└── PSO_diversity.png       ← Population diversity over iterations
```

The `_report.txt` file includes everything: scores, weaknesses, suggestions with code snippets — ready for your research notes.

#### Step 6: Deep Analysis with CEC 2017

For a more thorough analysis using all 29 CEC 2017 competition functions:

```python
from heurilab.analyzer import enhance, cec2017_benchmarks

result = enhance(
    algorithm=("PSO", PSO),
    output_dir="enhance_report_pso_cec2017",
    benchmarks=cec2017_benchmarks(dim=30),
    pop_size=30,
    max_iter=500,
    n_runs=10,
)
```

#### Step 7: Analyze Your Own Custom Algorithm

```python
from heurilab.analyzer import enhance

# Your custom algorithm (must extend _Base)
from my_algorithm import MyOptimizer

result = enhance(
    algorithm=("MyOptimizer", MyOptimizer),
    output_dir="enhance_report_myopt",
    pop_size=50,
    max_iter=300,
    n_runs=15,
)

# Then apply the suggested fixes and re-analyze:
# result_v2 = enhance(("MyOptimizer_v2", MyOptimizerV2), output_dir="enhance_report_v2")
# Compare scores to see improvement!
```

#### Step 8: Use Custom Benchmarks

You can define your own diagnostic tests:

```python
import numpy as np

benchmarks = {
    "sphere_10d": {
        "name": "Sphere (d=10)",
        "func": lambda x: np.sum(x**2),
        "lb": -100, "ub": 100, "dim": 10,
        "optimum": 0.0,
    },
    "sphere_50d": {
        "name": "Sphere (d=50)",
        "func": lambda x: np.sum(x**2),
        "lb": -100, "ub": 100, "dim": 50,
        "optimum": 0.0,
    },
    "rastrigin_50d": {
        "name": "Rastrigin (d=50)",
        "func": lambda x: 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x)),
        "lb": -5.12, "ub": 5.12, "dim": 50,
        "optimum": 0.0,
    },
}

result = enhance(
    algorithm=("PSO", PSO),
    output_dir="enhance_custom",
    benchmarks=benchmarks,
    pop_size=40,
    max_iter=200,
    n_runs=10,
)
```

#### Complete Enhancement Workflow Example

```python
from heurilab.analyzer import enhance, cec2017_benchmarks
from heurilab.algorithms import GWO

# ── Phase 1: Quick diagnostic ────────────────────────
result = enhance(("GWO", GWO), output_dir="gwo_analysis/diagnostic")
print(f"Overall Score: {result['scores']['overall']:.1f}/100")
print(f"Weaknesses: {result['weaknesses']}")
print(f"Top suggestion: {result['suggestions'][0]['name']}")

# ── Phase 2: CEC 2017 deep analysis ──────────────────
result_cec = enhance(
    ("GWO", GWO),
    output_dir="gwo_analysis/cec2017",
    benchmarks=cec2017_benchmarks(dim=30),
    pop_size=30,
    max_iter=500,
    n_runs=10,
)

# ── Phase 3: Apply fixes → re-analyze improved version ─
# (After implementing suggested enhancements in GWO_v2)
# result_v2 = enhance(("GWO_v2", GWO_v2), output_dir="gwo_analysis/v2")
# Compare: result["scores"]["overall"] vs result_v2["scores"]["overall"]
```

---

## 📤 Exporters Module

### CSV Export

Three CSV files written in real-time during experiments:

**`results.csv`** — Summary statistics per algorithm-function combination:

| Column     | Description           |
|------------|-----------------------|
| Benchmark  | Function name         |
| Algorithm  | Algorithm name        |
| Mean       | Mean best fitness     |
| Std        | Standard deviation    |
| Best       | Best (minimum) fitness|
| Worst      | Worst (maximum) fitness|
| Median     | Median fitness        |

**`raw_runs.csv`** — Individual run data:

| Column      | Description                    |
|-------------|--------------------------------|
| Benchmark   | Function name                  |
| Algorithm   | Algorithm name                 |
| Run         | Run index (1-based)            |
| BestFitness | Best fitness of this run       |
| Time_s      | Elapsed time in seconds        |
| Conv_0..N   | Convergence curve values       |

**`convergence.csv`** — Mean convergence curve per algorithm-function:

| Column      | Description                    |
|-------------|--------------------------------|
| Benchmark   | Function name                  |
| Algorithm   | Algorithm name                 |
| Iter_0..N   | Mean convergence at each iteration |

---

### Plots

**Convergence Curves** — One PNG per benchmark function:

- All algorithms plotted on the same figure
- Y-axis uses `symlog` scale (handles both large and small values)
- The proposed algorithm (first in list) is plotted last with thicker line (2.5px vs 1.5px)
- Markers every 10% of iterations for distinguishability
- 20 distinct colors and markers for up to 20 algorithms

**Box Plots** — One PNG per benchmark function:

- Side-by-side box plots with means shown as diamond markers
- Proposed algorithm highlighted with red border and higher opacity
- Auto-switches to `symlog` scale when range spans >1000×

---

### Excel Export

Three styled Excel workbooks:

**`Results.xlsx`** — Performance comparison:

- One sheet per benchmark suite category (Classical, CEC2017, etc.)
- "All Functions" combined sheet
- "Ranking" sheet with average rank and win count per algorithm
- Best mean per function highlighted in green
- Proposed algorithm column highlighted in blue

**`Wilcoxon.xlsx`** — Pairwise statistical tests:

- One sheet per category with p-values and winners
- Color-coded: 🟢 green = proposed wins, 🔴 red = proposed loses, 🟡 yellow = tie
- "Summary" sheet with total W/T/L counts per competitor

**`Friedman.xlsx`** — Multi-algorithm ranking:

- "Mean Ranks" sheet with Friedman χ² and p-value per category + overall
- "Nemenyi Post-Hoc" sheet with pairwise rank differences
- Color-coded: 🟢 green = not significantly different, 🔴 red = significantly different

---

## 📊 Stats Module

```python
from heurilab.stats.tests import wilcoxon_test, friedman_test, nemenyi_cd, nemenyi_pairwise
```

### Wilcoxon Rank-Sum Test

```python
p_value, winner = wilcoxon_test(proposed_vals, competitor_vals, alpha=0.05)
```

- Uses `scipy.stats.ranksums` (two-sided)
- Returns winner as `"proposed"`, `"competitor"`, or `"tie"` 
- Winner determined by mean comparison when p < alpha

### Friedman Test

```python
chi2, p_value, mean_ranks = friedman_test(all_data, algo_names, func_names)
```

- `all_data[func_name][algo_name]` = list of fitness values
- Ranks algorithms per function using `scipy.stats.rankdata`
- Returns mean rank per algorithm (lower = better)

### Nemenyi Post-Hoc Test

```python
cd = nemenyi_cd(n_algos, n_funcs, alpha=0.05)
# Critical Difference = q_alpha * sqrt(k(k+1) / (6N))

pairwise = nemenyi_pairwise(mean_ranks, n_funcs, algo_names)
# pairwise[(algo_i, algo_j)] = (rank_diff, is_significant)
```

Supports 2–20 algorithms with pre-tabulated q-values.

---

## 🏭 Engineering Module

Three constrained engineering design problems with **penalty-based constraint handling** (penalty = 10¹⁰ per violated constraint).

```python
from heurilab.engineering.problems import ENGINEERING_PROBLEMS
```

| Problem                       | Variables | Bounds                                                      | Constraints |
|-------------------------------|-----------|--------------------------------------------------------------|-------------|
| Pressure Vessel Design        | 4         | [0.0625, 6.1875] × [0.0625, 6.1875] × [10, 200] × [10, 200] | 4           |
| Welded Beam Design            | 4         | [0.1, 2] × [0.1, 10] × [0.1, 10] × [0.1, 2]                | 7           |
| Tension/Compression Spring    | 3         | [0.05, 2] × [0.25, 1.3] × [2, 15]                           | 4           |

Runs automatically when `run_engineering=True` in `run_experiment()`. Only the **proposed algorithm** (first in list) is evaluated.

**Engineering output** (`Engineering.xlsx`) includes per problem:
- Summary sheet (Best, Mean, Worst, Std, Median)
- Optimal variables sheet
- All 30 runs sheet
- Convergence data sheet
- Convergence plot + box plot

---

## 📁 Output Structure

```
output/
├── CSV Data/
│   ├── results.csv              # Mean, Std, Best, Worst, Median per algo × function
│   ├── raw_runs.csv             # Per-run fitness + convergence (saved in real-time)
│   └── convergence.csv          # Mean convergence curves
├── Convergence Curves/
│   ├── F1_Sphere.png            # One PNG per benchmark function
│   ├── F2_Schwefel2.22.png
│   └── ...
├── Box Plots/
│   ├── F1_Sphere.png            # One PNG per benchmark function
│   └── ...
└── Excel Files/
    ├── Results.xlsx             # Per-suite sheets + Ranking sheet
    ├── Wilcoxon.xlsx            # Pairwise p-values + W/T/L summary
    ├── Friedman.xlsx            # Mean ranks + Nemenyi post-hoc
    └── Engineering.xlsx         # Engineering design problem results
```

**Key features:**
- 🔄 **Real-time CSV saving** — `raw_runs.csv` is updated after every single run
- 📈 **Live progress bar** — tqdm shows current algorithm × benchmark
- 🎨 **Publication-ready plots** — convergence curves and box plots auto-generated
- 📊 **Statistical tests** — Wilcoxon and Friedman with p-values in Excel

---

## 📚 Complete Examples

### Example 1: Quick Comparison of 3 Algorithms

```python
from heurilab import run_experiment, get_unimodal_suite
from heurilab.algorithms import PSO, GWO, DE

run_experiment(
    algorithms=[("PSO", PSO), ("GWO", GWO), ("DE", DE)],
    benchmark_suites=[get_unimodal_suite()],
    output_dir="output",
    pop_size=30, max_iter=100, dim=10, n_runs=5,
    run_engineering=False,
)
```

### Example 2: Full Research Experiment

```python
from heurilab import run_experiment, get_classical_suite, get_cec2017_suite
from heurilab.algorithms import (
    PSO, GWO, WOA, HHO, MPA, SSA,
    DE, GA, ES,
    GSA, MVO, SCA, AOA,
    TLBO, JA,
    ABC, FA, SOS,
    CPO, HO,
)

algorithms = [
    ("CPO", CPO),      # Proposed (first = highlighted)
    ("PSO", PSO), ("GWO", GWO), ("WOA", WOA),
    ("HHO", HHO), ("MPA", MPA), ("SSA", SSA),
    ("DE", DE), ("GA", GA), ("ES", ES),
    ("GSA", GSA), ("MVO", MVO), ("SCA", SCA), ("AOA", AOA),
    ("TLBO", TLBO), ("JA", JA),
    ("ABC", ABC), ("FA", FA), ("SOS", SOS),
    ("HO", HO),
]

run_experiment(
    algorithms=algorithms,
    benchmark_suites=[get_classical_suite(), get_cec2017_suite()],
    output_dir="output",
    pop_size=50,
    max_iter=300,
    dim=30,
    n_runs=30,
    run_engineering=True,
    engineering_n_runs=30,
)
```

### Example 3: Using Modern 2024 Algorithms

```python
from heurilab import run_experiment, get_classical_suite
from heurilab.algorithms import CPO, PO, FO, HO, KOA, SBOA, GMO, FFO

algorithms = [
    ("CPO", CPO),    # Crested Porcupine Optimizer (2024)
    ("PO", PO),      # Parrot Optimizer (2024)
    ("FO", FO),      # Fox Optimizer (2024)
    ("HO", HO),      # Hippopotamus Optimization (2024)
    ("KOA", KOA),    # Kepler Optimization (2024)
    ("SBOA", SBOA),  # Secretary Bird Optimization (2024)
    ("GMO", GMO),    # Geometric Mean Optimizer (2024)
    ("FFO", FFO),    # Fennec Fox Optimization (2024)
]

run_experiment(
    algorithms=algorithms,
    benchmark_suites=[get_classical_suite()],
    output_dir="output_2024",
    pop_size=50, max_iter=300, dim=30, n_runs=30,
)
```

### Example 4: Enhancement Analysis Pipeline

```python
from heurilab.analyzer import enhance, cec2017_benchmarks
from heurilab.algorithms import PSO

# Step 1: Quick diagnostic (6 built-in tests)
result = enhance(
    algorithm=("PSO", PSO),
    output_dir="analysis/diagnostic",
    pop_size=50,
    max_iter=200,
    n_runs=10,
)
print(f"Overall Score: {result['scores']['overall']}/100")
print(f"Weaknesses: {result['weaknesses']}")

# Step 2: CEC 2017 deep analysis
result = enhance(
    algorithm=("PSO", PSO),
    output_dir="analysis/cec2017",
    benchmarks=cec2017_benchmarks(dim=30),
    pop_size=30,
    max_iter=500,
    n_runs=10,
)
```

---

## 🔬 Creating Custom Algorithms

Create your own algorithm in minutes:

```python
import numpy as np
from heurilab.algorithms.base import _Base

class MyOptimizer(_Base):
    def optimize(self):
        # Initialize population
        X = self._init_pop()                              # shape: (pop_size, dim)
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        # Track best
        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]
        convergence = [best_fit]

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                # --- Your update logic here ---
                new_x = self._clip(X[i] + np.random.randn(self.dim) * 0.1)
                new_fit = self._eval(new_x)
                if new_fit < fitness[i]:
                    X[i] = new_x
                    fitness[i] = new_fit
                    if new_fit < best_fit:
                        best = new_x.copy()
                        best_fit = new_fit
            convergence.append(best_fit)

        return best, best_fit, convergence
```

**Key rules:**
1. Use `self._init_pop()` for random initialization
2. Use `self._eval(x)` instead of `self.obj_func(x)` directly (enables tracking)
3. Use `self._clip(x)` for boundary enforcement
4. Return `(best_solution, best_fitness, convergence_list)`
5. Convergence list should have `max_iter + 1` entries (initial + one per iteration)

**Then use it:**

```python
from heurilab import run_experiment, get_classical_suite
from heurilab.analyzer import enhance
from heurilab.algorithms import PSO, GWO

# Enhancement analysis
result = enhance(("MyOpt", MyOptimizer))

# Full experiment
run_experiment(
    algorithms=[("MyOpt", MyOptimizer), ("PSO", PSO), ("GWO", GWO)],
    benchmark_suites=[get_classical_suite()],
    output_dir="output",
)
```

---

## 📋 Dependencies

| Package      | Purpose                                    |
|--------------|--------------------------------------------|
| `numpy`      | Numerical computation                      |
| `scipy`      | Statistical tests (Wilcoxon, Friedman)     |
| `matplotlib` | Convergence curves, box plots, radar charts |
| `openpyxl`   | Styled Excel workbook generation            |
| `tqdm`       | Progress bars during experiments            |

All dependencies are auto-installed via `pip install heurilab`.

---

## 🤝 Contributing

Contributions are welcome! Areas for contribution:
- Additional optimization algorithms
- More benchmark functions (CEC 2019, CEC 2022)
- Performance optimizations
- Documentation improvements
- Bug fixes

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 💬 Community

- 🐛 **Found a bug?** [Open an issue](https://github.com/arartawil/heurilab/issues/new)
- 💡 **Have an idea?** [Request a feature](https://github.com/arartawil/heurilab/issues/new)
- 💬 **Questions?** [Join discussions](https://github.com/arartawil/heurilab/discussions)
- 📧 **Email:** arartawil@gmail.com

---

## 🏆 Used By

HeuriLab is trusted by researchers and engineers worldwide:

- 🎓 **Universities:** Research institutions using HeuriLab for optimization research
- 🏢 **Industry:** Engineering teams leveraging metaheuristics for real-world problems
- 📊 **Publications:** Growing number of papers use HeuriLab for benchmarking

*Using HeuriLab? [Let us know!](mailto:arartawil@gmail.com)*

---

## 📄 Citation

If you use HeuriLab in your research, please cite:

```bibtex
@software{heurilab2025,
  title={HeuriLab: A Comprehensive Python Framework for Metaheuristic Optimization},
  author={Artawil, A. R.},
  year={2025},
  url={https://github.com/arartawil/heurilab},
  note={Python package with 100 metaheuristic algorithms and automated experiment runner}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- The original authors of all 100 metaheuristic algorithms
- NumPy, SciPy, Matplotlib teams for scientific computing tools
- The metaheuristic optimization research community

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=arartawil/heurilab&type=Date)](https://star-history.com/#arartawil/heurilab&Date)

**If HeuriLab helps your research, please ⭐ star the repo and cite our work!**

---

**HeuriLab** — The complete laboratory for metaheuristic optimization research 🚀
