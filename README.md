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

**[Get Started in 30 seconds →](#-installation)**

---

## 🎯 HeuriLab vs Others

| Feature | HeuriLab | MEALPY | PySwarms | SciPy |
|---------|----------|--------|----------|-------|
| **Algorithms** | ✅ 100 | ~200 | PSO only | Few |
| **Benchmarks** | ✅ 52 (Classical + CEC 2017) | Limited | None | None |
| **Automated Runner** | ✅ One function call | ❌ Manual | ❌ Manual | ❌ Manual |
| **CSV + Excel Export** | ✅ Real-time | ❌ | ❌ | ❌ |
| **Statistical Tests** | ✅ Wilcoxon + Friedman | ❌ | ❌ | ❌ |
| **Convergence Plots** | ✅ Auto-generated | Manual | Manual | ❌ |
| **Box Plots** | ✅ Auto-generated | ❌ | ❌ | ❌ |
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
- [100 Built-in Algorithms](#-100-built-in-algorithms)
- [Benchmark Suites](#-benchmark-suites)
- [Algorithm Interface](#-algorithm-interface)
- [Output Structure](#-output-structure)
- [Complete Examples](#-complete-examples)
- [Creating Custom Algorithms](#-creating-custom-algorithms)
- [Engineering Design Problems](#-engineering-design-problems)
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

## 🧠 100 Built-in Algorithms

### Swarm Intelligence (20)

| Abbr | Algorithm | Year |
|------|-----------|------|
| PSO | Particle Swarm Optimization | 1995 |
| GWO | Grey Wolf Optimizer | 2014 |
| WOA | Whale Optimization Algorithm | 2016 |
| MFO | Moth-Flame Optimization | 2015 |
| SSA | Salp Swarm Algorithm | 2017 |
| HHO | Harris Hawks Optimization | 2019 |
| MPA | Marine Predators Algorithm | 2020 |
| BA | Bat Algorithm | 2010 |
| CS | Cuckoo Search | 2009 |
| FPA | Flower Pollination Algorithm | 2012 |
| DA | Dragonfly Algorithm | 2016 |
| GOA | Grasshopper Optimization | 2017 |
| ALO | Ant Lion Optimizer | 2015 |
| SHO | Spotted Hyena Optimizer | 2017 |
| DO | Dragonfly Optimizer | 2016 |
| EHO | Elephant Herding Optimization | 2016 |
| AO | Aquila Optimizer | 2021 |
| HGS | Hunger Games Search | 2021 |
| GTO | Giant Trevally Optimizer | 2022 |
| RUN | RUNge Kutta Optimizer | 2021 |

### Evolutionary (15)

| Abbr | Algorithm | Year |
|------|-----------|------|
| GA | Genetic Algorithm | 1975 |
| DE | Differential Evolution | 1997 |
| ES | Evolution Strategies | 1973 |
| EP | Evolutionary Programming | 1966 |
| CMA | CMA-ES | 2001 |
| BBO | Biogeography-Based Optimization | 2008 |
| SHADE | Success-History Adaptive DE | 2013 |
| TLGO | Teaching-Learning-GO | 2016 |
| CoDE | Composite DE | 2011 |
| SaDE | Self-adaptive DE | 2009 |
| OXDE | Orthogonal-Crossover DE | 2014 |
| AGDE | Adaptive Guided DE | 2017 |
| LSHADE | L-SHADE | 2014 |
| EBOwithCMAR | EBO with CMA Restart | 2017 |
| IMODE | Improved Multi-Operator DE | 2020 |

### Physics-based (16)

| Abbr | Algorithm | Year |
|------|-----------|------|
| GSA | Gravitational Search Algorithm | 2009 |
| MVO | Multi-Verse Optimizer | 2016 |
| SCA | Sine Cosine Algorithm | 2016 |
| AOA | Arithmetic Optimization Algorithm | 2021 |
| SA | Simulated Annealing | 1983 |
| EO | Equilibrium Optimizer | 2020 |
| WDO | Wind Driven Optimization | 2010 |
| HGSO | Henry Gas Solubility Optimization | 2019 |
| CSS | Charged System Search | 2010 |
| CFO | Central Force Optimization | 2007 |
| TWO | Tug of War Optimization | 2016 |
| ASO | Atom Search Optimization | 2019 |
| RIME | RIME Optimization | 2023 |
| AEO | Artificial Ecosystem Optimization | 2020 |
| GBO | Gradient-Based Optimizer | 2020 |
| TSO | Transient Search Optimization | 2021 |

### Human/Social (14)

| Abbr | Algorithm | Year |
|------|-----------|------|
| TLBO | Teaching-Learning-Based Optimization | 2011 |
| JA | Jaya Algorithm | 2016 |
| HS | Harmony Search | 2001 |
| ICA | Imperialist Competitive Algorithm | 2007 |
| CA | Cultural Algorithm | 1994 |
| BSO | Brain Storm Optimization | 2011 |
| SOS_H | Symbiotic Organisms Search | 2014 |
| QLA | Quantum-inspired Learning Automata | 2018 |
| INFO | Weighted Mean of Vectors | 2022 |
| HBO | Heap-Based Optimizer | 2020 |
| AOArch | Archimedes Optimization | 2021 |
| CHIO | Coronavirus Herd Immunity Optimizer | 2021 |
| SSOA | Social Ski-Driver Optimization | 2019 |
| POA | Pelican Optimization Algorithm | 2022 |

### Bio-inspired (15)

| Abbr | Algorithm | Year |
|------|-----------|------|
| ABC | Artificial Bee Colony | 2007 |
| FA | Firefly Algorithm | 2009 |
| SOS | Symbiotic Organisms Search | 2014 |
| BFO | Bacterial Foraging Optimization | 2002 |
| CSA | Crow Search Algorithm | 2016 |
| BOA | Butterfly Optimization Algorithm | 2019 |
| TSA | Tunicate Swarm Algorithm | 2020 |
| WHO | Wild Horse Optimizer | 2021 |
| SBO | Satin Bowerbird Optimizer | 2017 |
| MBO | Monarch Butterfly Optimization | 2019 |
| EPO | Emperor Penguin Optimizer | 2018 |
| SMA | Slime Mould Algorithm | 2020 |
| HBA | Honey Badger Algorithm | 2022 |
| RSA | Reptile Search Algorithm | 2022 |
| GJO | Golden Jackal Optimization | 2022 |

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

---

## 📐 Benchmark Suites

### Classical Benchmarks (F1–F23)

| # | Name | Type | Dim |
|---|------|------|-----|
| F1 | Sphere | Unimodal | 30 |
| F2 | Schwefel 2.22 | Unimodal | 30 |
| F3 | Schwefel 1.2 | Unimodal | 30 |
| F4 | Schwefel 2.21 | Unimodal | 30 |
| F5 | Rosenbrock | Unimodal | 30 |
| F6 | Step | Unimodal | 30 |
| F7 | Quartic + Noise | Unimodal | 30 |
| F8 | Schwefel 2.26 | Multimodal | 30 |
| F9 | Rastrigin | Multimodal | 30 |
| F10 | Ackley | Multimodal | 30 |
| F11 | Griewank | Multimodal | 30 |
| F12 | Penalized 1 | Multimodal | 30 |
| F13 | Penalized 2 | Multimodal | 30 |
| F14–F23 | Fixed-dimension functions | Fixed-dim | 2–6 |

### CEC 2017 (29 Functions)

F1, F3–F30 with shifted/rotated optima for unbiased evaluation.

```python
from heurilab import (
    get_classical_suite,    # All 23 classical
    get_unimodal_suite,     # F1–F7
    get_multimodal_suite,   # F8–F13
    get_fixeddim_suite,     # F14–F23
    get_cec2017_suite,      # All 29 CEC 2017
)
```

---

## 🏗️ Algorithm Interface

All 100 algorithms share a clean, unified interface:

```python
from heurilab.algorithms.base import _Base

class MyAlgorithm(_Base):
    def optimize(self):
        X = self._init_pop()               # Initialize population
        fitness = [self._eval(x) for x in X]  # Evaluate
        convergence = []

        for t in range(self.max_iter):
            # ... your optimization logic ...
            X = self._clip(X)              # Enforce bounds
            convergence.append(best_fit)

        return best_solution, best_fitness, convergence
```

**Base class provides:**
- `self.pop_size`, `self.dim`, `self.lb`, `self.ub`, `self.max_iter`, `self.obj_func`
- `self._init_pop()` — random uniform initialization
- `self._clip(x)` — boundary enforcement
- `self._eval(x)` — objective evaluation with optional progress callback

---

## 📁 Output Structure

```
output/
├── CSV Data/
│   ├── results.csv              # Mean, Std, Best, Worst, Median per algo × function
│   ├── raw_runs.csv             # Per-run fitness + convergence (saved in real-time)
│   └── convergence.csv          # Mean convergence curves
├── Convergence Curves/          # One PNG per benchmark function
├── Box Plots/                   # One PNG per benchmark function
└── Excel Files/
    ├── Results.xlsx             # Styled summary with color-coded rankings
    ├── Wilcoxon.xlsx            # Wilcoxon signed-rank p-values
    ├── Friedman.xlsx            # Friedman test + Nemenyi post-hoc
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

---

## 🔬 Creating Custom Algorithms

Create your own algorithm in minutes:

```python
import numpy as np
from heurilab.algorithms.base import _Base

class MyOptimizer(_Base):
    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()
        best_fit = fitness[best_idx]
        convergence = [best_fit]

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                # Your optimization logic here
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

Then use it directly in experiments:

```python
run_experiment(
    algorithms=[("MyOpt", MyOptimizer), ("PSO", PSO), ("GWO", GWO)],
    benchmark_suites=[get_classical_suite()],
    output_dir="output",
)
```

---

## 🏭 Engineering Design Problems

HeuriLab includes built-in engineering design problems (Welded Beam, Pressure Vessel, Tension/Compression Spring) that run automatically when `run_engineering=True`:

```python
run_experiment(
    algorithms=algorithms,
    benchmark_suites=[get_classical_suite()],
    output_dir="output",
    run_engineering=True,
    engineering_pop_size=50,
    engineering_max_iter=500,
    engineering_n_runs=30,
)
```

Results are saved to `output/Excel Files/Engineering.xlsx`.

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
