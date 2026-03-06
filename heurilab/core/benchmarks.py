"""
Benchmark function configurations and suites.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class BenchmarkConfig:
    """Single benchmark function configuration."""
    name: str
    obj_func: Callable
    lb: float
    ub: float
    dim: int = 30

    def __repr__(self):
        return f"BenchmarkConfig(name='{self.name}', dim={self.dim}, lb={self.lb}, ub={self.ub})"


@dataclass
class BenchmarkSuite:
    """A named collection of benchmark functions (e.g. 'Classical', 'CEC2017')."""
    category: str
    benchmarks: List[BenchmarkConfig] = field(default_factory=list)

    def add(self, name: str, obj_func: Callable, lb: float, ub: float, dim: int = 30):
        self.benchmarks.append(BenchmarkConfig(name, obj_func, lb, ub, dim))
        return self

    def __iter__(self):
        return iter(self.benchmarks)

    def __len__(self):
        return len(self.benchmarks)

    def __repr__(self):
        return f"BenchmarkSuite(category='{self.category}', n_benchmarks={len(self.benchmarks)})"
