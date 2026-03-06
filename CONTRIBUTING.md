# Contributing to HeuriLab

Thank you for your interest in contributing to HeuriLab! We welcome contributions of all kinds — bug fixes, new algorithms, documentation improvements, and more.

## Getting Started

1. **Fork** the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/heurilab.git
   cd heurilab
   ```

2. **Install** the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Adding a New Algorithm

1. Choose the appropriate category folder under `heurilab/algorithms/` (e.g., `swarm/`, `evolutionary/`, `bio/`, `physics/`, `human/`).
2. Create a new file for your algorithm.
3. Your algorithm class must implement the standard interface:
   ```python
   class MyAlgorithm:
       def __init__(self, pop_size, dim, lb, ub, max_iter, obj_func):
           ...

       def optimize(self):
           # Returns: (best_solution, best_fitness, convergence_list)
           return best_solution, best_fitness, convergence_list
   ```
4. Register your algorithm in the category's `__init__.py`.

## Code Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) style conventions.
- Keep code simple and readable.
- Use NumPy for numerical operations.
- Ensure `convergence_list` has `max_iter + 1` entries (index 0 = initial fitness).

## Submitting Changes

1. Commit your changes with clear, descriptive messages:
   ```bash
   git commit -m "Add XYZ algorithm to swarm category"
   ```
2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
3. Open a **Pull Request** against the `main` branch with a clear description of your changes.

## Reporting Issues

- Use the GitHub issue tracker to report bugs or request features.
- Include steps to reproduce, expected behavior, and actual behavior.
- Mention your Python version and OS.

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive environment for everyone.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
