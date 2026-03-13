# Changelog

All notable changes to this project will be documented in this file.

## [0.1.9] - 2026-03-10

### Fixed
- IWOA bug: Fixed undefined variable `l` (should be `l_val`)
- IGWO bug: Fixed discarded expressions for `A2`, `C2` calculation
- SOS bug: Fixed missing second random factor `r2` in mutualism
- DE bug: Fixed missing `j_rand` for mandatory crossover dimension
- HHO: Removed dead code from hot path
- GorillaTroopsOptimizer: Removed dead expression
- Removed dead `defaults = dict(...)` from 7 optimizer constructors

### Changed
- Refactored duplicated `_set_params`, `_get_params`, `_evaluate_fitness` methods into base `SwarmOptimizer` class (~400 lines removed)
- Moved `pandas` and `matplotlib` to optional dependencies (`pip install swarmtorch[benchmarks]`)
- Expanded top-level exports to include all 120 classes (60 optimizers + 60 HPO searchers)

### Added
- `GorillaTroopsOptimizerSearch` for complete 60+60 algorithm coverage

## [0.1.0] - 2026-03-08

### Added
- **120 Metaheuristic Algorithms** (60 model-training + 60 HPO searchers):
  - Swarm Intelligence (PSO, GWO, HHO, SSA, etc.) - 32 optimizers
  - Evolutionary (GA, DE, CEM, etc.) - 8 optimizers
  - Physics-Based (SA, GSA, FPA, etc.) - 3 optimizers
  - Human-Based (TLBO, Harmony Search) - 2 optimizers
  - Bio-Inspired (ALO, BBO, MVO, etc.) - 5 optimizers
  - Hybrid Optimizers (SMA, Gorilla, etc.) - 10 optimizers
- **Gradient-Free Training**: Native PyTorch `Optimizer` interface for training models without backpropagation.
- **Advanced HPO Searchers**: Automated hyperparameter optimization with a standardized API.
- **Massive-Scale Benchmarks**: Research artifacts for all algorithms included in `benchmarks/`.
- **Production Visuals**: High-resolution scientific plots documenting library performance.
- **CI/CD Pipeline**: GitHub Actions for automated linting, testing, and PyPI deployment.

### Changed
- Standardized project structure into a standalone PyTorch package.
- Optimized numerical stability by switching to `BCEWithLogitsLoss` for metaheuristic training.

### References
- Core algorithmic references derived from the `pyMetaheuristic` library.
