# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-03-08

### Added
- **120+ Metaheuristic Algorithms**: Comprehensive collection across 6 categories:
  - Swarm Intelligence (PSO, GWO, HHO, SSA, etc.)
  - Evolutionary (GA, DE, CEM, etc.)
  - Physics-Based (SA, GSA, FPA, etc.)
  - Human-Based (TLBO, Harmony Search)
  - Bio-Inspired (ALO, BBO, MVO, etc.)
  - Hybrid Optimizers (SMA, Gorilla, etc.)
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
