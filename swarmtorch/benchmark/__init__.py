"""Benchmark harness for fair, multi-seed, FE-budgeted comparisons.

This sub-package implements the methodology described in the Stage 2
plan: function-evaluation budget enforcement, multi-seed runs with
deterministic seeding, statistical tests (Friedman / Nemenyi / Wilcoxon),
convergence curves with std bands, and aggregated reports with sensible
precision.
"""

from swarmtorch.benchmark.budget import BudgetExceeded, FEBudgetTracker
from swarmtorch.benchmark.hardware import hardware_info, print_banner
from swarmtorch.benchmark.run import (
    BenchmarkConfig,
    RunResult,
    load_results,
    run_one,
    seed_everything,
)
from swarmtorch.benchmark.stats import (
    friedman_test,
    nemenyi_critical_difference,
    wilcoxon_test,
)
from swarmtorch.benchmark.plots import (
    convergence_plot,
    critical_difference_diagram,
)
from swarmtorch.benchmark.report import build_report, aggregate_results

__all__ = [
    "BudgetExceeded",
    "FEBudgetTracker",
    "BenchmarkConfig",
    "RunResult",
    "load_results",
    "run_one",
    "seed_everything",
    "hardware_info",
    "print_banner",
    "friedman_test",
    "nemenyi_critical_difference",
    "wilcoxon_test",
    "convergence_plot",
    "critical_difference_diagram",
    "build_report",
    "aggregate_results",
]
