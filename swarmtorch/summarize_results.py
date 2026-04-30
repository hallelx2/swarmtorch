"""Aggregate per-seed RunResult JSONs into a markdown report.

Replaces the old hardcoded-path script that read
``..\\test_tuning_results.txt`` in UTF-16-LE — that file is now a tree
of JSONs written by :func:`swarmtorch.benchmark.run.run_one`.

Usage:

    python -m swarmtorch.summarize_results results/synthetic
    python -m swarmtorch.summarize_results results/training --title "MNIST training"

Output: a single ``report.md`` in the same directory, containing per-task
mean +/- std tables, wall-clock and peak-memory columns, and a
Friedman + Nemenyi block whenever the grid is wide enough.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from swarmtorch.benchmark import build_report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing per-seed RunResult JSONs.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write report.md (defaults to results_dir).",
    )
    p.add_argument(
        "--title",
        type=str,
        default="swarmtorch benchmark report",
    )
    args = p.parse_args()

    out = args.output_dir or args.results_dir
    report = build_report(out, results_dir=args.results_dir, title=args.title)
    print(f"Report written to: {report}")


if __name__ == "__main__":
    main()
