#!/usr/bin/env bash
# Bash convenience wrapper around scripts/cloud_bench.py.
#
# Use the .py file directly if you want; this exists only because some users
# (and agents) reach for ``bash scripts/cloud_bench.sh`` by reflex on Linux
# cloud VMs. Args are forwarded verbatim:
#
#   bash scripts/cloud_bench.sh                  # default sweeps
#   bash scripts/cloud_bench.sh --quick          # smoke
#   bash scripts/cloud_bench.sh --include-gpu    # add GPU benchmark
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/cloud_bench.py "$@"
