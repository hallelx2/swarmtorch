"""One-paste Kaggle / Colab benchmark runner.

Paste the entire contents of this file into a single Kaggle (or Colab)
code cell, run it, and you'll get a zipped results bundle covering
every standardised benchmark we need for the paper:

  Stage 4.1   bench-synthetic     dimensionality-wall study
  Stage 4.2   bench-training      MNIST + CIFAR-10 NN training
  Stage 4.3   bench-hpo           HPO comparison vs TPE / Hyperband
  Stage 4.4   ablation-init       Kaiming-init falsifiable test
  Stage 4.4   ablation-swarm      swarm-size sweep
  Stage 6     bench-gpu           swarmtorch GPU vs NumPy headline

The hardware fingerprint (CPU, GPU, RAM, OS, Python, PyTorch + CUDA
versions) is printed at the top of the run AND embedded in every
result JSON. When you paste a result back later, the machine that
produced it is fully identified.

Prerequisites on Kaggle:
  Settings -> Accelerator: GPU T4 x2 (or P100)
  Settings -> Internet: ON
  Save & Run All (or run this cell)

Runtime estimate on Kaggle T4: ~4-6 hours total for the --all suite.
Kaggle's hard session limit is 12 hours, so this fits comfortably.
"""

import os
import shutil
import subprocess
import sys
import textwrap
import time

REPO_URL = "https://github.com/hallelx2/swarmtorch.git"
REPO_DIR = "swarmtorch"  # cloned directory name on Kaggle's /kaggle/working/


def run(cmd, **kwargs):
    """Run a shell command, stream output, raise on failure."""
    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.call(cmd, shell=True, **kwargs)
    if rc != 0:
        raise SystemExit(f"Command failed (rc={rc}): {cmd}")


def section(title):
    bar = "#" * 72
    print(f"\n{bar}\n# {title}\n{bar}", flush=True)


# ---------------------------------------------------------------- 1. Clone

section("1/6 Clone latest swarmtorch from GitHub master")
if os.path.isdir(REPO_DIR):
    print(f"{REPO_DIR}/ already exists -- pulling latest")
    run(f"git -C {REPO_DIR} fetch --all")
    run(f"git -C {REPO_DIR} reset --hard origin/master")
else:
    run(f"git clone {REPO_URL} {REPO_DIR}")

os.chdir(REPO_DIR)
run("git log --oneline -1")


# ---------------------------------------------------------------- 2. Install

section("2/6 Install with [benchmark,cmaes] extras")
# Kaggle's PyTorch is already CUDA-built; pip-install just adds our deps.
run(f'{sys.executable} -m pip install -q -e ".[benchmark,cmaes]"')


# ---------------------------------------------------------------- 3. Identify

section("3/6 Hardware fingerprint")
run(f"{sys.executable} -c \"from swarmtorch.benchmark import print_banner; print_banner()\"")


# ---------------------------------------------------------------- 4. Smoke

section("4/6 Smoke test (~1 minute)")
# Quick check that the install + GPU + everything wires up before
# we kick off the multi-hour run. Fails fast on broken environments.
run(f"{sys.executable} scripts/cloud_bench.py --quick --include-gpu")


# ---------------------------------------------------------------- 5. Full run

section("5/6 Full standardised benchmark suite")
print("This is the long one. Expect ~4-6 hours on T4.")
print("Sweeps:")
print("  ablation-init    -- Kaiming init falsifiable test")
print("  ablation-swarm   -- swarm-size sweep")
print("  bench-hpo        -- HPO comparison (Random / TPE / Hyperband)")
print("  bench-synthetic  -- dimensionality-wall (5 funcs x 3 dims x 13 algos x 5 seeds)")
print("  bench-gpu        -- swarmtorch GPU vs NumPy (the headline)")
print("  bench-training   -- MNIST + CIFAR-10 NN training")
print(flush=True)

t0 = time.perf_counter()
run(f"{sys.executable} scripts/cloud_bench.py --all")
print(f"\nFull suite wall-clock: {(time.perf_counter() - t0) / 3600:.2f} hours")


# ---------------------------------------------------------------- 6. Bundle

section("6/6 Bundle results for download")
bundle_name = f"swarmtorch_results_{time.strftime('%Y%m%d_%H%M')}"
print(f"Bundling -> /kaggle/working/{bundle_name}.zip")

# Include both the quick smoke output and the real run; ignore the
# repo source itself.
shutil.make_archive(
    f"/kaggle/working/{bundle_name}",
    "zip",
    root_dir=".",
    base_dir="results",
)
print(f"Bundle ready: /kaggle/working/{bundle_name}.zip")
print("\nDownload from Kaggle: Right pane -> Output tab -> swarmtorch_results_*.zip")
print("Or use: from IPython.display import FileLink; FileLink(path)")
print(textwrap.dedent("""
    ====================================================================
      DONE. Every result JSON in this bundle has the machine
      fingerprint embedded under meta.hardware so when you share the
      bundle, the runtime is fully identified.
    ====================================================================
"""))
