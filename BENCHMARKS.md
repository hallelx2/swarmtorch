# Running the swarmtorch benchmarks

This file documents how to reproduce the paper's empirical results.

## TL;DR — Kaggle (one paste, ~5 hours, fully standardised)

The cleanest way to produce paper-quality results: paste
`scripts/kaggle_run_all.py` into a single Kaggle code cell with a GPU
runtime. Everything runs end-to-end, every result JSON carries a
hardware fingerprint, and the final zip downloads from
`/kaggle/working/`.

1. Kaggle → New Notebook → Settings:
   * Accelerator: **GPU T4 x2** (or P100/V100 if you have access)
   * Internet: **On**
2. In a single cell, paste the entire contents of
   `scripts/kaggle_run_all.py` and run it.
3. Wait ~4–6 hours (well within Kaggle's 12-hour session limit).
4. Download `swarmtorch_results_<timestamp>.zip` from the Output tab.

This produces:

* Stage 4.1 dimensionality-wall study (synthetic functions)
* Stage 4.2 real-NN training (MNIST + CIFAR-10)
* Stage 4.3 HPO comparison (metaheuristics vs Random / TPE / Hyperband)
* Stage 4.4 init-strategy and swarm-size ablations
* Stage 6 swarmtorch GPU vs NumPy headline speedup

Every JSON includes `meta.hardware` with CPU model, GPU model, RAM,
OS, Python and PyTorch versions, so paper figures stay traceable to
the exact runtime that produced them.

## Other paths

* **Local / Linux with `make` installed** — use the Makefile.
* **Cloud agent (Jules), Colab, custom containers** — use
  `python scripts/cloud_bench.py` (no `make` dependency).

---

## Quick start

```bash
git clone https://github.com/hallelx2/swarmtorch.git
cd swarmtorch
pip install -e ".[benchmark,cmaes]"

# Verify install (~5 minutes):
python scripts/cloud_bench.py --quick

# Full CPU sweeps (~3–5 hours):
python scripts/cloud_bench.py
```

The orchestrator writes everything under `results/`. Each sweep
produces per-seed JSONs, a `report.md`, and convergence PNGs.

---

## What `cloud_bench.py` runs

By default, four sweeps in sequence:

| Sweep | Stage | What it tests |
|---|---|---|
| `ablation-init` | 4.4 | model vs uniform vs gaussian swarm initialization (~15 min) |
| `ablation-swarm` | 4.4 | swarm_size sweep {10, 30, 100, 300} (~20 min) |
| `bench-hpo` | 4.3 | metaheuristic searchers vs Random/TPE/Hyperband (~30 min) |
| `bench-synthetic` | 4.1 | dimensionality-wall study at d ∈ {10, 50, 200} (~2–4 hours) |

Override the default with `--sweeps`:

```bash
python scripts/cloud_bench.py --sweeps ablation-init bench-hpo
```

Add the GPU benchmark on a CUDA machine:

```bash
python scripts/cloud_bench.py --include-gpu
```

Smoke test the full pipeline in ~5 minutes:

```bash
python scripts/cloud_bench.py --quick
```

---

## Jules prompt (cloud agent)

Paste this verbatim into Jules:

> Clone the repo. Run `pip install -e ".[benchmark,cmaes]"`. Then
> run `python scripts/cloud_bench.py --quick` to verify the
> pipeline; if it succeeds, run `python scripts/cloud_bench.py` for
> the full default sweeps. Expected wall-clock: ~3–5 hours on a CPU
> VM.
>
> When the orchestrator exits with code 0, commit the contents of
> `results/ablations/`, `results/hpo/`, and `results/synthetic/`
> (including each subdirectory's `report.md` and convergence PNGs)
> to a new branch named `bench/cpu-sweeps-<YYYY-MM-DD>`. Do NOT
> modify any files under `swarmtorch/`, `tests/`, `scripts/`, or
> `notebooks/`. Open a PR titled "Stage 4 CPU benchmark results".

`results/` is in `.gitignore`, so the agent must `git add -f` the
specific files it wants to commit (the orchestrator's own output is
already structured to support this).

---

## Colab / Kaggle (GPU sweeps)

The GPU-vs-NumPy headline benchmark needs CUDA, which Jules doesn't
provide. Run it on free Colab T4 instead:

1. Open https://colab.research.google.com → File → Open notebook →
   GitHub tab → `hallelx2/swarmtorch` →
   `notebooks/gpu_vs_numpy.ipynb`.
2. Runtime → Change runtime type → Hardware accelerator: **T4 GPU**.
3. Runtime → Run all.

Six cells: install, smoke run, headline grid (~10–30 min), view
report, plot speedup curves, bundle results for download.

---

## Reading the output

Each sweep produces a `results/<sweep>/report.md` with:

* **Per-task tables** — algorithm × (mean ± std final score, median,
  wall-clock seconds, peak memory MB).
* **Statistical tests block** — Friedman χ² + p-value (significance
  test on the global null), Nemenyi critical difference (alpha=0.05),
  average ranks per algorithm. Appears when k ≥ 3 algorithms × N ≥ 2
  tasks.
* **Convergence PNGs** — one per task, mean ± 1σ band per algorithm,
  log-y scale.

Per-seed JSONs are kept alongside for re-aggregation:

```bash
python -m swarmtorch.summarize_results results/synthetic
```

---

## Adding new sweeps

The orchestrator is just a dispatch table; to add a sweep:

1. Add a new entry to `SWEEPS` in `scripts/cloud_bench.py`.
2. Reference it in this file.
3. (Optional) Add a `make` target alias for users who prefer the
   Makefile.
