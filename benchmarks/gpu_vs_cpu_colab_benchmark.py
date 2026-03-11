"""
SwarmTorch GPU vs CPU Benchmark
================================
Run this script on Google Colab (with GPU runtime enabled) to measure
the wall-clock speedup of GPU-accelerated population evaluation.

Instructions:
  1. Open Google Colab, select Runtime -> Change runtime type -> T4 GPU
  2. Install SwarmTorch:  !pip install swarmtorch
  3. Upload or paste this script, then run it
  4. Copy the FULL printed output and share it back

The benchmark measures:
  - Per-step wall-clock time for PSO, GWO, DE, GA, CEM across CPU and CUDA
  - Varying model sizes (small MLP, medium MLP, larger MLP)
  - Varying swarm/population sizes (10, 30, 50, 100)
  - Varying batch sizes for the training data (256, 1024, 4096)
  - Total throughput: evaluations per second
"""

import time
import json
import torch
import torch.nn as nn
import numpy as np

# ── Helpers ──────────────────────────────────────────────────────────

def make_model(input_dim, hidden_dims, output_dim):
    """Build an MLP with given hidden layer sizes."""
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def generate_data(n_samples, input_dim, device):
    """Binary classification data."""
    X = torch.randn(n_samples, input_dim, device=device)
    y = (X[:, 0] * X[:, 1] > 0).float().unsqueeze(1)
    return X, y


# ── Benchmark runner ─────────────────────────────────────────────────

def benchmark_optimizer(opt_class, opt_name, model_fn, input_dim,
                        hidden_dims, output_dim, batch_size,
                        swarm_size, n_steps, device):
    """Time n_steps of an optimizer on the given device. Returns dict."""

    dev = torch.device(device)

    # Build model and data
    model = model_fn(input_dim, hidden_dims, output_dim).to(dev)
    n_params = count_params(model)
    X, y = generate_data(batch_size, input_dim, dev)
    criterion = nn.BCEWithLogitsLoss()

    # Instantiate SwarmTorch optimizer
    try:
        optimizer = opt_class(
            model.parameters(),
            population_size=swarm_size,
            device=device,
        )
    except TypeError:
        try:
            optimizer = opt_class(
                model.parameters(),
                swarm_size=swarm_size,
                device=device,
            )
        except TypeError:
            # Some optimizers may use different kwarg names
            optimizer = opt_class(
                model.parameters(),
                device=device,
            )

    def closure():
        return criterion(model(X), y)

    # Warmup (1 step, not timed)
    try:
        optimizer.step(closure)
    except Exception as e:
        return {"error": str(e), "optimizer": opt_name, "device": device}

    # Synchronize before timing
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # Timed run
    times = []
    for _ in range(n_steps):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.step(closure)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    total_evals = swarm_size * n_steps  # approximate

    return {
        "optimizer": opt_name,
        "device": device,
        "n_params": n_params,
        "hidden_dims": hidden_dims,
        "batch_size": batch_size,
        "swarm_size": swarm_size,
        "n_steps": n_steps,
        "mean_step_time_s": float(np.mean(times)),
        "std_step_time_s": float(np.std(times)),
        "median_step_time_s": float(np.median(times)),
        "total_time_s": float(np.sum(times)),
        "evals_per_sec": float(total_evals / np.sum(times)),
    }


# ── Main benchmark suite ────────────────────────────────────────────

def run_full_benchmark():
    print("=" * 70)
    print("SwarmTorch GPU vs CPU Benchmark")
    print("=" * 70)

    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No GPU detected! Only CPU benchmarks will run.")
        print("Please enable GPU runtime in Colab: Runtime -> Change runtime type -> T4 GPU")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    # Import SwarmTorch optimizers
    from swarmtorch import PSO, GWO, DE, GA, HHO

    optimizers = [
        (PSO, "PSO"),
        (GWO, "GWO"),
        (DE,  "DE"),
        (GA,  "GA"),
        (HHO, "HHO"),
    ]

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    # ── Experiment 1: Varying model size ─────────────────────────────
    # Fixed: swarm_size=30, batch_size=1024, n_steps=10
    print("-" * 70)
    print("EXPERIMENT 1: Effect of Model Size")
    print("  Fixed: swarm_size=30, batch_size=1024, n_steps=10")
    print("-" * 70)

    model_configs = [
        {"name": "Small (2->8->1)",       "input": 2,  "hidden": [8],           "output": 1},
        {"name": "Medium (10->64->32->1)", "input": 10, "hidden": [64, 32],      "output": 1},
        {"name": "Large (50->256->128->64->1)", "input": 50, "hidden": [256, 128, 64], "output": 1},
        {"name": "XL (100->512->256->128->1)",  "input": 100, "hidden": [512, 256, 128], "output": 1},
    ]

    exp1_results = []

    for cfg in model_configs:
        dummy = make_model(cfg["input"], cfg["hidden"], cfg["output"])
        n_p = count_params(dummy)
        print(f"\n  Model: {cfg['name']} ({n_p:,} params)")

        for opt_class, opt_name in optimizers:
            for device in devices:
                result = benchmark_optimizer(
                    opt_class, opt_name, make_model,
                    cfg["input"], cfg["hidden"], cfg["output"],
                    batch_size=1024, swarm_size=30, n_steps=10,
                    device=device,
                )
                result["model_name"] = cfg["name"]
                exp1_results.append(result)

                if "error" in result:
                    print(f"    {opt_name:6s} [{device:4s}]: ERROR - {result['error']}")
                else:
                    print(f"    {opt_name:6s} [{device:4s}]: {result['mean_step_time_s']:.4f}s/step "
                          f"(median={result['median_step_time_s']:.4f}s, "
                          f"{result['evals_per_sec']:.0f} evals/s)")

    # ── Experiment 2: Varying swarm size ─────────────────────────────
    # Fixed: Medium model, batch_size=1024, n_steps=10
    print("\n" + "-" * 70)
    print("EXPERIMENT 2: Effect of Swarm/Population Size")
    print("  Fixed: Medium model (10->64->32->1), batch_size=1024, n_steps=10")
    print("-" * 70)

    swarm_sizes = [10, 30, 50, 100]
    exp2_results = []

    for ss in swarm_sizes:
        print(f"\n  Swarm size: {ss}")
        for opt_class, opt_name in optimizers:
            for device in devices:
                result = benchmark_optimizer(
                    opt_class, opt_name, make_model,
                    input_dim=10, hidden_dims=[64, 32], output_dim=1,
                    batch_size=1024, swarm_size=ss, n_steps=10,
                    device=device,
                )
                result["model_name"] = "Medium (10->64->32->1)"
                exp2_results.append(result)

                if "error" in result:
                    print(f"    {opt_name:6s} [{device:4s}] ss={ss:3d}: ERROR - {result['error']}")
                else:
                    print(f"    {opt_name:6s} [{device:4s}] ss={ss:3d}: {result['mean_step_time_s']:.4f}s/step "
                          f"({result['evals_per_sec']:.0f} evals/s)")

    # ── Experiment 3: Varying batch size ─────────────────────────────
    # Fixed: Large model, swarm_size=30, n_steps=10
    print("\n" + "-" * 70)
    print("EXPERIMENT 3: Effect of Training Batch Size")
    print("  Fixed: Large model (50->256->128->64->1), swarm_size=30, n_steps=10")
    print("-" * 70)

    batch_sizes = [256, 1024, 4096]
    exp3_results = []

    for bs in batch_sizes:
        print(f"\n  Batch size: {bs}")
        for opt_class, opt_name in [(PSO, "PSO"), (DE, "DE"), (HHO, "HHO")]:
            for device in devices:
                result = benchmark_optimizer(
                    opt_class, opt_name, make_model,
                    input_dim=50, hidden_dims=[256, 128, 64], output_dim=1,
                    batch_size=bs, swarm_size=30, n_steps=10,
                    device=device,
                )
                result["model_name"] = "Large (50->256->128->64->1)"
                exp3_results.append(result)

                if "error" in result:
                    print(f"    {opt_name:6s} [{device:4s}] bs={bs:5d}: ERROR - {result['error']}")
                else:
                    print(f"    {opt_name:6s} [{device:4s}] bs={bs:5d}: {result['mean_step_time_s']:.4f}s/step "
                          f"({result['evals_per_sec']:.0f} evals/s)")

    # ── Summary: Speedup ratios ──────────────────────────────────────
    if torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("SPEEDUP SUMMARY (GPU time / CPU time)")
        print("=" * 70)

        all_results = exp1_results + exp2_results + exp3_results

        # Group by experiment key and compute speedup
        from collections import defaultdict
        groups = defaultdict(dict)
        for r in all_results:
            if "error" in r:
                continue
            key = (r["optimizer"], r.get("model_name", ""), r["swarm_size"], r["batch_size"])
            groups[key][r["device"]] = r["mean_step_time_s"]

        print(f"\n  {'Optimizer':8s} {'Model':35s} {'SS':>4s} {'BS':>6s} {'CPU(s)':>8s} {'GPU(s)':>8s} {'Speedup':>8s}")
        print("  " + "-" * 100)
        for key, devs in sorted(groups.items()):
            if "cpu" in devs and "cuda" in devs:
                speedup = devs["cpu"] / devs["cuda"]
                opt, model, ss, bs = key
                print(f"  {opt:8s} {model:35s} {ss:4d} {bs:6d} {devs['cpu']:8.4f} {devs['cuda']:8.4f} {speedup:7.2f}x")

    # ── Dump raw JSON ────────────────────────────────────────────────
    all_data = {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__,
        "experiment_1_model_size": exp1_results,
        "experiment_2_swarm_size": exp2_results,
        "experiment_3_batch_size": exp3_results,
    }

    print("\n" + "=" * 70)
    print("RAW JSON RESULTS (copy everything between the markers)")
    print("=" * 70)
    print("<<<JSON_START>>>")
    print(json.dumps(all_data, indent=2))
    print("<<<JSON_END>>>")

    return all_data


if __name__ == "__main__":
    run_full_benchmark()
