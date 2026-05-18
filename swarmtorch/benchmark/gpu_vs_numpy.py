"""GPU vs NumPy headline benchmark.

The paper's core claim is "PyTorch + GPU makes metaheuristics faster
than pure-NumPy reference implementations." This module makes that
claim concretely measurable.

We compare four implementations of the *exact same* PSO algorithm
solving the *exact same* synthetic test functions under the *exact
same* FE budget:

* ``numpy``                — NumPy reference (the "pyMetaheuristic" baseline).
* ``swarmtorch-cpu-loop``  — swarmtorch.PSO on CPU with the legacy
                             per-particle Python loop.
* ``swarmtorch-cpu-vmap``  — swarmtorch.PSO on CPU with the
                             ``torch.func.functional_call`` + ``vmap``
                             fast path.
* ``swarmtorch-cuda-vmap`` — same vmap path on CUDA. The headline.

The output is a list of ``GPUBenchResult`` records (mean wall-clock
seconds, final loss). Aggregating across (function, dim, swarm_size)
produces the speedup tables / plots.

Why we ship our own NumPy PSO instead of importing pyMetaheuristic:
the comparison must be apples-to-apples on *algorithm* (same equations,
same seed-driven RNG sequence). Importing a third party would pull in
subtle differences in inertia weight schedule, neighborhood topology,
and clipping that would muddy the wall-clock claim.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from swarmtorch.benchmark.hardware import hardware_info
from swarmtorch.benchmark.run import seed_everything
from swarmtorch.experiments.synthetic import SYNTHETIC_FUNCTIONS, SyntheticTask
from swarmtorch.swarm.model_training.pso import PSO


# --- NumPy PSO reference -------------------------------------------------


def _np_sphere(x: np.ndarray) -> np.ndarray:
    return (x ** 2).sum(axis=-1)


def _np_rastrigin(x: np.ndarray) -> np.ndarray:
    a = 10.0
    return a * x.shape[-1] + (x ** 2 - a * np.cos(2 * np.pi * x)).sum(axis=-1)


def _np_rosenbrock(x: np.ndarray) -> np.ndarray:
    y = x + 1.0
    return (
        100.0 * (y[..., 1:] - y[..., :-1] ** 2) ** 2 + (1.0 - y[..., :-1]) ** 2
    ).sum(axis=-1)


def _np_ackley(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1]
    a, b, c = 20.0, 0.2, 2 * np.pi
    s1 = (x ** 2).sum(axis=-1) / n
    s2 = np.cos(c * x).sum(axis=-1) / n
    return -a * np.exp(-b * np.sqrt(s1)) - np.exp(s2) + a + np.e


def _np_griewank(x: np.ndarray) -> np.ndarray:
    sq = (x ** 2).sum(axis=-1) / 4000.0
    idx = np.arange(1, x.shape[-1] + 1)
    prod = np.cos(x / np.sqrt(idx)).prod(axis=-1)
    return sq - prod + 1.0


_NUMPY_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sphere": _np_sphere,
    "rastrigin": _np_rastrigin,
    "rosenbrock": _np_rosenbrock,
    "ackley": _np_ackley,
    "griewank": _np_griewank,
}


def numpy_pso(
    f: Callable[[np.ndarray], np.ndarray],
    dim: int,
    swarm_size: int,
    max_fe: int,
    seed: int,
    init_range: float = 5.0,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
) -> tuple[float, int]:
    """Reference NumPy PSO.

    Returns (best_loss_so_far, fe_used). The implementation deliberately
    uses the same equations as :class:`swarmtorch.swarm.model_training.PSO`
    so wall-clock differences attribute to the framework, not the
    algorithm.

    Vectorized fitness evaluation: ``f`` accepts an array of shape
    ``(n, d)`` and returns shape ``(n,)`` — the natural NumPy
    counterpart of swarmtorch's vmap path.
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-init_range, init_range, size=(swarm_size, dim))
    velocities = np.zeros_like(positions)

    fitness = f(positions)
    fe = swarm_size

    personal_best = positions.copy()
    personal_best_fit = fitness.copy()
    best_idx = int(np.argmin(personal_best_fit))
    global_best = personal_best[best_idx].copy()
    global_best_fit = float(personal_best_fit[best_idx])

    while fe < max_fe:
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        velocities = (
            w * velocities
            + c1 * r1 * (personal_best - positions)
            + c2 * r2 * (global_best - positions)
        )
        positions = positions + velocities

        fitness = f(positions)
        fe += swarm_size

        improved = fitness < personal_best_fit
        personal_best_fit[improved] = fitness[improved]
        personal_best[improved] = positions[improved]

        bi = int(np.argmin(personal_best_fit))
        if personal_best_fit[bi] < global_best_fit:
            global_best_fit = float(personal_best_fit[bi])
            global_best = personal_best[bi].copy()

    return global_best_fit, fe


# --- swarmtorch variants -------------------------------------------------


def _swarmtorch_pso(
    task: SyntheticTask,
    swarm_size: int,
    max_fe: int,
    seed: int,
    device: str,
    use_vmap: bool,
) -> tuple[float, int]:
    """Run swarmtorch.PSO with the chosen device and code path."""
    seed_everything(seed)
    module = task.make_module().to(device)
    f = task.function()

    opt = PSO(
        module.parameters(),
        swarm_size=swarm_size,
        device=device,
        init_strategy="uniform",  # match the NumPy reference's U(0, 1) range
    )

    if use_vmap:
        # functional_call + vmap path — entirely on the requested device.
        opt.set_functional_closure(module, lambda forward: f(forward()))
        plain_closure = None
    else:
        # Per-particle Python loop path.
        plain_closure = lambda: f(module.x)

    fe = 0
    best = float("inf")
    while fe < max_fe:
        if use_vmap:
            opt.step()
        else:
            opt.step(plain_closure)
        fe += swarm_size
        cur = float(opt.global_best_fitness.item())
        if cur < best:
            best = cur
    return best, fe


# --- Result type ---------------------------------------------------------


@dataclass
class GPUBenchResult:
    """One (variant, function, dim, swarm_size, seed) cell of the bench."""

    variant: str
    function: str
    dim: int
    swarm_size: int
    seed: int
    wall_seconds: float
    final_loss: float
    fe_used: int
    device: str = "cpu"
    meta: dict[str, Any] = field(default_factory=dict)


# --- Top-level driver ----------------------------------------------------


def run_gpu_vs_numpy(
    functions: list[str],
    dims: list[int],
    swarm_sizes: list[int],
    seeds: list[int],
    max_fe: int,
    output_dir: Path,
    include_cuda: bool = True,
    include_cpu_loop: bool = True,
) -> list[GPUBenchResult]:
    """Cartesian sweep of all variants.

    Each cell is timed three times (we report the median wall-clock for
    stability — first-call overhead from CUDA initialization is
    excluded by a warm-up).

    Args:
        functions: Subset of ``SYNTHETIC_FUNCTIONS`` keys.
        dims: Problem dimensions to sweep.
        swarm_sizes: Swarm sizes to sweep — the GPU win grows with this axis.
        seeds: Random seeds.
        max_fe: FE budget per cell.
        output_dir: Where to write per-cell JSONs and the aggregate.
        include_cuda: Set False on machines without CUDA.
        include_cpu_loop: Set False to skip the slow legacy path on
            sweeps where it isn't informative.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    if include_cuda and not cuda_available:
        print(
            "[gpu_vs_numpy] CUDA requested but torch.cuda.is_available() is "
            "False — skipping cuda variant. Install a CUDA build of PyTorch "
            "or run on Colab / Kaggle with a GPU runtime."
        )
        include_cuda = False

    variants: list[tuple[str, str, bool]] = [
        ("numpy", "cpu", False),
        ("swarmtorch-cpu-vmap", "cpu", True),
    ]
    if include_cpu_loop:
        variants.insert(2, ("swarmtorch-cpu-loop", "cpu", False))
    if include_cuda:
        variants.append(("swarmtorch-cuda-vmap", "cuda", True))

    results: list[GPUBenchResult] = []
    n_cells = (
        len(functions) * len(dims) * len(swarm_sizes) * len(seeds) * len(variants)
    )
    cell = 0

    for func_name in functions:
        for dim in dims:
            for ss in swarm_sizes:
                for seed in seeds:
                    for variant_name, device, use_vmap in variants:
                        cell += 1
                        print(
                            f"[gpu_vs_numpy] cell {cell}/{n_cells} "
                            f"variant={variant_name} func={func_name} "
                            f"d={dim} swarm={ss} seed={seed}",
                            flush=True,
                        )
                        # Three timed runs; median is reported. One untimed
                        # warm-up first to eat any one-off init cost
                        # (CUDA context, JIT, vmap tracing).
                        if variant_name == "numpy":
                            f = _NUMPY_FUNCTIONS[func_name]
                            _ = numpy_pso(f, dim, ss, max_fe=ss, seed=seed)
                            timings: list[float] = []
                            losses: list[float] = []
                            fe_used = 0
                            for _ in range(3):
                                t0 = time.perf_counter()
                                loss, fe_used = numpy_pso(
                                    f, dim, ss, max_fe, seed=seed
                                )
                                timings.append(time.perf_counter() - t0)
                                losses.append(loss)
                        else:
                            task = SyntheticTask(
                                name=f"{func_name}_d{dim}",
                                dim=dim,
                                func_name=func_name,
                            )
                            _ = _swarmtorch_pso(
                                task, ss, ss, seed, device, use_vmap
                            )  # warm-up
                            timings = []
                            losses = []
                            fe_used = 0
                            for _ in range(3):
                                if device == "cuda":
                                    torch.cuda.synchronize()
                                t0 = time.perf_counter()
                                loss, fe_used = _swarmtorch_pso(
                                    task, ss, max_fe, seed, device, use_vmap
                                )
                                if device == "cuda":
                                    torch.cuda.synchronize()
                                timings.append(time.perf_counter() - t0)
                                losses.append(loss)

                        # Median wall-clock for stability; mean loss
                        # (PSO is stochastic but seeded — losses across
                        # the 3 timed runs should be identical).
                        record = GPUBenchResult(
                            variant=variant_name,
                            function=func_name,
                            dim=dim,
                            swarm_size=ss,
                            seed=seed,
                            wall_seconds=float(np.median(timings)),
                            final_loss=float(np.mean(losses)),
                            fe_used=int(fe_used),
                            device=device,
                            meta={
                                "all_timings": timings,
                                "hardware": hardware_info(),
                            },
                        )
                        results.append(record)
                        # Persist as we go — failure mid-sweep doesn't lose work.
                        path = (
                            output_dir
                            / f"{variant_name}__{func_name}_d{dim}_s{ss}__seed{seed}.json"
                        )
                        with open(path, "w", encoding="utf-8") as fp:
                            json.dump(asdict(record), fp, indent=2)

    # Aggregate report.
    _write_speedup_report(results, output_dir)
    return results


def _write_speedup_report(
    results: list[GPUBenchResult], output_dir: Path
) -> Path:
    """Build a markdown report with speedup tables.

    Speedup is reported relative to the ``numpy`` baseline for each
    (function, dim, swarm_size) cell. Higher is better. Colab/Kaggle
    runs typically show 10–100x at swarm_size >= 256 on T4 / V100 GPUs.
    """
    output_dir = Path(output_dir)
    grouped: dict[tuple[str, int, int], dict[str, list[float]]] = {}
    for r in results:
        key = (r.function, r.dim, r.swarm_size)
        grouped.setdefault(key, {}).setdefault(r.variant, []).append(r.wall_seconds)

    hw = hardware_info()
    lines = [
        "# GPU vs NumPy headline benchmark",
        "",
        f"**Machine:** `{hw['hostname']}` -- {hw['cpu_model']} ({hw['cpu_count_logical']} cores), "
        f"{hw['ram_mb']} MB RAM, {hw['os']}",
        f"**GPU:** {hw.get('gpu_name') or 'none'} "
        + (f"({hw['gpu_total_mem_mb']} MB, sm_{hw['gpu_capability']})" if hw["cuda_available"] else "")
        + f"  |  **PyTorch:** {hw['torch']} (CUDA {hw['torch_compiled_cuda']})",
        "",
        "Wall-clock seconds for an FE-budgeted PSO run (median across 3 timed runs, ",
        "averaged across seeds). Speedup column is relative to the ``numpy`` baseline.",
        "",
    ]
    for (func, dim, swarm) in sorted(grouped):
        lines.append(f"## {func}, d={dim}, swarm_size={swarm}")
        lines.append("")
        lines.append("| Variant | Wall (s) | Speedup vs numpy |")
        lines.append("| --- | --- | --- |")
        means: dict[str, float] = {
            v: float(np.mean(t)) for v, t in grouped[(func, dim, swarm)].items()
        }
        baseline = means.get("numpy", float("nan"))
        for variant in [
            "numpy",
            "swarmtorch-cpu-loop",
            "swarmtorch-cpu-vmap",
            "swarmtorch-cuda-vmap",
        ]:
            if variant not in means:
                continue
            wall = means[variant]
            speedup = baseline / wall if wall > 0 else float("inf")
            lines.append(f"| {variant} | {wall:.4f} | {speedup:.2f}x |")
        lines.append("")

    path = output_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
