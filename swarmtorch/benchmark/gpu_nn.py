"""Multi-algorithm GPU speedup on a REAL neural-network fitness objective.

The headline gpu_vs_numpy benchmark isolates framework overhead on cheap
analytic test functions. This complementary benchmark answers the question
a reviewer asks next: does the GPU speedup (a) generalize across
algorithms, and (b) hold for the real use case -- evaluating a neural
network forward pass as the fitness function?

For each algorithm and model size we time an FE-budgeted run in three
modes:

* ``cpu-loop``  -- per-candidate Python loop (what NumPy libraries do).
* ``cpu-vmap``  -- swarmtorch's functional_call + vmap path on CPU.
* ``cuda-vmap`` -- the same path on CUDA (the real use case).

Fitness is the loss of an MLP forward pass over a fixed minibatch, so the
dominant cost is genuine neural-network matrix multiplications -- exactly
where GPUs help. Run on a CUDA machine (Kaggle/Colab) to populate the
cuda-vmap rows.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from swarmtorch import DE, GWO, PSO, WOA
from swarmtorch.evolutionary.model_training import CMAES
from swarmtorch.benchmark.hardware import hardware_info
from swarmtorch.benchmark.run import seed_everything

ALGOS = {"PSO": PSO, "GWO": GWO, "WOA": WOA, "DE": DE, "CMAES": CMAES}


def _build_mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


@dataclass
class NNBenchResult:
    algorithm: str
    n_params: int
    swarm_size: int
    variant: str          # cpu-loop | cpu-vmap | cuda-vmap
    device: str
    seed: int
    wall_seconds: float
    final_loss: float
    fe_used: int
    meta: dict = field(default_factory=dict)


def _run_one(algo_cls, hidden, swarm_size, max_fe, seed, device, use_vmap):
    seed_everything(seed)
    in_dim, out_dim, batch = 64, 10, 128
    model = _build_mlp(in_dim, hidden, out_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(batch, in_dim, device=device)
    y = torch.randint(0, out_dim, (batch,), device=device)

    # Build kwargs by introspection: some optimizers name the population
    # ``swarm_size`` (PSO/GWO/WOA/CMAES) and others ``population_size`` (DE/GA).
    import inspect

    sig = inspect.signature(algo_cls.__init__)
    params = sig.parameters
    accepts_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs: dict[str, Any] = {"device": device}
    if "swarm_size" in params or accepts_kw:
        kwargs["swarm_size"] = swarm_size
    elif "population_size" in params:
        kwargs["population_size"] = swarm_size
    if "init_strategy" in params or accepts_kw:
        kwargs["init_strategy"] = "model"
    opt = algo_cls(model.parameters(), **kwargs)

    if use_vmap:
        opt.set_functional_closure(model, lambda fwd: F.cross_entropy(fwd(x), y))
        closure = None
    else:
        closure = lambda: F.cross_entropy(model(x), y)

    fe, best = 0, float("inf")
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    while fe < max_fe:
        opt.step() if use_vmap else opt.step(closure)
        fe += swarm_size
        cur = getattr(opt, "global_best_fitness", getattr(opt, "best_fitness", None))
        if cur is not None:
            best = min(best, float(cur.item()))
    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return wall, best, n_params, fe


def run_nn_speedup(
    hiddens: list[int],
    algorithms: list[str],
    swarm_sizes: list[int],
    seeds: list[int],
    max_fe: int,
    output_dir: Path,
    include_cuda: bool = True,
    include_cpu_loop: bool = True,
) -> list[NNBenchResult]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cuda_ok = torch.cuda.is_available()
    if include_cuda and not cuda_ok:
        print("[gpu_nn] CUDA unavailable -- skipping cuda-vmap rows. Run on Kaggle/Colab GPU.")
        include_cuda = False

    variants = [("cpu-vmap", "cpu", True)]
    if include_cpu_loop:
        variants.insert(0, ("cpu-loop", "cpu", False))
    if include_cuda:
        variants.append(("cuda-vmap", "cuda", True))

    results: list[NNBenchResult] = []
    total = len(hiddens) * len(algorithms) * len(swarm_sizes) * len(seeds) * len(variants)
    cell = 0
    for h in hiddens:
        for algo in algorithms:
            for ss in swarm_sizes:
                for seed in seeds:
                    for vname, dev, use_vmap in variants:
                        cell += 1
                        try:
                            # warm-up (untimed)
                            _run_one(ALGOS[algo], h, ss, ss, seed, dev, use_vmap)
                            wall, best, npar, fe = _run_one(
                                ALGOS[algo], h, ss, max_fe, seed, dev, use_vmap)
                        except Exception as e:  # one bad cell must not kill the sweep
                            print(f"[gpu_nn] {cell}/{total} {algo} {vname} FAILED: {e}",
                                  flush=True)
                            continue
                        rec = NNBenchResult(
                            algorithm=algo, n_params=npar, swarm_size=ss,
                            variant=vname, device=dev, seed=seed,
                            wall_seconds=wall, final_loss=best, fe_used=fe,
                            meta={"hardware": hardware_info(), "hidden": h},
                        )
                        results.append(rec)
                        path = output_dir / f"{algo}__h{h}_s{ss}__{vname}__seed{seed}.json"
                        with open(path, "w", encoding="utf-8") as fp:
                            json.dump(asdict(rec), fp, indent=2)
                        print(f"[gpu_nn] {cell}/{total} {algo} h={h} pop={ss} "
                              f"{vname} {wall:.3f}s", flush=True)
    _write_report(results, output_dir)
    return results


def _write_report(results: list[NNBenchResult], output_dir: Path) -> Path:
    hw = hardware_info()
    from collections import defaultdict
    g = defaultdict(dict)
    for r in results:
        g[(r.algorithm, r.n_params, r.swarm_size)].setdefault(r.variant, []).append(r.wall_seconds)
    lines = [
        "# Multi-algorithm GPU speedup on a real NN objective",
        "",
        f"**Machine:** `{hw['hostname']}` -- {hw.get('cpu_model','?')} | "
        f"**GPU:** {hw.get('gpu_name') or 'none'} | PyTorch {hw.get('torch','?')}",
        "",
        "Speedup = (CPU vmap wall) / (CUDA vmap wall). cpu-loop shown for reference.",
        "",
        "| Algorithm | params | pop | cpu-loop (s) | cpu-vmap (s) | cuda-vmap (s) | CUDA speedup |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (algo, npar, ss) in sorted(g):
        d = {k: float(np.mean(v)) for k, v in g[(algo, npar, ss)].items()}
        loop = d.get("cpu-loop", float("nan"))
        cv = d.get("cpu-vmap", float("nan"))
        gv = d.get("cuda-vmap", float("nan"))
        sp = (cv / gv) if (gv and gv == gv and gv > 0) else float("nan")
        lines.append(f"| {algo} | {npar} | {ss} | {loop:.3f} | {cv:.3f} | "
                     f"{gv:.3f} | {sp:.2f}x |")
    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
