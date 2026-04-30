"""Tests for the GPU vs NumPy headline benchmark (Stage 6)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from swarmtorch.benchmark.gpu_vs_numpy import (
    _NUMPY_FUNCTIONS,
    GPUBenchResult,
    numpy_pso,
    run_gpu_vs_numpy,
)


@pytest.mark.parametrize("name", list(_NUMPY_FUNCTIONS))
def test_numpy_function_zero_at_origin(name):
    f = _NUMPY_FUNCTIONS[name]
    val = float(f(np.zeros((1, 20)))[0])
    assert abs(val) < 1e-3, f"{name}(0) = {val}"


def test_numpy_pso_minimizes_sphere():
    """The reference NumPy PSO must converge on Sphere — otherwise it's not a valid baseline."""
    f = _NUMPY_FUNCTIONS["sphere"]
    initial, _ = numpy_pso(f, dim=10, swarm_size=20, max_fe=20, seed=0)
    final, _ = numpy_pso(f, dim=10, swarm_size=20, max_fe=2000, seed=0)
    assert final < initial * 0.1, (
        f"NumPy PSO failed to make progress on Sphere: initial={initial:.3f} final={final:.3f}"
    )


def test_run_gpu_vs_numpy_smoke(tmp_path: Path):
    """Tiny grid that runs in a few seconds; verifies harness produces results + report."""
    results = run_gpu_vs_numpy(
        functions=["sphere"],
        dims=[10],
        swarm_sizes=[16],
        seeds=[0],
        max_fe=200,
        output_dir=tmp_path,
        include_cuda=False,  # CI doesn't have CUDA
        include_cpu_loop=True,
    )
    # 1 cell per variant: numpy + cpu-vmap + cpu-loop = 3 results
    variants = sorted({r.variant for r in results})
    assert variants == [
        "numpy",
        "swarmtorch-cpu-loop",
        "swarmtorch-cpu-vmap",
    ]
    # Each result has positive wall time and a finite final loss.
    for r in results:
        assert isinstance(r, GPUBenchResult)
        assert r.wall_seconds > 0
        assert np.isfinite(r.final_loss)
        assert r.fe_used >= 200
    # Report and per-cell JSONs were written.
    assert (tmp_path / "report.md").exists()
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_speedup_report_includes_numpy_baseline(tmp_path: Path):
    run_gpu_vs_numpy(
        functions=["sphere"],
        dims=[10],
        swarm_sizes=[16, 32],
        seeds=[0],
        max_fe=160,
        output_dir=tmp_path,
        include_cuda=False,
        include_cpu_loop=False,
    )
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "numpy" in text
    assert "swarmtorch-cpu-vmap" in text
    assert "Speedup vs numpy" in text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_variant_runs(tmp_path: Path):
    """Sanity check the CUDA path when run on a GPU machine."""
    results = run_gpu_vs_numpy(
        functions=["sphere"],
        dims=[10],
        swarm_sizes=[32],
        seeds=[0],
        max_fe=200,
        output_dir=tmp_path,
        include_cuda=True,
        include_cpu_loop=False,
    )
    cuda_results = [r for r in results if r.variant == "swarmtorch-cuda-vmap"]
    assert len(cuda_results) == 1
    assert cuda_results[0].device == "cuda"
    assert np.isfinite(cuda_results[0].final_loss)
