"""
GPU Acceleration Benchmark
=======================
Demonstrates the key advantage of SwarmTorch: PyTorch tensor operations
running on GPU can handle massive swarms efficiently.

This benchmark compares scaling behavior:
- swarmtorch (PyTorch GPU): Should scale linearly due to vectorization
- Pure Python/NumPy: Will scale poorly due to for-loops
"""

import torch
import numpy as np
import time
from typing import Dict
import matplotlib.pyplot as plt


class GPUSpeedBenchmark:
    """Benchmark GPU acceleration of swarm algorithms."""

    def __init__(self, dim: int = 100):
        self.dim = dim

    def rastrigin(self, x: torch.Tensor) -> torch.Tensor:
        """Rastrigin function for testing."""
        A = 10
        return A * self.dim + torch.sum(x**2 - A * torch.cos(2 * torch.pi * x), dim=-1)

    def benchmark_swarmtorch(
        self, swarm_size: int, n_iterations: int, device: str = "cuda"
    ) -> Dict:
        """Benchmark swarmtorch with given swarm size."""

        from swarmtorch import PSO

        # Check if CUDA is available
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            print("  Warning: CUDA not available, using CPU")

        # Initialize
        x = torch.nn.Parameter(torch.zeros(self.dim, device=device))
        optimizer = PSO([x], swarm_size=swarm_size, device=device)

        # Warm up
        for _ in range(3):
            _ = optimizer.step(lambda: torch.tensor(1.0))

        # Benchmark
        start = time.perf_counter()

        for _ in range(n_iterations):

            def closure():
                loss = self.rastrigin(x.unsqueeze(0))
                return loss

            optimizer.step(closure)

        elapsed = time.perf_counter() - start

        # Calculate throughput
        total_evaluations = swarm_size * n_iterations
        throughput = total_evaluations / elapsed

        return {
            "swarm_size": swarm_size,
            "iterations": n_iterations,
            "time": elapsed,
            "evaluations_per_second": throughput,
            "device": device,
        }

    def benchmark_numpy(self, swarm_size: int, n_iterations: int) -> Dict:
        """Benchmark pure numpy implementation (baseline)."""

        # Simple PSO-like implementation in numpy
        positions = np.random.randn(swarm_size, self.dim)
        velocities = np.zeros((swarm_size, self.dim))
        personal_best = positions.copy()
        personal_best_fitness = np.full(swarm_size, np.inf)
        global_best = np.zeros(self.dim)
        global_best_fitness = np.inf

        w, c1, c2 = 0.7, 1.5, 1.5

        def rastrigin_np(x):
            A = 10
            return A * self.dim + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=-1)

        # Warm up
        for _ in range(3):
            fitness = rastrigin_np(positions)

        # Benchmark
        start = time.perf_counter()

        for _ in range(n_iterations):
            # Evaluate
            fitness = rastrigin_np(positions)

            # Update bests
            improved = fitness < personal_best_fitness
            personal_best_fitness[improved] = fitness[improved]
            personal_best[improved] = positions[improved]

            best_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[best_idx] < global_best_fitness:
                global_best_fitness = personal_best_fitness[best_idx]
                global_best = personal_best[best_idx].copy()

            # Update velocities and positions
            r1 = np.random.rand(swarm_size, self.dim)
            r2 = np.random.rand(swarm_size, self.dim)

            velocities = (
                w * velocities
                + c1 * r1 * (personal_best - positions)
                + c2 * r2 * (global_best - positions)
            )
            positions = positions + velocities

        elapsed = time.perf_counter() - start

        total_evaluations = swarm_size * n_iterations
        throughput = total_evaluations / elapsed

        return {
            "swarm_size": swarm_size,
            "iterations": n_iterations,
            "time": elapsed,
            "evaluations_per_second": throughput,
            "device": "numpy",
        }

    def run_scaling_benchmark(self):
        """Run scaling comparison."""

        print("=" * 60)
        print("GPU ACCELERATION BENCHMARK")
        print("=" * 60)
        print(f"Dimension: {self.dim}")
        print("Iterations: 50")
        print("=" * 60)

        swarm_sizes = [10, 50, 100, 500, 1000]
        n_iterations = 50

        results = {"swarmtorch_gpu": [], "swarmtorch_cpu": [], "numpy": []}

        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"\nCUDA available: {cuda_available}")
        if cuda_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        print("\n" + "-" * 60)
        print(f"{'Swarm Size':>12} | {'NumPy':>12} | {'CPU':>12} | {'GPU':>12}")
        print("-" * 60)

        for size in swarm_sizes:
            # NumPy baseline
            print(f"{size:>12} ", end="|")
            np_result = self.benchmark_numpy(size, n_iterations)
            results["numpy"].append(np_result)
            print(f" {np_result['time']:>10.3f}s", end="|")

            # swarmtorch CPU
            st_cpu = self.benchmark_swarmtorch(size, n_iterations, device="cpu")
            results["swarmtorch_cpu"].append(st_cpu)
            print(f" {st_cpu['time']:>10.3f}s", end="|")

            # swarmtorch GPU
            if cuda_available:
                st_gpu = self.benchmark_swarmtorch(size, n_iterations, device="cuda")
                results["swarmtorch_gpu"].append(st_gpu)
                print(f" {st_gpu['time']:>10.3f}s")
            else:
                print(" N/A")

        print("-" * 60)

        # Speedup analysis
        print("\n" + "=" * 60)
        print("SPEEDUP ANALYSIS")
        print("=" * 60)

        print(f"\n{'Swarm Size':>12} | {'GPU vs NumPy':>15} | {'GPU vs CPU':>12}")
        print("-" * 45)

        for i, size in enumerate(swarm_sizes):
            np_time = results["numpy"][i]["time"]
            cpu_time = results["swarmtorch_cpu"][i]["time"]

            if cuda_available:
                gpu_time = results["swarmtorch_gpu"][i]["time"]
                speedup_np = np_time / gpu_time
                speedup_cpu = cpu_time / gpu_time
                print(f"{size:>12} | {speedup_np:>12.1f}x | {speedup_cpu:>10.1f}x")
            else:
                speedup_np = np_time / cpu_time
                print(f"{size:>12} | {speedup_np:>12.1f}x | (GPU N/A)")

        # Plot results
        self.plot_scaling(results, swarm_sizes, cuda_available)

        return results

    def plot_scaling(self, results, swarm_sizes, cuda_available):
        """Plot scaling comparison."""

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Time vs Swarm Size
        ax1 = axes[0]
        ax1.plot(
            swarm_sizes,
            [r["time"] for r in results["numpy"]],
            "o-",
            label="NumPy",
            linewidth=2,
        )
        ax1.plot(
            swarm_sizes,
            [r["time"] for r in results["swarmtorch_cpu"]],
            "s-",
            label="SwarmTorch CPU",
            linewidth=2,
        )
        if cuda_available:
            ax1.plot(
                swarm_sizes,
                [r["time"] for r in results["swarmtorch_gpu"]],
                "^-",
                label="SwarmTorch GPU",
                linewidth=2,
            )

        ax1.set_xlabel("Swarm Size")
        ax1.set_ylabel("Time (seconds)")
        ax1.set_title("Scaling: Time vs Swarm Size")
        ax1.legend()
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)

        # Throughput
        ax2 = axes[1]
        ax2.plot(
            swarm_sizes,
            [r["evaluations_per_second"] for r in results["numpy"]],
            "o-",
            label="NumPy",
            linewidth=2,
        )
        ax2.plot(
            swarm_sizes,
            [r["evaluations_per_second"] for r in results["swarmtorch_cpu"]],
            "s-",
            label="SwarmTorch CPU",
            linewidth=2,
        )
        if cuda_available:
            ax2.plot(
                swarm_sizes,
                [r["evaluations_per_second"] for r in results["swarmtorch_gpu"]],
                "^-",
                label="SwarmTorch GPU",
                linewidth=2,
            )

        ax2.set_xlabel("Swarm Size")
        ax2.set_ylabel("Evaluations/Second")
        ax2.set_title("Throughput: Evaluations per Second")
        ax2.legend()
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("benchmarks/gpu_scaling.png", dpi=150)
        print("\nPlot saved to benchmarks/gpu_scaling.png")


if __name__ == "__main__":
    benchmark = GPUSpeedBenchmark(dim=100)
    results = benchmark.run_scaling_benchmark()
