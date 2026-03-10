"""
Non-Convex Optimization Benchmark
================================
Compares SwarmTorch metaheuristics against gradient-based methods on
classic optimization test functions where gradients fail.

These functions are specifically designed to trap gradient-based optimizers:
- Rastrigin: Many local minima (traps gradient descent)
- Ackley: Flat regions with sharp spikes
- Rosenbrock: Narrow, curved valley (hard for any method)
- Schwefel: Massive local minima landscape
- Griewank: Multi-scale oscillations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Tuple
import json


class NonConvexBenchmark:
    """Benchmark for non-convex optimization functions."""

    def __init__(self, dim: int = 10, device: str = "cpu"):
        self.dim = dim
        self.device = device
        self.results = {}

    # ============== Test Functions ==============

    def rastrigin(self, x: torch.Tensor) -> torch.Tensor:
        """Rastrigin function - highly multimodal with many local minima.

        Global minimum at x=0 with f(x)=0
        Has ~dim local minima.
        """
        A = 10
        return A * self.dim + torch.sum(x**2 - A * torch.cos(2 * torch.pi * x), dim=-1)

    def ackley(self, x: torch.Tensor) -> torch.Tensor:
        """Ackley function - flat region with central hole and many local minima.

        Global minimum at x=0 with f(x)=0
        """
        a, b, c = 20, 0.2, 2 * torch.pi
        sum1 = torch.sum(x**2, dim=-1)
        sum2 = torch.sum(torch.cos(c * x), dim=-1)
        return (
            -a * torch.exp(-b * torch.sqrt(sum1 / self.dim))
            - torch.exp(sum2 / self.dim)
            + a
            + torch.exp(torch.tensor(1.0, device=self.device))
        )

    def rosenbrock(self, x: torch.Tensor) -> torch.Tensor:
        """Rosenbrock function - narrow valley (banana function).

        Global minimum at x=1 with f(x)=0
        Known for being difficult to optimize.
        """
        x = x.unsqueeze(0) if x.dim() == 1 else x
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return torch.sum(
            100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2, dim=-1
        )

    def schwefel(self, x: torch.Tensor) -> torch.Tensor:
        """Schwefel function - massive number of local minima.

        Global minimum at x=420.9687 with f(x)=0
        """
        return torch.sum(-x * torch.sin(torch.sqrt(torch.abs(x))), dim=-1)

    def griewank(self, x: torch.Tensor) -> torch.Tensor:
        """Griewank function - multi-scale oscillatory.

        Global minimum at x=0 with f(x)=0
        Has oscillations at different scales.
        """
        x = x.unsqueeze(0) if x.dim() == 1 else x
        if x.dim() == 1:
            x = x.unsqueeze(0)
        sum1 = torch.sum(x**2, dim=-1) / 4000
        prod = torch.prod(
            torch.cos(
                x
                / torch.sqrt(
                    torch.arange(1, x.shape[-1] + 1, device=self.device).float()
                )
            ),
            dim=-1,
        )
        return sum1 - prod + 1

    def sphere(self, x: torch.Tensor) -> torch.Tensor:
        """Simple sphere function - convex baseline."""
        return torch.sum(x**2, dim=-1)

    # ============== Benchmark Methods ==============

    def benchmark_function(
        self,
        func: Callable,
        optimizer_name: str,
        bounds: Tuple[float, float] = (-5.12, 5.12),
        max_iter: int = 500,
        swarm_size: int = 30,
        n_runs: int = 3,
    ) -> Dict:
        """Benchmark a single function with an optimizer."""

        from swarmtorch import PSO, GWO, WOA, HHO, DE, GA

        optimizers = {
            "PSO": PSO,
            "GWO": GWO,
            "WOA": WOA,
            "HHO": HHO,
            "DE": DE,
            "GA": GA,
        }

        if optimizer_name == "Adam":
            # Adam benchmark - just use simple gradient descent on the function
            # (not truly applicable but included for comparison)
            results = []
            for _ in range(n_runs):
                x = torch.randn(
                    max_iter + 1, self.dim, device=self.device, requires_grad=True
                )
                opt = torch.optim.Adam([x], lr=0.01)

                best_loss = float("inf")
                losses = []

                for i in range(max_iter):
                    loss = func(x.unsqueeze(0))
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                    losses.append(best_loss)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    # Project back to bounds
                    with torch.no_grad():
                        x.clamp_(bounds[0], bounds[1])

                results.append({"final": best_loss, "trajectory": losses})

            return {
                "final_loss": np.mean([r["final"] for r in results]),
                "std": np.std([r["final"] for r in results]),
                "trajectory": results[0]["trajectory"],
            }

        else:
            OptClass = optimizers.get(optimizer_name)
            if OptClass is None:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")

            results = []
            for run in range(n_runs):
                # Initialize at random position
                x_init = (
                    torch.rand(self.dim, device=self.device) * (bounds[1] - bounds[0])
                    + bounds[0]
                )

                # Use a simple parameter tensor for the optimizer
                x = torch.nn.Parameter(x_init.clone())

                if optimizer_name in ["DE", "GA"]:
                    opt = OptClass([x], population_size=swarm_size, device=self.device)
                else:
                    opt = OptClass([x], swarm_size=swarm_size, device=self.device)

                best_loss = float("inf")
                losses = []

                for i in range(max_iter):

                    def closure():
                        loss = func(x.unsqueeze(0))
                        return loss

                    loss = opt.step(closure)
                    if loss is not None and loss.item() < best_loss:
                        best_loss = loss.item()
                    elif loss is None:
                        current = func(x.unsqueeze(0)).item()
                        if current < best_loss:
                            best_loss = current
                    losses.append(best_loss)

                results.append({"final": best_loss, "trajectory": losses})

            return {
                "final_loss": np.mean([r["final"] for r in results]),
                "std": np.std([r["final"] for r in results]),
                "trajectory": results[0]["trajectory"],
            }

    def run_full_benchmark(self):
        """Run complete benchmark across all functions and optimizers."""

        functions = {
            "Rastrigin": (self.rastrigin, (-5.12, 5.12)),
            "Ackley": (self.ackley, (-32.768, 32.768)),
            "Rosenbrock": (self.rosenbrock, (-10, 10)),
            "Schwefel": (self.schwefel, (-500, 500)),
            "Griewank": (self.griewank, (-600, 600)),
            "Sphere": (self.sphere, (-10, 10)),  # Baseline - convex
        }

        optimizers = ["PSO", "GWO", "WOA", "HHO", "DE", "GA"]

        print("=" * 60)
        print("NON-CONVEX OPTIMIZATION BENCHMARK")
        print("=" * 60)
        print(f"Dimension: {self.dim}")
        print("Swarm Size: 30")
        print("Iterations: 500")
        print("Runs: 3")
        print("=" * 60)

        all_results = {}

        for func_name, (func, bounds) in functions.items():
            print(f"\n{'=' * 40}")
            print(f"Function: {func_name}")
            print(f"{'=' * 40}")

            func_results = {}

            for opt_name in optimizers:
                print(f"  Testing {opt_name}...", end=" ")

                result = self.benchmark_function(
                    func, opt_name, bounds, max_iter=500, swarm_size=30, n_runs=3
                )

                func_results[opt_name] = {
                    "final_loss": result["final_loss"],
                    "std": result["std"],
                }

                print(f"Loss: {result['final_loss']:.4f} ± {result['std']:.4f}")

            all_results[func_name] = func_results

        self.results = all_results

        # Find best optimizer per function
        print("\n" + "=" * 60)
        print("SUMMARY: Best Optimizer per Function")
        print("=" * 60)

        for func_name, func_results in all_results.items():
            best_opt = min(func_results, key=lambda x: func_results[x]["final_loss"])
            best_loss = func_results[best_opt]["final_loss"]
            print(f"{func_name:15} -> {best_opt:5} (loss: {best_loss:.4f})")

        return all_results

    def plot_results(self, save_path: str = "benchmarks/nonconvex_results.png"):
        """Plot convergence curves."""

        if not self.results:
            print("No results to plot. Run benchmark first.")
            return

        # Re-run with trajectory tracking for plotting
        functions = {
            "Rastrigin": self.rastrigin,
            "Ackley": self.ackley,
            "Schwefel": self.schwefel,
        }

        optimizers = ["PSO", "GWO", "WOA", "HHO"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (func_name, func) in enumerate(functions.items()):
            ax = axes[idx]

            for opt_name in optimizers:
                result = self.benchmark_function(
                    func, opt_name, max_iter=200, swarm_size=30, n_runs=1
                )
                ax.plot(result["trajectory"], label=opt_name, alpha=0.8)

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.set_title(f"{func_name} Function")
            ax.legend()
            ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")

    def save_results(self, path: str = "benchmarks/nonconvex_results.json"):
        """Save results to JSON."""

        if self.results:
            with open(path, "w") as f:
                json.dump(self.results, f, indent=2)
            print(f"Results saved to {path}")


if __name__ == "__main__":
    benchmark = NonConvexBenchmark(dim=10, device="cpu")
    results = benchmark.run_full_benchmark()
    benchmark.plot_results()
    benchmark.save_results()
