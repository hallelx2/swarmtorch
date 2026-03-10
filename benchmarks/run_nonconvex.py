"""Reduced non-convex benchmark runner for paper results."""
import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swarmtorch import PSO, GWO, WOA, HHO, DE, GA

DIM = 10
DEVICE = "cpu"
MAX_ITER = 200
SWARM_SIZE = 20
N_RUNS = 2


def rastrigin(x):
    A = 10
    return A * DIM + torch.sum(x**2 - A * torch.cos(2 * torch.pi * x), dim=-1)


def ackley(x):
    a, b, c = 20, 0.2, 2 * torch.pi
    sum1 = torch.sum(x**2, dim=-1)
    sum2 = torch.sum(torch.cos(c * x), dim=-1)
    return (
        -a * torch.exp(-b * torch.sqrt(sum1 / DIM))
        - torch.exp(sum2 / DIM)
        + a
        + torch.exp(torch.tensor(1.0))
    )


def rosenbrock(x):
    x = x.unsqueeze(0) if x.dim() == 1 else x
    return torch.sum(
        100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2, dim=-1
    )


def schwefel(x):
    return torch.sum(-x * torch.sin(torch.sqrt(torch.abs(x))), dim=-1)


def griewank(x):
    x = x.unsqueeze(0) if x.dim() == 1 else x
    sum1 = torch.sum(x**2, dim=-1) / 4000
    prod = torch.prod(
        torch.cos(
            x / torch.sqrt(torch.arange(1, x.shape[-1] + 1, device=DEVICE).float())
        ),
        dim=-1,
    )
    return sum1 - prod + 1


def sphere(x):
    return torch.sum(x**2, dim=-1)


FUNCTIONS = {
    "Rastrigin": (rastrigin, (-5.12, 5.12)),
    "Ackley": (ackley, (-32.768, 32.768)),
    "Rosenbrock": (rosenbrock, (-10, 10)),
    "Schwefel": (schwefel, (-500, 500)),
    "Griewank": (griewank, (-600, 600)),
    "Sphere": (sphere, (-10, 10)),
}

OPTIMIZERS = {"PSO": PSO, "GWO": GWO, "WOA": WOA, "HHO": HHO, "DE": DE, "GA": GA}


def run_adam(func, bounds):
    losses = []
    for _ in range(N_RUNS):
        x = torch.nn.Parameter(torch.randn(DIM))
        opt = torch.optim.Adam([x], lr=0.01)
        best = float("inf")
        for _ in range(MAX_ITER):
            opt.zero_grad()
            loss = func(x.unsqueeze(0))
            val = loss.item()
            if val < best:
                best = val
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(bounds[0], bounds[1])
        losses.append(best)
    return float(np.mean(losses)), float(np.std(losses))


def run_meta(func, bounds, OptClass, opt_name):
    losses = []
    for _ in range(N_RUNS):
        x_init = torch.rand(DIM) * (bounds[1] - bounds[0]) + bounds[0]
        x = torch.nn.Parameter(x_init.clone())
        if opt_name in ["DE", "GA"]:
            opt = OptClass([x], population_size=SWARM_SIZE, device=DEVICE)
        else:
            opt = OptClass([x], swarm_size=SWARM_SIZE, device=DEVICE)
        best = float("inf")
        for _ in range(MAX_ITER):
            def closure():
                return func(x.unsqueeze(0))

            loss = opt.step(closure)
            if loss is not None:
                val = loss.item()
                if val < best:
                    best = val
            else:
                val = func(x.unsqueeze(0)).item()
                if val < best:
                    best = val
        losses.append(best)
    return float(np.mean(losses)), float(np.std(losses))


if __name__ == "__main__":
    all_results = {}
    for func_name, (func, bounds) in FUNCTIONS.items():
        print(f"=== {func_name} ===", flush=True)
        func_results = {}

        mean, std = run_adam(func, bounds)
        func_results["Adam"] = {"final_loss": mean, "std": std}
        print(f"  Adam: {mean:.4f} +/- {std:.4f}", flush=True)

        for opt_name, OptClass in OPTIMIZERS.items():
            mean, std = run_meta(func, bounds, OptClass, opt_name)
            func_results[opt_name] = {"final_loss": mean, "std": std}
            print(f"  {opt_name}: {mean:.4f} +/- {std:.4f}", flush=True)

        all_results[func_name] = func_results

    outpath = os.path.join(os.path.dirname(__file__), "nonconvex_results.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone! Saved to {outpath}")
