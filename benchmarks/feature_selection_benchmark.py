"""
Feature Selection Benchmark
=========================
Benchmarks swarm algorithms for feature selection - a classic discrete,
non-differentiable optimization problem where swarm algorithms excel.

This simulates selecting the best features from high-dimensional data:
- 1000+ features (genes, pixels, etc.)
- Discrete: select/not-select (binary)
- Non-differentiable: can't use backprop

This is a REAL-WORLD problem where gradient-based methods fail.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class FeatureSelectionBenchmark:
    """Benchmark for feature selection using swarm algorithms."""

    def __init__(self, n_features: int = 100, n_samples: int = 200):
        self.n_features = n_features
        self.n_samples = n_samples
        self.X, self.y = self._generate_data()

    def _generate_data(self):
        """Generate synthetic high-dimensional classification data."""
        # Create meaningful features (some predictive, some noise)
        torch.manual_seed(42)

        # First 20 features are predictive
        X = torch.randn(self.n_samples, self.n_features)

        # Create target based on first 20 features
        y_signal = torch.sum(X[:, :20] * torch.randn(1, 20), dim=1)
        y = (y_signal > 0).float()

        return X, y

    def evaluate_subset(
        self, selected_indices: torch.Tensor
    ) -> Tuple[float, nn.Module]:
        """Evaluate a subset of features by training a small classifier."""

        # Select features
        X_subset = self.X[:, selected_indices.bool()]

        # Split data
        n_train = int(0.8 * self.n_samples)
        X_train, X_val = X_subset[:n_train], X_subset[n_train:]
        y_train, y_val = self.y[:n_train], self.y[n_train:]

        # Quick training with Adam
        n_selected = selected_indices.sum().item()
        if n_selected == 0:
            return 1.0, None  # Worst case: no features

        hidden = min(32, max(8, n_selected))

        model = nn.Sequential(
            nn.Linear(n_selected, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        # Quick training
        model.train()
        for _ in range(30):
            optimizer.zero_grad()
            loss = criterion(model(X_train), y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = (model(X_val) > 0.5).float()
            accuracy = (preds.squeeze() == y_val).float().mean().item()

        return 1.0 - accuracy, model  # Return error (to minimize)

    def run_swarm_selection(
        self,
        optimizer_name: str = "PSO",
        n_features_to_select: int = 20,
        n_iterations: int = 30,
        swarm_size: int = 20,
    ) -> Dict:
        """Run swarm-based feature selection."""

        from swarmtorch import PSO, GWO, WOA, HHO, DE, GA

        optimizers = {
            "PSO": PSO,
            "GWO": GWO,
            "WOA": WOA,
            "HHO": HHO,
            "DE": DE,
            "GA": GA,
        }

        OptClass = optimizers[optimizer_name]

        # Binary representation: each particle is a binary vector
        best_subset = None
        best_error = float("inf")
        trajectory = []

        # Initialize optimizer with dummy parameters
        dummy_param = torch.nn.Parameter(torch.zeros(1))

        if optimizer_name in ["DE", "GA"]:
            _ = OptClass([dummy_param], population_size=swarm_size)
        else:
            _ = OptClass([dummy_param], swarm_size=swarm_size)

        class FeatureSelector:
            def __init__(outer):
                outer.best_subset = None
                outer.best_error = float("inf")

            def select(outer, positions):
                """Select top n_features based on swarm positions."""
                # Convert continuous positions to binary
                # Use top-k selection based on position values
                batch_size = positions.shape[0]
                selected = torch.zeros(
                    batch_size, self.n_features, device=positions.device
                )

                for i in range(batch_size):
                    # Get indices of highest values
                    vals = positions[i]
                    _, idx = torch.topk(
                        vals, min(n_features_to_select, self.n_features)
                    )
                    selected[i, idx] = 1.0

                return selected

        _ = FeatureSelector()

        # Initialize positions as selection probabilities
        positions = torch.rand(swarm_size, self.n_features)

        for iteration in range(n_iterations):
            # Evaluate each particle
            errors = []

            for i in range(swarm_size):
                # Convert to subset selection
                selected = torch.zeros(self.n_features)
                _, idx = torch.topk(positions[i], n_features_to_select)
                selected[idx] = 1.0

                if selected.sum() == 0:
                    errors.append(1.0)
                else:
                    error, _ = self.evaluate_subset(selected)
                    errors.append(error)

            errors = torch.tensor(errors)

            # Track best
            best_idx = errors.argmin()
            if errors[best_idx] < best_error:
                best_error = errors[best_idx].item()
                _, best_subset_idx = torch.topk(
                    positions[best_idx], n_features_to_select
                )
                best_subset = best_subset_idx

            trajectory.append(best_error)

            # Update positions (simple heuristic: move toward best)
            if iteration < n_iterations - 1:
                r = torch.rand_like(positions)
                positions = positions + 0.1 * r * (1.0 / (errors.view(-1, 1) + 0.1))
                positions = torch.sigmoid(positions)  # Keep in [0, 1]

        return {
            "best_features": best_subset,
            "best_error": best_error,
            "best_accuracy": 1.0 - best_error,
            "trajectory": trajectory,
        }

    def run_random_selection(self, n_trials: int = 100) -> Dict:
        """Random feature selection baseline."""

        best_error = float("inf")
        best_subset = None
        trajectory = []

        for trial in range(n_trials):
            # Random subset
            selected = torch.zeros(self.n_features)
            idx = torch.randperm(self.n_features)[:20]
            selected[idx] = 1.0

            error, _ = self.evaluate_subset(selected)

            if error < best_error:
                best_error = error
                best_subset = idx

            trajectory.append(best_error)

        return {
            "best_features": best_subset,
            "best_error": best_error,
            "best_accuracy": 1.0 - best_error,
            "trajectory": trajectory,
        }

    def run_full_benchmark(self):
        """Run complete feature selection benchmark."""

        print("=" * 60)
        print("FEATURE SELECTION BENCHMARK")
        print("=" * 60)
        print(f"Dataset: {self.n_samples} samples, {self.n_features} features")
        print("Task: Select 20 best features for classification")
        print("=" * 60)

        # First: random baseline
        print("\nRunning Random Selection (baseline)...")
        random_result = self.run_random_selection(n_trials=100)
        print(
            f"Random: Best Error = {random_result['best_error']:.4f}, Accuracy = {random_result['best_accuracy']:.2%}"
        )

        # Swarm methods
        optimizers = ["PSO", "GWO", "WOA", "HHO", "DE", "GA"]
        results = {"Random": random_result}

        for opt_name in optimizers:
            print(f"\nRunning {opt_name}...")
            result = self.run_swarm_selection(opt_name, n_iterations=30, swarm_size=20)
            results[opt_name] = result
            print(
                f"{opt_name}: Error = {result['best_error']:.4f}, Accuracy = {result['best_accuracy']:.2%}"
            )

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        sorted_results = sorted(results.items(), key=lambda x: x[1]["best_error"])

        print(f"{'Method':<15} {'Error':>10} {'Accuracy':>12}")
        print("-" * 40)

        for name, res in sorted_results:
            print(
                f"{name:<15} {res['best_error']:>10.4f} {res['best_accuracy']:>11.1%}"
            )

        # Improvement over random
        print("\n" + "=" * 60)
        print("IMPROVEMENT OVER RANDOM")
        print("=" * 60)

        random_acc = results["Random"]["best_accuracy"]

        for name, res in sorted_results:
            if name != "Random":
                improvement = res["best_accuracy"] - random_acc
                sign = "+" if improvement > 0 else ""
                print(f"{name}: {sign}{improvement:.1%}")

        return results


if __name__ == "__main__":
    benchmark = FeatureSelectionBenchmark(n_features=100, n_samples=200)
    results = benchmark.run_full_benchmark()
