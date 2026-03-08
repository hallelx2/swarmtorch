import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Import Tuners from all categories
from swarmtorch.swarm.hyperparameter_tuning import (
    PSOSearch,
    GWOSearch,
    WOASearch,
    HHOSearch,
    RandomSearchHT,
)
from swarmtorch.evolutionary.hyperparameter_tuning import DESearch, GASearch

# Configuration for Research Paper standard
N_RUNS = 2  # Number of independent runs for statistical significance
BUDGET_ITER = 3  # Number of iterations for each search
SWARM_SIZE = 5  # Number of particles/individuals per search
DEVICE = "cpu"  # Force CPU for portability in this benchmark


# 1. Simulated Dataset Generation (Pure Torch)
def generate_synthetic_data(n_samples=500):
    X = torch.randn(n_samples, 20)
    # Target depends on some "hidden" logic that hyperparameters need to find
    y = (X[:, 0] + X[:, 5] * 2 - X[:, 10] > 0).float().unsqueeze(1)
    return X, y


X_train, y_train = generate_synthetic_data(500)
X_test, y_test = generate_synthetic_data(200)


# 2. Model Definition
class SimpleMLP(nn.Module):
    def __init__(self, hidden_dim=32, dropout_rate=0.2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(20, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


# 3. Hyperparameter Space
param_space = {
    "lr": (0.0001, 0.1),
    "hidden_dim": [16, 32, 64, 128],
    "dropout_rate": (0.0, 0.5),
}


def build_model(params):
    return SimpleMLP(
        hidden_dim=int(params["hidden_dim"]), dropout_rate=params["dropout_rate"]
    )


def train_fn(model, params):
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.BCELoss()

    # Train for a few steps
    for _ in range(10):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        pred = (output > 0.5).float()
        accuracy = (pred == y_test).float().mean().item()

    return 1.0 - accuracy


# 4. Benchmarking Logic
tuners = {
    "Random": RandomSearchHT,
    "PSO": PSOSearch,
    "GWO": GWOSearch,
    "WOA": WOASearch,
    "HHO": HHOSearch,
    "GA": GASearch,
    "DE": DESearch,
}

results_summary = []

print(f"Starting Portability-Friendly Benchmark on {DEVICE}...")

for name, tuner_class in tuners.items():
    run_scores = []
    print(f"Evaluating {name}...", end=" ", flush=True)

    for i in range(N_RUNS):
        tuner = tuner_class(
            model_fn=build_model,
            param_space=param_space,
            train_fn=train_fn,
            iterations=BUDGET_ITER,
            swarm_size=SWARM_SIZE,
            device=DEVICE,
            verbose=False,
        )

        tuner.search()
        run_scores.append(1.0 - tuner.best_score)

    mean_acc = np.mean(run_scores)
    std_acc = np.std(run_scores)
    results_summary.append((name, mean_acc, std_acc))
    print(f"Done. Mean Acc: {mean_acc:.4f}")

# 5. Final Report
results_summary.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 50)
print(f"{'Algorithm':<15} | {'Mean Acc':<10} | {'Std Dev':<10}")
print("-" * 50)
for name, m, s in results_summary:
    print(f"{name:<15} | {m:<10.4f} | {s:<10.4f}")
print("=" * 50)
