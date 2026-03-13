import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from algo_registry import TRAIN_MAP, TUNE_MAP

# --- Configuration for Research Paper ---
DEVICE = "cpu"
BUDGET_TRAIN = 1000  # Forward evaluations for each weight optimizer
BUDGET_HPO = 2  # Iterations for each hyperparameter tuner
SWARM_SIZE = 10  # Population size for all metaheuristics


# --- 1. Dataset Generation ---
def generate_xor_data(n_samples=500):
    X = torch.randn(n_samples, 2)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).float().unsqueeze(1)
    return X, y


def generate_hpo_data(n_samples=500, n_features=10):
    X = torch.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] * 2 - X[:, 2] > 0).float().unsqueeze(1)
    return X, y


# --- 2. Model Architectures ---
class TrainingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class HPOModel(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(10, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


# --- 3. Mass Benchmarking Logic: Model Training ---
def run_full_training_benchmark():
    X, y = generate_xor_data(500)
    criterion = nn.BCEWithLogitsLoss()

    results = []

    # Baselines
    for baseline in ["Adam", "SGD"]:
        print(f"Running Baseline: {baseline}...")
        model = TrainingMLP().to(DEVICE)
        opt = (
            optim.Adam(model.parameters(), lr=0.01)
            if baseline == "Adam"
            else optim.SGD(model.parameters(), lr=0.1)
        )
        eval_count = 0
        final_loss = 0.0
        while eval_count < BUDGET_TRAIN:
            opt.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            eval_count += 1
            final_loss = loss.item()
        results.append(
            {"Algorithm": baseline, "Category": "Gradient", "Final Loss": final_loss}
        )

    # Metaheuristics
    for category, algos in TRAIN_MAP.items():
        module = importlib.import_module(f"swarmtorch.{category}.model_training")
        for algo_name in algos:
            print(f"Testing Weight Optimizer: {algo_name} ({category})...")
            try:
                opt_class = getattr(module, algo_name)
                model = TrainingMLP().to(DEVICE)

                try:
                    optimizer = opt_class(model.parameters(), swarm_size=SWARM_SIZE)
                except Exception:
                    optimizer = opt_class(
                        model.parameters(), population_size=SWARM_SIZE
                    )

                eval_count = 0
                last_loss = 0.0
                while eval_count < BUDGET_TRAIN:

                    def closure():
                        nonlocal eval_count
                        eval_count += 1
                        optimizer.zero_grad()
                        loss = criterion(model(X), y)
                        return loss

                    try:
                        loss_tensor = optimizer.step(closure)
                        last_loss = loss_tensor.item()
                    except Exception:
                        break
                results.append(
                    {
                        "Algorithm": algo_name,
                        "Category": category,
                        "Final Loss": last_loss,
                    }
                )
            except Exception as e:
                print(f"  FAILED {algo_name}: {e}")

    return pd.DataFrame(results)


# --- 4. Mass Benchmarking Logic: HPO ---
def run_full_hpo_benchmark():
    X_train, y_train = generate_hpo_data(500)
    X_test, y_test = generate_hpo_data(200)

    param_space = {
        "lr": (0.001, 0.1),
        "hidden_dim": [16, 32, 64],
        "dropout": (0.0, 0.4),
    }

    def build_fn(params):
        return HPOModel(int(params["hidden_dim"]), params["dropout"])

    def train_fn(model, params):
        opt = optim.Adam(model.parameters(), lr=params["lr"])
        crit = nn.BCEWithLogitsLoss()
        for _ in range(5):  # Short training
            opt.zero_grad()
            loss = crit(model(X_train), y_train)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = ((model(X_test) > 0.0).float() == y_test).float().mean().item()
        return 1.0 - acc

    results = []
    for category, algos in TUNE_MAP.items():
        module = importlib.import_module(f"swarmtorch.{category}.hyperparameter_tuning")
        for algo_name in algos:
            print(f"Testing HPO Searcher: {algo_name} ({category})...")
            try:
                tuner_class = getattr(module, algo_name)
                tuner = tuner_class(
                    build_fn,
                    param_space,
                    train_fn,
                    iterations=BUDGET_HPO,
                    swarm_size=SWARM_SIZE,
                    verbose=False,
                )
                tuner.search()
                results.append(
                    {
                        "Algorithm": algo_name,
                        "Category": category,
                        "Best Accuracy": 1.0 - tuner.best_score,
                    }
                )
            except Exception as e:
                print(f"  FAILED {algo_name}: {e}")

    return pd.DataFrame(results)


# --- 5. Report Generation ---
if __name__ == "__main__":
    print("STARTING MASS BENCHMARK...")
    df_train = run_full_training_benchmark()
    df_hpo = run_full_hpo_benchmark()

    # Visualization: Model Training Top 20
    plt.figure(figsize=(12, 8))
    top_train = df_train.sort_values(by="Final Loss").head(25)
    plt.barh(top_train["Algorithm"], top_train["Final Loss"], color="skyblue")
    plt.gca().invert_yaxis()
    plt.title("Top 25 Model Weight Optimizers (Lowest Final Loss)")
    plt.xlabel("Loss (Binary Cross Entropy)")
    plt.tight_layout()
    plt.savefig("top_optimizers_training.png")

    # Visualization: HPO Comparison
    plt.figure(figsize=(12, 8))
    top_hpo = df_hpo.sort_values(by="Best Accuracy", ascending=False).head(25)
    plt.barh(top_hpo["Algorithm"], top_hpo["Best Accuracy"], color="lightgreen")
    plt.gca().invert_yaxis()
    plt.title("Top 25 Hyperparameter Searchers (Highest Best Accuracy)")
    plt.xlabel("Accuracy")
    plt.tight_layout()
    plt.savefig("top_hpo_searchers.png")

    # Export Report
    report = "# Comprehensive Research Benchmark Report\n"
    report += f"Generated on: {time.ctime()}\n\n"
    report += f"Total Model Training Optimizers Evaluated: {len(df_train)}\n"
    report += f"Total HPO Searchers Evaluated: {len(df_hpo)}\n\n"

    report += "## Best Performing Algorithms by Category\n"
    for cat in df_train["Category"].unique():
        if cat == "Gradient":
            continue
        best = (
            df_train[df_train["Category"] == cat].sort_values(by="Final Loss").iloc[0]
        )
        report += f"- **{cat}**: Best Trainer is {best['Algorithm']} (Loss: {best['Final Loss']:.4f})\n"

    report += "\n## Detailed Results (Training Weight Optimization)\n"
    report += "| " + " | ".join(df_train.columns) + " |\n"
    report += "| " + " | ".join(["---"] * len(df_train.columns)) + " |\n"
    for _, row in df_train.sort_values(by="Final Loss").iterrows():
        report += "| " + " | ".join([str(v) for v in row.values]) + " |\n"

    report += "\n\n## Detailed Results (Hyperparameter Optimization)\n"
    report += "| " + " | ".join(df_hpo.columns) + " |\n"
    report += "| " + " | ".join(["---"] * len(df_hpo.columns)) + " |\n"
    for _, row in df_hpo.sort_values(by="Best Accuracy", ascending=False).iterrows():
        report += "| " + " | ".join([str(v) for v in row.values]) + " |\n"

    with open("COMPREHENSIVE_EXPERIMENT_REPORT.md", "w") as f:
        f.write(report)

    print(
        "\nBenchmark Complete! View COMPREHENSIVE_EXPERIMENT_REPORT.md and PNG visualizations."
    )
