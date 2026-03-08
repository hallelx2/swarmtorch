import torch
import torch.nn as nn
import torch.optim as optim
import time

# Import my optimizers
from swarmtorch.evolutionary.model_training import GA, DE
from swarmtorch.human_based.model_training import TLBO
from swarmtorch.swarm.model_training import PSO


# 1. Dataset Generation (Pure Torch)
def generate_data(n_samples=1000):
    # Create two non-linearly separable clusters (XOR-like)
    X = torch.randn(n_samples, 2)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).float().unsqueeze(1)

    # Shuffle
    idx = torch.randperm(n_samples)
    X, y = X[idx], y[idx]

    # Split
    split = int(0.8 * n_samples)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    return X_train, y_train, X_test, y_test


X_train_t, y_train_t, X_test_t, y_test_t = generate_data(1000)


# 2. Model Definition
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


criterion = nn.BCELoss()


def evaluate_model(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y).sum().item() / y.size(0)
    return accuracy


def train_one_optimizer(name, optimizer_class, is_swarm, budget=3000):
    print(f"Running {name}...")
    model = SimpleMLP()
    if is_swarm:
        # Some optimizers use 'swarm_size', others use 'population_size'
        try:
            optimizer = optimizer_class(model.parameters(), swarm_size=30)
        except TypeError:
            optimizer = optimizer_class(model.parameters(), population_size=30)
    else:
        optimizer = optimizer_class(model.parameters(), lr=0.05)

    eval_count = 0
    start_time = time.time()
    last_loss = 0.0

    # We use a while loop to respect the budget exactly
    while eval_count < budget:

        def closure():
            nonlocal eval_count
            eval_count += 1
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            if not is_swarm:
                loss.backward()
            return loss

        # Note: some swarm optimizers call closure multiple times inside step()
        try:
            loss = optimizer.step(closure)
            last_loss = loss.item()
        except Exception as e:
            # If we hit the budget inside step(), some implementations might fail or we just catch it
            # For now, let's assume they handle it or we just stop
            print(f"  {name} stopped early: {e}")
            break

        if eval_count % 1000 == 0 or eval_count >= budget:
            print(
                f"  {name}: {eval_count}/{budget} evals, Current Loss: {last_loss:.4f}"
            )

        if eval_count >= budget:
            break

    duration = time.time() - start_time
    acc = evaluate_model(model, X_test_t, y_test_t)
    return last_loss, acc, duration


# 3. Running Benchmarks
BUDGET = 3000  # Total forward passes allowed
results = {}

# Standard
results["Adam"] = train_one_optimizer("Adam", optim.Adam, is_swarm=False, budget=BUDGET)
results["SGD"] = train_one_optimizer("SGD", optim.SGD, is_swarm=False, budget=BUDGET)

# Swarm
results["GA"] = train_one_optimizer("GA", GA, is_swarm=True, budget=BUDGET)
results["DE"] = train_one_optimizer("DE", DE, is_swarm=True, budget=BUDGET)
results["TLBO"] = train_one_optimizer("TLBO", TLBO, is_swarm=True, budget=BUDGET)
results["PSO"] = train_one_optimizer("PSO", PSO, is_swarm=True, budget=BUDGET)

# 4. Summary
print("\n" + "=" * 60)
print(f"{'Optimizer':<15} | {'Loss':<10} | {'Accuracy':<10} | {'Time (s)':<10}")
print("-" * 60)
for name, (loss, acc, duration) in results.items():
    print(f"{name:<15} | {loss:<10.4f} | {acc:<10.4f} | {duration:<10.2f}")
print("=" * 60)

print("\nNote: BUDGET is total number of forward passes (model evaluations).")
print("Standard optimizers (Adam/SGD) use gradients, Swarm use only loss values.")
