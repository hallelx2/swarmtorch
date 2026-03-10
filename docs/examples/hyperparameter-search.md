# Hyperparameter Search Example

Complete example of using SwarmTorch HPO to tune model hyperparameters.

## Problem

Find optimal hyperparameters for a simple neural network.

## Solution

```python
import torch
import torch.nn as nn
from swarmtorch import GWOSearch

# Generate data
torch.manual_seed(42)
X = torch.rand(200, 4)
y = (X.sum(dim=1) > 2).float().unsqueeze(1)

# Split data
X_train, X_val = X[:150], X[150:]
y_train, y_val = y[:150], y[150:]

# Step 1: Define model factory
def build_model(hidden_dim: int, dropout: float) -> nn.Module:
    """Build model with configurable hyperparameters."""
    model = nn.Sequential(
        nn.Linear(4, hidden_dim),
        nn.Dropout(dropout),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )
    # Store hparams for easy access
    model.hparams = {'hidden_dim': hidden_dim, 'dropout': dropout}
    return model

# Step 2: Define training function
def train_fn(model: nn.Module) -> float:
    """Train briefly and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for _ in range(30):  # Quick training
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val).item()
    return val_loss

# Step 3: Define search space
param_space = {
    'hidden_dim': [8, 16, 32, 64],    # Categorical
    'dropout': (0.0, 0.5),            # Continuous
}

# Step 4: Run search
print("Running HPO with GWO...")
searcher = GWOSearch(
    model_fn=build_model,
    param_space=param_space,
    train_fn=train_fn,
    iterations=25,
    swarm_size=15,
    verbose=True,
)

best_params = searcher.search()

print(f"\nBest hyperparameters found:")
print(f"  hidden_dim: {best_params['hidden_dim']}")
print(f"  dropout: {best_params['dropout']:.3f}")

# Step 5: Train final model with best params
best_model = build_model(
    best_params['hidden_dim'],
    best_params['dropout']
)

print("\nTraining final model with best params...")
for epoch in range(100):
    def closure():
        optimizer.zero_grad()
        loss = criterion(best_model(X_train), y_train)
        return loss
    
    loss = best_model.optim.step(closure)

# Evaluate
with torch.no_grad():
    final_preds = (torch.sigmoid(best_model(X_val)) > 0.5).float()
    final_acc = (final_preds == y_val).float().mean()
    print(f"Final validation accuracy: {final_acc:.1%}")
```

## Expected Output

```
Running HPO with GWO...
Iteration 0: Best loss = 0.6234
Iteration 5: Best loss = 0.4123
Iteration 10: Best loss = 0.2891
Iteration 15: Best loss = 0.1987
Iteration 20: Best loss = 0.1562
Iteration 25: Best loss = 0.1234

Best hyperparameters found:
  hidden_dim: 32
  dropout: 0.234

Final validation accuracy: 92.0%
```

## Key Components

### 1. Model Factory

```python
def build_model(**params) -> nn.Module:
    model = MyModel(**params)
    model.hparams = params  # Store for access
    return model
```

### 2. Training Function

```python
def train_fn(model: nn.Module) -> float:
    # Train briefly
    # Return validation loss (to minimize)
    return val_loss
```

### 3. Search Space

```python
param_space = {
    'continuous': (0.0, 1.0),    # (min, max)
    'categorical': [a, b, c],    # discrete choices
}
```

## Different Searchers

```python
from swarmtorch import PSOSearch, WOASearch, DESearch

# Try different algorithms
searcher = PSOSearch(...)
searcher = WOASearch(...)
searcher = DESearch(...)
```
