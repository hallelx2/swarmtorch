# Model Training Guide

This guide covers gradient-free neural network training using SwarmTorch optimizers.

## How It Works

SwarmTorch optimizers inherit from `torch.optim.Optimizer` but use a different optimization paradigm:

1. **Population of Solutions**: Instead of single weights, maintain a population of candidate solutions
2. **Loss-Based Evaluation**: Each candidate is evaluated using the loss function (no gradients needed)
3. **Swarm Updates**: Candidates influence each other through social/cognitive mechanisms

## Basic Usage

### The Closure Pattern

Unlike standard PyTorch optimizers, swarm optimizers require a **closure** function:

```python
import torch.nn as nn
from swarmtorch import PSO

model = nn.Linear(10, 2)
optimizer = PSO(model.parameters(), swarm_size=30)

# Required: define a closure that returns the loss
def closure():
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    return loss

# Each step evaluates multiple candidates
loss = optimizer.step(closure)
```

### Loss Function Recommendation

Always use `BCEWithLogitsLoss` for metaheuristic training:

```python
# Good: BCEWithLogitsLoss (expects raw logits)
criterion = nn.BCEWithLogitsLoss()
output = model(x)  # No sigmoid needed
loss = criterion(output, y)

# Avoid: BCEWithSigmoid (requires explicit sigmoid)
criterion = nn.BCEWithSigmoid()
output = torch.sigmoid(model(x))  # Extra sigmoid
loss = criterion(output, y)
```

## Complete Example: Training a MLP

```python
import torch
import torch.nn as nn
from swarmtorch import GWO

class XORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Data
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# Model and optimizer
model = XORModel()
optimizer = GWO(model.parameters(), swarm_size=50)

criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(500):
    def closure():
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        return loss
    
    loss = optimizer.step(closure)
    
    if epoch % 100 == 0:
        with torch.no_grad():
            preds = (torch.sigmoid(model(X)) > 0.5).float()
            acc = (preds == y).float().mean()
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc.item():.2%}")
```

## Key Parameters

### Common Parameters Across Optimizers

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | Required | Model parameters to optimize |
| `swarm_size` | int | 30 | Number of particles/individuals |
| `device` | str | "cpu" | "cpu" or "cuda" |

### Algorithm-Specific Parameters

**PSO**:
- `w`: Inertia weight (default: 0.7)
- `c1`: Cognitive coefficient (default: 1.5)
- `c2`: Social coefficient (default: 1.5)

**DE**:
- `population_size`: Alias for swarm_size
- `cr`: Crossover rate (default: 0.9)
- `f`: Mutation factor (default: 0.8)

**GA**:
- `crossover_rate`: Probability of crossover (default: 0.9)
- `mutation_rate`: Probability of mutation (default: 0.1)

## Comparison: Swarm vs Gradient-Based

| Aspect | Gradient (Adam/SGD) | SwarmTorch |
|--------|---------------------|------------|
| Evaluations per step | 1 | swarm_size |
| Gradient required | Yes | No |
| Memory overhead | Low | Higher (population) |
| Convergence | Fast | Slower but more robust |
| Local minima | Can get stuck | Better exploration |

## Best Practices

### 1. Set Appropriate Swarm Size

```python
# Small model, simple problem
optimizer = PSO(params, swarm_size=20)

# Large model, complex problem
optimizer = PSO(params, swarm_size=50)
```

### 2. Use Sufficient Iterations

```python
# More iterations = better exploration
for _ in range(500):  # 500 optimizer steps
    optimizer.step(closure)
```

### 3. Reproducibility

```python
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
```

### 4. Loss Scaling

Metaheuristics work better with scaled losses. Consider normalizing your loss:

```python
# Scale down large losses
loss = raw_loss / 1000
```

## Troubleshooting

### Loss Not Decreasing

- Increase `swarm_size` for better exploration
- Increase number of iterations
- Try a different algorithm (GWO/WOA for complex landscapes)

### Slow Training

- Reduce `swarm_size` (but may reduce quality)
- Use GPU: `optimizer = PSO(params, device="cuda")`
- Use simpler model architecture

### Convergence to Poor Solution

- Problem may be multi-modal; try GWO or WOA
- Increase swarm size
- Try multiple runs and pick best

## Advanced: Custom Loss Functions

Since metaheuristics only need loss values, you can use any PyTorch loss:

```python
# Custom loss function
def custom_loss(output, target):
    # Anything that produces a scalar tensor
    mse = nn.MSELoss()(output, target)
    reg = 0.01 * sum(p.abs().sum() for p in model.parameters())
    return mse + reg

def closure():
    optimizer.zero_grad()
    output = model(x)
    return custom_loss(output, y)

optimizer.step(closure)
```
