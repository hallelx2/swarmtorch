# XOR Classification

Complete example of gradient-free neural network training on XOR problem.

## Problem

XOR is a classic non-linearly separable problem:
- (0,0) → 0
- (0,1) → 1
- (1,0) → 1
- (1,1) → 0

## Solution

```python
import torch
import torch.nn as nn
from swarmtorch import PSO

# Define the model
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

# Prepare data
X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# Create model and optimizer
model = XORModel()
optimizer = PSO(model.parameters(), swarm_size=50)

criterion = nn.BCEWithLogitsLoss()

# Training loop
print("Training XOR classifier with PSO...")
for epoch in range(300):
    def closure():
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        return loss
    
    loss = optimizer.step(closure)
    
    if epoch % 50 == 0:
        with torch.no_grad():
            preds = (torch.sigmoid(model(X)) > 0.5).float()
            acc = (preds == y).float().mean()
            print(f"Epoch {epoch:3d}: Loss={loss.item():.4f}, Accuracy={acc:.0%}")

# Final results
with torch.no_grad():
    predictions = (torch.sigmoid(model(X)) > 0.5).float()
    print("\nFinal Results:")
    print(f"Input: {X.tolist()}")
    print(f"Expected: {y.T.tolist()[0]}")
    print(f"Predicted: {predictions.T.tolist()[0]}")
```

## Expected Output

```
Training XOR classifier with PSO...
Epoch   0: Loss=0.7234, Accuracy=50%
Epoch  50: Loss=0.5621, Accuracy=50%
Epoch 100: Loss=0.3124, Accuracy=75%
Epoch 150: Loss=0.1892, Accuracy=100%
Epoch 200: Loss=0.1245, Accuracy=100%
Epoch 250: Loss=0.0892, Accuracy=100%

Final Results:
Input: [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Expected: [0.0, 1.0, 1.0, 0.0]
Predicted: [0.0, 1.0, 1.0, 0.0]
```

## Key Points

1. **BCEWithLogitsLoss**: Use this instead of BCEWithSigmoid
2. **Closure Pattern**: Required for swarm optimizers
3. **Swarm Size**: 50 particles for this problem
4. **Iterations**: 300 steps for convergence

## Try Different Optimizers

```python
# Swap PSO with other optimizers
from swarmtorch import GWO, WOA, HHO, DE

# Grey Wolf Optimizer
optimizer = GWO(model.parameters(), swarm_size=50)

# Whale Optimization Algorithm
optimizer = WOA(model.parameters(), swarm_size=50)

# Harris Hawks Optimization
optimizer = HHO(model.parameters(), swarm_size=50)

# Differential Evolution
optimizer = DE(model.parameters(), population_size=50)
```
