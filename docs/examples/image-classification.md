# Image Classification

Training a CNN on MNIST-like data with gradient-free optimization.

## Problem

Classify 28x28 grayscale images into 10 digits (0-9).

## Solution

```python
import torch
import torch.nn as nn
from swarmtorch import GWO

# Simple CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.conv(x)

# Generate dummy data (replace with real MNIST)
torch.manual_seed(42)
X = torch.rand(100, 1, 28, 28)  # 100 samples
y = torch.randint(0, 10, (100,))  # 10 classes

# Split train/val
X_train, X_val = X[:80], X[80:]
y_train, y_val = y[:80], y[80:]

# Model and optimizer
model = CNN()
optimizer = GWO(model.parameters(), swarm_size=30)

criterion = nn.CrossEntropyLoss()

# Training
print("Training CNN with GWO...")
for epoch in range(100):
    def closure():
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        return loss
    
    loss = optimizer.step(closure)
    
    if epoch % 20 == 0:
        with torch.no_grad():
            val_preds = model(X_val).argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean()
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Val Acc={val_acc:.1%}")

# Final evaluation
with torch.no_grad():
    val_preds = model(X_val).argmax(dim=1)
    final_acc = (val_preds == y_val).float().mean()
    print(f"\nFinal Validation Accuracy: {final_acc:.1%}")
```

## Notes

- **Swarm Size**: 30 for CNN-sized models
- **Iterations**: 100-200 typically sufficient
- **Loss**: Use CrossEntropyLoss (no sigmoid needed)
- **GPU**: Enable with `device="cuda"` for faster training

## Tips for Image Classification

1. **Start Simple**: Try MLP first, then CNN
2. **Smaller Swarms**: 20-30 for larger models
3. **More Iterations**: Image tasks need 200-500 steps
4. **Consider HPO**: Use HPO to find optimal architecture
