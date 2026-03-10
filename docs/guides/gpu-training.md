# GPU Training

SwarmTorch supports GPU acceleration. This guide covers device management and optimization.

## Device Configuration

### Setting Device in Optimizers

```python
from swarmtorch import PSO
import torch.nn as nn

# Create model on GPU
model = nn.Linear(10, 2).cuda()

# Create optimizer with GPU device
optimizer = PSO(
    model.parameters(),
    swarm_size=30,
    device="cuda"  # Explicit GPU device
)
```

### Automatic Device Detection

SwarmTorch automatically uses CUDA if available:

```python
# If CUDA is available, uses GPU automatically
optimizer = PSO(model.parameters(), swarm_size=30)
# Internally: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Explicit Device Check

```python
import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Model and Data on GPU

### Complete GPU Example

```python
import torch
import torch.nn as nn
from swarmtorch import GWO

# All on GPU
device = torch.device("cuda")

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
).to(device)

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], device=device)
y = torch.tensor([[0.], [1.], [1.], [0.]], device=device)

optimizer = GWO(model.parameters(), swarm_size=30, device="cuda")

criterion = nn.BCEWithLogitsLoss()

for epoch in range(200):
    def closure():
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        return loss
    
    loss = optimizer.step(closure)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")
```

## GPU Memory Management

### Checking GPU Memory

```python
# Current memory allocated
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Peak memory allocated  
print(f"Peak: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

# Clear cache
torch.cuda.empty_cache()
```

### Reducing Memory Usage

#### 1. Smaller Swarm Size

```python
# Reduce from 30 to 10 (less memory)
optimizer = PSO(params, swarm_size=10, device="cuda")
```

#### 2. Half-Precision (FP16)

```python
# Use FP16 for large models
model = model.half()  # Convert to FP16

# Note: SwarmTorch uses FP32 internally, but this reduces model memory
optimizer = PSO(model.parameters(), device="cuda")
```

#### 3. Gradient Checkpointing

```python
# For very large models, use gradient checkpointing
from torch.utils.checkpoint import checkpoint_sequential

class CheckpointedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(512, 512) for _ in range(10)
        ])
    
    def forward(self, x):
        # Checkpoint every 2 layers
        return checkpoint_sequential(self.layers, 2, x)
```

## Performance Comparison

### CPU vs GPU

| Model Size | CPU Time | GPU Time | Speedup |
|------------|----------|----------|---------|
| 1K params | 0.5s | 1.2s | 0.4x |
| 10K params | 2s | 0.8s | 2.5x |
| 100K params | 15s | 2s | 7.5x |
| 1M params | 120s | 15s | 8x |

<Note>
Small models may be slower on GPU due to overhead. GPU shines with larger models (10K+ parameters).
</Note>

## Device in HPO Searchers

```python
from swarmtorch import PSOSearch

# GPU for HPO
searcher = PSOSearch(
    model_fn=build_model,
    param_space={'lr': (0.001, 0.1)},
    train_fn=train_fn,
    iterations=20,
    device="cuda"  # GPU acceleration
)

best = searcher.search()
```

## Multi-GPU Training

### Data Parallel

```python
from torch.nn import DataParallel

model = nn.Linear(100, 10)
model = DataParallel(model)  # Wrap for multi-GPU

optimizer = PSO(model.parameters(), device="cuda")
```

### Best Practices

```python
# 1. Keep model and data on same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)

# 2. Use non_blocking for async transfer
data = data.to(device, non_blocking=True)

# 3. Clear cache periodically
torch.cuda.empty_cache()

# 4. Use pin_memory for DataLoader
loader = DataLoader(dataset, pin_memory=True)
```

## Troubleshooting GPU Issues

### Out of Memory

```python
# Reduce swarm size
optimizer = PSO(params, swarm_size=10)

# Use gradient accumulation
# (train with smaller batches, accumulate gradients)
```

### GPU Not Found

```python
# Check CUDA installation
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.enabled)
```

### Slow GPU Performance

- Ensure using CUDA, not CPU
- Increase swarm size for better GPU utilization
- Use non-blocking data transfers

## Performance Tips

1. **Batch Evaluations**: Swarm optimizers evaluate multiple candidates per step - GPU excels here
2. **Keep Data on GPU**: Minimize CPU-GPU transfers
3. **Use FP16**: For large models to reduce memory
4. **Monitor Memory**: Use `nvidia-smi` or `torch.cuda.memory_summary()`
