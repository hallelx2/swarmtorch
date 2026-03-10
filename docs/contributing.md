# Contributing to SwarmTorch

Thank you for your interest in contributing to SwarmTorch!

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Git

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/hallelx2/swarmtorch.git
cd swarmtorch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[all]"
```

## Project Structure

```
swarmtorch/
├── swarmtorch/           # Main package
│   ├── base/            # Base classes
│   ├── swarm/           # Swarm intelligence algorithms
│   ├── evolutionary/    # Evolutionary algorithms
│   ├── physics/         # Physics-based algorithms
│   ├── bio_inspired/   # Bio-inspired algorithms
│   ├── human_based/    # Human-based algorithms
│   ├── hybrid/         # Hybrid algorithms
│   └── utils/          # Utilities
├── docs/                # Documentation
├── benchmarks/          # Benchmark scripts
└── tests/              # Test suite
```

## Adding New Algorithms

### Step 1: Create Model Training Optimizer

Create a new file in the appropriate category:

```python
# swarmtorch/swarm/model_training/my_algorithm.py

from swarmtorch.base import SwarmOptimizer

class MyAlgorithm(SwarmOptimizer):
    """Description of your algorithm."""
    
    def __init__(self, params, swarm_size=30, device="cpu", **kwargs):
        super().__init__(params, swarm_size=swarm_size, device=device)
        # Your initialization
    
    def _init_swarm(self):
        # Initialize positions, velocities, etc.
        pass
    
    def _update_positions(self):
        # Update particle positions
        pass
```

### Step 2: Update Category Exports

Add to `<category>/model_training/__init__.py`:
```python
from swarmtorch.swarm.model_training.my_algorithm import MyAlgorithm

__all__ = [..., "MyAlgorithm"]
```

### Step 3: Add HPO Searcher (Optional)

For hyperparameter tuning support, add to `<category>/hyperparameter_tuning/__init__.py`:

```python
from swarmtorch.base import GenericSwarmSearch
from swarmtorch.swarm.model_training import MyAlgorithm

class MyAlgorithmSearch(GenericSwarmSearch):
    def __init__(self, *args, **kwargs):
        super().__init__(MyAlgorithm, *args, **kwargs)
```

### Step 4: Add to Top-Level Exports

Add to `swarmtorch/__init__.py`:
```python
from swarmtorch.swarm.model_training import MyAlgorithm
from swarmtorch.swarm.hyperparameter_tuning import MyAlgorithmSearch

__all__ = [..., "MyAlgorithm", "MyAlgorithmSearch"]
```

## Code Style

### Format

We use Ruff for formatting:

```bash
# Format code
ruff format .

# Check linting
ruff check .
```

### Type Hints

All public APIs should have type hints:

```python
def my_function(param1: int, param2: str = "default") -> None:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def my_method(self, param: int) -> float:
    """Short description.

    Args:
        param: Description of param.

    Returns:
        Description of return value.

    Example:
        >>> result = my_method(42)
    """
    return 0.0
```

## Testing

### Run Tests

```bash
pytest tests/ -v
```

### Add Tests

```python
# tests/test_my_algorithm.py

import pytest
import torch
from swarmtorch import MyAlgorithm

class TestMyAlgorithm:
    def test_init(self):
        model = torch.nn.Linear(10, 2)
        opt = MyAlgorithm(model.parameters(), swarm_size=10)
        assert opt.defaults["swarm_size"] == 10
    
    def test_step(self):
        model = torch.nn.Linear(10, 2)
        opt = MyAlgorithm(model.parameters(), swarm_size=5)
        
        def closure():
            return torch.tensor(1.0)
        
        loss = opt.step(closure)
        assert loss is not None
```

## Documentation

### Adding Docs

1. Add docstrings to your algorithm
2. Update API reference in `docs/api-reference/`
3. Add example if applicable in `docs/examples/`

### Building Docs Locally

Mintlify docs are in the `docs/` folder. To preview:

```bash
# Install Mintlify CLI
npm install -g mintlify

# Preview locally
mintlify dev
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Run linting: `ruff check .`
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

## Issue Reporting

Found a bug? Please open an issue with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details

## Questions?

- Open an issue for questions
- Join discussions in GitHub Issues
