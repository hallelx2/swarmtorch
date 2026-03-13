# Installation

SwarmTorch supports Python 3.10+ and requires PyTorch 2.0+.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- NumPy >= 1.21.0

Optional (for benchmarks):
- Pandas >= 1.3.0
- Matplotlib >= 3.5.0

## Install from PyPI

```bash
pip install swarmtorch
```

## Install with uv (Recommended)

```bash
uv add swarmtorch
```

## Install from Source

```bash
git clone https://github.com/hallelx2/swarmtorch.git
cd swarmtorch
pip install -e .
```

## Install with Benchmark Dependencies

To run the benchmark scripts included in the library:

```bash
pip install swarmtorch[benchmarks]
```

Or install all dependencies including dev tools:

```bash
pip install swarmtorch[all]
```

## Verify Installation

```python
import swarmtorch
print(f"SwarmTorch version: {swarmtorch.__version__ if hasattr(swarmtorch, '__version__') else 'installed'}")
print(f"Available optimizers: {len([x for x in dir(swarmtorch) if not x.startswith('_')])} classes")
```

## GPU Support

SwarmTorch automatically detects CUDA. To explicitly use GPU:

```python
optimizer = PSO(model.parameters(), swarm_size=30, device="cuda")
```

<Note>
Make sure PyTorch is installed with CUDA support for GPU acceleration. See [PyTorch installation](https://pytorch.org/get-started/locally/) for details.
</Note>
