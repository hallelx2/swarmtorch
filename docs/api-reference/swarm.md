# Swarm Intelligence

32 model-training optimizers and 32 HPO searchers.

## Model Training Optimizers

### PSO

Particle Swarm Optimization optimizer.

```python
from swarmtorch import PSO
optimizer = PSO(params, swarm_size=30, w=0.7, c1=1.5, c2=1.5, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of particles (default: 30)
- `w`: Inertia weight (default: 0.7)
- `c1`: Cognitive coefficient (default: 1.5)
- `c2`: Social coefficient (default: 1.5)
- `device`: Device (default: "cpu")

---

### GWO

Grey Wolf Optimizer for model training.

```python
from swarmtorch import GWO
optimizer = GWO(params, swarm_size=30, a_decay=0.001, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of wolves (default: 30)
- `a_decay`: Linear parameter decay (default: 0.001)
- `device`: Device (default: "cpu")

---

### WOA

Whale Optimization Algorithm for model training.

```python
from swarmtorch import WOA
optimizer = WOA(params, swarm_size=30, b=0.5, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of whales (default: 30)
- `b`: Spiral shape parameter (default: 0.5)
- `device`: Device (default: "cpu")

---

### HHO

Harris Hawks Optimization for model training.

```python
from swarmtorch import HHO
optimizer = HHO(params, swarm_size=30, beta=1.5, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of hawks (default: 30)
- `beta`: Beta parameter (default: 1.5)
- `device`: Device (default: "cpu")

---

### SSA

Sparrow Search Algorithm for model training.

```python
from swarmtorch import SSA
optimizer = SSA(params, swarm_size=30, ST=0.8, PD=0.2, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of sparrows (default: 30)
- `ST`: Safety threshold (default: 0.8)
- `PD`: Producer proportion (default: 0.2)
- `device`: Device (default: "cpu")

---

### Firefly

Firefly Algorithm for model training.

```python
from swarmtorch import Firefly
optimizer = Firefly(params, swarm_size=30, alpha=0.5, beta0=1.0, gamma=1.0, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of fireflies (default: 30)
- `alpha`: Randomization parameter (default: 0.5)
- `beta0`: Attractiveness at r=0 (default: 1.0)
- `gamma`: Light absorption coefficient (default: 1.0)
- `device`: Device (default: "cpu")

---

### Bat

Bat Algorithm for model training.

```python
from swarmtorch import Bat
optimizer = Bat(params, swarm_size=30, A=0.5, r=0.5, fmin=0, fmax=2, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of bats (default: 30)
- `A`: Loudness (default: 0.5)
- `r`: Pulse rate (default: 0.5)
- `fmin`: Minimum frequency (default: 0)
- `fmax`: Maximum frequency (default: 2)
- `device`: Device (default: "cpu")

---

### Dragonfly

Dragonfly Algorithm for model training.

```python
from swarmtorch import Dragonfly
optimizer = Dragonfly(params, swarm_size=30, alpha=0.1, beta=0.1, gamma=0.1, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of dragonflies (default: 30)
- `alpha`: Alignment weight (default: 0.1)
- `beta`: Cohesion weight (default: 0.1)
- `gamma`: Separation weight (default: 0.1)
- `device`: Device (default: "cpu")

---

### CuckooSearch

Cuckoo Search Algorithm for model training.

```python
from swarmtorch import CuckooSearch
optimizer = CuckooSearch(params, swarm_size=30, pa=0.25, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of cuckoos (default: 30)
- `pa`: Discovery probability (default: 0.25)
- `device`: Device (default: "cpu")

---

### Salp

Salp Chain Algorithm for model training.

```python
from swarmtorch import Salp
optimizer = Salp(params, swarm_size=30, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of salps (default: 30)
- `device`: Device (default: "cpu")

---

### Bee

Artificial Bee Colony for model training.

```python
from swarmtorch import Bee
optimizer = Bee(params, swarm_size=30, limit=10, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of bees (default: 30)
- `limit`: Abandonment limit (default: 10)
- `device`: Device (default: "cpu")

---

### Fish

Artificial Fish Swarm Algorithm for model training.

```python
from swarmtorch import Fish
optimizer = Fish(params, swarm_size=30, visual=0.3, step=0.1, delta=0.4, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of fish (default: 30)
- `visual`: Visual range (default: 0.3)
- `step`: Step size (default: 0.1)
- `delta`: Crowding factor (default: 0.4)
- `device`: Device (default: "cpu")

---

### DFO

Derandomized Optimization for model training.

```python
from swarmtorch import DFO
optimizer = DFO(params, swarm_size=30, mu=0.5, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of particles (default: 30)
- `mu`: Mutation rate (default: 0.5)
- `device`: Device (default: "cpu")

---

### MBO

Monarch Butterfly Optimization for model training.

```python
from swarmtorch import MBO
optimizer = MBO(params, swarm_size=30, period=1.5, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of butterflies (default: 30)
- `period`: Migration period (default: 1.5)
- `device`: Device (default: "cpu")

---

### CSA

Cheetah Swarm Algorithm for model training.

```python
from swarmtorch import CSA
optimizer = CSA(params, swarm_size=30, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of cheetahs (default: 30)
- `device`: Device (default: "cpu")

---

### AOA

Arithmetic Optimization Algorithm for model training.

```python
from swarmtorch import AOA
optimizer = AOA(params, swarm_size=30, alpha=0.5, mu=0.5, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of agents (default: 30)
- `alpha`: Exploration parameter (default: 0.5)
- `mu`: Convergence parameter (default: 0.5)
- `device`: Device (default: "cpu")

---

### SOS

Symbiotic Organisms Search for model training.

```python
from swarmtorch import SOS
optimizer = SOS(params, swarm_size=30, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of organisms (default: 30)
- `device`: Device (default: "cpu")

---

### DVBA

Differential Vector Biogeography for model training.

```python
from swarmtorch import DVBA
optimizer = DVBA(params, swarm_size=30, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of species (default: 30)
- `device`: Device (default: "cpu")

---

### ABCO

Artificial Bee Colony with Opposition for model training.

```python
from swarmtorch import ABCO
optimizer = ABCO(params, swarm_size=30, limit=10, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of bees (default: 30)
- `limit`: Abandonment limit (default: 10)
- `device`: Device (default: "cpu")

---

### GOA

Grasshopper Optimization Algorithm for model training.

```python
from swarmtorch import GOA
optimizer = GOA(params, swarm_size=30, c_min=0.00004, c_max=1.0, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of grasshoppers (default: 30)
- `c_min`: Minimum coefficient (default: 0.00004)
- `c_max`: Maximum coefficient (default: 1.0)
- `device`: Device (default: "cpu")

---

### HUS

Human Urban Search for model training.

```python
from swarmtorch import HUS
optimizer = HUS(params, swarm_size=30, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of agents (default: 30)
- `device`: Device (default: "cpu")

---

### JY

 Jaya Optimization for model training.

```python
from swarmtorch import JY
optimizer = JY(params, swarm_size=30, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of candidates (default: 30)
- `device`: Device (default: "cpu")

---

### SPBO

Stochastic Population-Based Optimization for model training.

```python
from swarmtorch import SPBO
optimizer = SPBO(params, swarm_size=30, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of particles (default: 30)
- `device`: Device (default: "cpu")

---

### RandomSearch

Random Search for model training.

```python
from swarmtorch import RandomSearch
optimizer = RandomSearch(params, swarm_size=30, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of candidates (default: 30)
- `device`: Device (default: "cpu")

---

### IGWO

Improved Grey Wolf Optimizer for model training.

```python
from swarmtorch import IGWO
optimizer = IGWO(params, swarm_size=30, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of wolves (default: 30)
- `device`: Device (default: "cpu")

---

### IWOA

Improved Whale Optimization Algorithm for model training.

```python
from swarmtorch import IWOA
optimizer = IWOA(params, swarm_size=30, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of whales (default: 30)
- `device`: Device (default: "cpu")

---

### ACGWO

Adaptive Chaotic Grey Wolf Optimizer for model training.

```python
from swarmtorch import ACGWO
optimizer = ACGWO(params, swarm_size=30, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of wolves (default: 30)
- `device`: Device (default: "cpu")

---

### Memetic

Memetic Algorithm for model training.

```python
from swarmtorch import Memetic
optimizer = Memetic(params, swarm_size=30, local_iter=5, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of memes (default: 30)
- `local_iter`: Local search iterations (default: 5)
- `device`: Device (default: "cpu")

---

### Clonalg

Clonal Selection Algorithm for model training.

```python
from swarmtorch import Clonalg
optimizer = Clonalg(params, swarm_size=30, beta=1.0, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of antibodies (default: 30)
- `beta`: Selection pressure (default: 1.0)
- `device`: Device (default: "cpu")

---

### AFSA

Artificial Fish Swarm Algorithm for model training.

```python
from swarmtorch import AFSA
optimizer = AFSA(params, swarm_size=30, visual=0.3, step=0.1, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of fish (default: 30)
- `visual`: Visual range (default: 0.3)
- `step`: Step size (default: 0.1)
- `device`: Device (default: "cpu")

---

### HSA

Harmony Search Algorithm for model training.

```python
from swarmtorch import HSA
optimizer = HSA(params, swarm_size=30, HMCR=0.7, PAR=0.3, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of harmonies (default: 30)
- `HMCR`: Harmony memory consideration rate (default: 0.7)
- `PAR`: Pitch adjustment rate (default: 0.3)
- `device`: Device (default: "cpu")

---

### DFO2

Distributed Faithful Optimization for model training.

```python
from swarmtorch import DFO2
optimizer = DFO2(params, swarm_size=30, device="cpu")
```

**Parameters:**
- `params`: Model parameters
- `swarm_size`: Number of particles (default: 30)
- `device`: Device (default: "cpu")

---

## HPO Searchers

### PSOSearch

PSO-based hyperparameter search.

```python
from swarmtorch import PSOSearch
searcher = PSOSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of particles (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### GWOSearch

GWO-based hyperparameter search.

```python
from swarmtorch import GWOSearch
searcher = GWOSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of wolves (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### WOASearch

WOA-based hyperparameter search.

```python
from swarmtorch import WOASearch
searcher = WOASearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of whales (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### HHOSearch

HHO-based hyperparameter search.

```python
from swarmtorch import HHOSearch
searcher = HHOSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of hawks (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### SSASearch

SSA-based hyperparameter search.

```python
from swarmtorch import SSASearch
searcher = SSASearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of sparrows (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### FireflySearch

Firefly-based hyperparameter search.

```python
from swarmtorch import FireflySearch
searcher = FireflySearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of fireflies (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### BatSearch

Bat-based hyperparameter search.

```python
from swarmtorch import BatSearch
searcher = BatSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of bats (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### DragonflySearch

Dragonfly-based hyperparameter search.

```python
from swarmtorch import DragonflySearch
searcher = DragonflySearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of dragonflies (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### CuckooSearchHT

Cuckoo Search hyperparameter tuning.

```python
from swarmtorch import CuckooSearchHT
searcher = CuckooSearchHT(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of cuckoos (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### SalpSearch

Salp-based hyperparameter search.

```python
from swarmtorch import SalpSearch
searcher = SalpSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of salps (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### BeeSearch

Bee-based hyperparameter search.

```python
from swarmtorch import BeeSearch
searcher = BeeSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of bees (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### FishSearch

Fish-based hyperparameter search.

```python
from swarmtorch import FishSearch
searcher = FishSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of fish (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### DFOSearch

DFO-based hyperparameter search.

```python
from swarmtorch import DFOSearch
searcher = DFOSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of particles (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### MBOSearch

MBO-based hyperparameter search.

```python
from swarmtorch import MBOSearch
searcher = MBOSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of butterflies (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### CSASearch

CSA-based hyperparameter search.

```python
from swarmtorch import CSASearch
searcher = CSASearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of cheetahs (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### AOASearch

AOA-based hyperparameter search.

```python
from swarmtorch import AOASearch
searcher = AOASearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of agents (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### SOSSearch

SOS-based hyperparameter search.

```python
from swarmtorch import SOSSearch
searcher = SOSSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of organisms (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### DVBASearch

DVBA-based hyperparameter search.

```python
from swarmtorch import DVBASearch
searcher = DVBASearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of species (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### ABCOSearch

ABCO-based hyperparameter search.

```python
from swarmtorch import ABCOSearch
searcher = ABCOSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of bees (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### GOASearch

GOA-based hyperparameter search.

```python
from swarmtorch import GOASearch
searcher = GOASearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of grasshoppers (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### HUSSearch

HUS-based hyperparameter search.

```python
from swarmtorch import HUSSearch
searcher = HUSSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of agents (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### JYSearch

Jaya-based hyperparameter search.

```python
from swarmtorch import JYSearch
searcher = JYSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of candidates (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### SPBOSearch

SPBO-based hyperparameter search.

```python
from swarmtorch import SPBOSearch
searcher = SPBOSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of particles (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### RandomSearchHT

Random search hyperparameter tuning.

```python
from swarmtorch import RandomSearchHT
searcher = RandomSearchHT(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of candidates (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### IGWOSearch

IGWO-based hyperparameter search.

```python
from swarmtorch import IGWOSearch
searcher = IGWOSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of wolves (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### IWOASearch

IWOA-based hyperparameter search.

```python
from swarmtorch import IWOASearch
searcher = IWOASearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of whales (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### ACGWOSearch

ACGWO-based hyperparameter search.

```python
from swarmtorch import ACGWOSearch
searcher = ACGWOSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of wolves (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### MemeticSearch

Memetic algorithm hyperparameter search.

```python
from swarmtorch import MemeticSearch
searcher = MemeticSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of memes (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### ClonalgSearch

Clonal Selection hyperparameter search.

```python
from swarmtorch import ClonalgSearch
searcher = ClonalgSearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of antibodies (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### AFSASearch

AFSA-based hyperparameter search.

```python
from swarmtorch import AFSASearch
searcher = AFSASearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of fish (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### HSASearch

Harmony Search hyperparameter tuning.

```python
from swarmtorch import HSASearch
searcher = HSASearch(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of harmonies (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)

---

### DFO2Search

DFO2-based hyperparameter search.

```python
from swarmtorch import DFO2Search
searcher = DFO2Search(model_fn, param_space, train_fn, iterations=50, swarm_size=10)
```

**Parameters:**
- `model_fn`: Function to build model from params
- `param_space`: Hyperparameter search space
- `train_fn`: Training function returning loss
- `iterations`: Number of iterations (default: 50)
- `swarm_size`: Number of particles (default: 10)
- `device`: Device (default: "cpu")
- `verbose`: Print progress (default: True)
