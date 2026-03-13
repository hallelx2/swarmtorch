# SwarmTorch: 120 metaheuristic optimization algorithms for PyTorch
# 60 model-training optimizers + 60 hyperparameter-tuning searchers

# --- Base classes ---
from swarmtorch.base import GenericSwarmSearch, HyperparameterSearch, SwarmOptimizer

# --- Swarm Intelligence (32 optimizers) ---
from swarmtorch.swarm.model_training import (
    ABCO,
    ACGWO,
    AFSA,
    AOA,
    Bat,
    Bee,
    Clonalg,
    CSA,
    CuckooSearch,
    DFO,
    DFO2,
    Dragonfly,
    DVBA,
    Firefly,
    Fish,
    GOA,
    GWO,
    HHO,
    HSA,
    HUS,
    IGWO,
    IWOA,
    JY,
    MBO,
    Memetic,
    PSO,
    RandomSearch,
    Salp,
    SOS,
    SPBO,
    SSA,
    WOA,
)

# --- Evolutionary (8 optimizers) ---
from swarmtorch.evolutionary.model_training import (
    ARS,
    CA,
    CEM,
    DE,
    FDA,
    GA,
    PBIL,
    PFA,
)

# --- Physics-Based (3 optimizers) ---
from swarmtorch.physics.model_training import FPA, GSA, SA

# --- Bio-Inspired (5 optimizers) ---
from swarmtorch.bio_inspired.model_training import ALO, BBO, MFO, MVO, SineCosine

# --- Human-Based (2 optimizers) ---
from swarmtorch.human_based.model_training import HarmonySearch, TLBO

# --- Hybrid (10 optimizers) ---
from swarmtorch.hybrid.model_training import (
    CatSwarm,
    ChickenSwarm,
    Coati,
    Cockroach,
    EHO,
    Gorilla,
    GorillaTroopsOptimizer,
    JSO,
    KHA,
    SMA,
)

# --- Swarm HPO Searchers (32 searchers) ---
from swarmtorch.swarm.hyperparameter_tuning import (
    ABCOSearch,
    ACGWOSearch,
    AFSASearch,
    AOASearch,
    BatSearch,
    BeeSearch,
    ClonalgSearch,
    CSASearch,
    CuckooSearchHT,
    DFO2Search,
    DFOSearch,
    DragonflySearch,
    DVBASearch,
    FireflySearch,
    FishSearch,
    GOASearch,
    GWOSearch,
    HHOSearch,
    HSASearch,
    HUSSearch,
    IGWOSearch,
    IWOASearch,
    JYSearch,
    MBOSearch,
    MemeticSearch,
    PSOSearch,
    RandomSearchHT,
    SalpSearch,
    SOSSearch,
    SPBOSearch,
    SSASearch,
    WOASearch,
)

# --- Evolutionary HPO Searchers (8 searchers) ---
from swarmtorch.evolutionary.hyperparameter_tuning import (
    ARSSearch,
    CASearch,
    CEMSearch,
    DESearch,
    FDASearch,
    GASearch,
    PBILSearch,
    PFASearch,
)

# --- Physics HPO Searchers (3 searchers) ---
from swarmtorch.physics.hyperparameter_tuning import FPASearch, GSASearch, SASearch

# --- Bio-Inspired HPO Searchers (5 searchers) ---
from swarmtorch.bio_inspired.hyperparameter_tuning import (
    ALOSearch,
    BBOSearch,
    MFOSearch,
    MVOSearch,
    SineCosineSearch,
)

# --- Human-Based HPO Searchers (2 searchers) ---
from swarmtorch.human_based.hyperparameter_tuning import HarmonySearchHT, TLBOSearch

# --- Hybrid HPO Searchers (10 searchers) ---
from swarmtorch.hybrid.hyperparameter_tuning import (
    CatSwarmSearch,
    ChickenSwarmSearch,
    CoatiSearch,
    CockroachSearch,
    EHOSearch,
    GorillaSearch,
    GorillaTroopsOptimizerSearch,
    JSOSearch,
    KHASearch,
    SMASearch,
)

__all__ = [
    # Base
    "SwarmOptimizer",
    "HyperparameterSearch",
    "GenericSwarmSearch",
    # Swarm - Model Training (32)
    "PSO",
    "GWO",
    "WOA",
    "HHO",
    "SSA",
    "Firefly",
    "Bat",
    "Dragonfly",
    "CuckooSearch",
    "Salp",
    "Bee",
    "Fish",
    "DFO",
    "MBO",
    "CSA",
    "AOA",
    "SOS",
    "DVBA",
    "ABCO",
    "GOA",
    "HUS",
    "JY",
    "SPBO",
    "RandomSearch",
    "IGWO",
    "IWOA",
    "ACGWO",
    "Memetic",
    "Clonalg",
    "AFSA",
    "HSA",
    "DFO2",
    # Evolutionary - Model Training (8)
    "DE",
    "GA",
    "CEM",
    "PFA",
    "ARS",
    "FDA",
    "CA",
    "PBIL",
    # Physics - Model Training (3)
    "SA",
    "GSA",
    "FPA",
    # Bio-Inspired - Model Training (5)
    "ALO",
    "BBO",
    "MVO",
    "SineCosine",
    "MFO",
    # Human-Based - Model Training (2)
    "TLBO",
    "HarmonySearch",
    # Hybrid - Model Training (10)
    "EHO",
    "ChickenSwarm",
    "SMA",
    "CatSwarm",
    "Cockroach",
    "Coati",
    "Gorilla",
    "GorillaTroopsOptimizer",
    "JSO",
    "KHA",
    # Swarm - HPO Searchers (32)
    "PSOSearch",
    "GWOSearch",
    "WOASearch",
    "HHOSearch",
    "SSASearch",
    "FireflySearch",
    "BatSearch",
    "DragonflySearch",
    "CuckooSearchHT",
    "SalpSearch",
    "BeeSearch",
    "FishSearch",
    "DFOSearch",
    "MBOSearch",
    "CSASearch",
    "AOASearch",
    "SOSSearch",
    "DVBASearch",
    "ABCOSearch",
    "GOASearch",
    "HUSSearch",
    "JYSearch",
    "SPBOSearch",
    "RandomSearchHT",
    "IGWOSearch",
    "IWOASearch",
    "ACGWOSearch",
    "MemeticSearch",
    "ClonalgSearch",
    "AFSASearch",
    "HSASearch",
    "DFO2Search",
    # Evolutionary - HPO Searchers (8)
    "DESearch",
    "GASearch",
    "CEMSearch",
    "PFASearch",
    "ARSSearch",
    "FDASearch",
    "CASearch",
    "PBILSearch",
    # Physics - HPO Searchers (3)
    "SASearch",
    "GSASearch",
    "FPASearch",
    # Bio-Inspired - HPO Searchers (5)
    "ALOSearch",
    "BBOSearch",
    "MVOSearch",
    "SineCosineSearch",
    "MFOSearch",
    # Human-Based - HPO Searchers (2)
    "TLBOSearch",
    "HarmonySearchHT",
    # Hybrid - HPO Searchers (10)
    "EHOSearch",
    "ChickenSwarmSearch",
    "SMASearch",
    "CatSwarmSearch",
    "CockroachSearch",
    "CoatiSearch",
    "GorillaSearch",
    "GorillaTroopsOptimizerSearch",
    "JSOSearch",
    "KHASearch",
]
