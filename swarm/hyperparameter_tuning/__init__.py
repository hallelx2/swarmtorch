from swarmtorch.base.generic_search import GenericSwarmSearch
from swarmtorch.swarm.model_training import (
    PSO,
    GWO,
    WOA,
    HHO,
    SSA,
    Firefly,
    Bat,
    Dragonfly,
    CuckooSearch,
    Salp,
    Bee,
    Fish,
    DFO,
    MBO,
    CSA,
    AOA,
    SOS,
    DVBA,
    ABCO,
    GOA,
    HUS,
    JY,
    SPBO,
    RandomSearch,
    IGWO,
    IWOA,
    ACGWO,
    Memetic,
    Clonalg,
    AFSA,
    HSA,
    DFO2,
)


class PSOSearch(GenericSwarmSearch):
    """PSOSearch hyperparameter search using PSO."""

    def __init__(self, *args, **kwargs):
        super().__init__(PSO, *args, **kwargs)


class GWOSearch(GenericSwarmSearch):
    """GWOSearch hyperparameter search using GWO."""

    def __init__(self, *args, **kwargs):
        super().__init__(GWO, *args, **kwargs)


class WOASearch(GenericSwarmSearch):
    """WOASearch hyperparameter search using WOA."""

    def __init__(self, *args, **kwargs):
        super().__init__(WOA, *args, **kwargs)


class HHOSearch(GenericSwarmSearch):
    """HHOSearch hyperparameter search using HHO."""

    def __init__(self, *args, **kwargs):
        super().__init__(HHO, *args, **kwargs)


class SSASearch(GenericSwarmSearch):
    """SSASearch hyperparameter search using SSA."""

    def __init__(self, *args, **kwargs):
        super().__init__(SSA, *args, **kwargs)


class FireflySearch(GenericSwarmSearch):
    """FireflySearch hyperparameter search using Firefly."""

    def __init__(self, *args, **kwargs):
        super().__init__(Firefly, *args, **kwargs)


class BatSearch(GenericSwarmSearch):
    """BatSearch hyperparameter search using Bat."""

    def __init__(self, *args, **kwargs):
        super().__init__(Bat, *args, **kwargs)


class DragonflySearch(GenericSwarmSearch):
    """DragonflySearch hyperparameter search using Dragonfly."""

    def __init__(self, *args, **kwargs):
        super().__init__(Dragonfly, *args, **kwargs)


class CuckooSearchHT(GenericSwarmSearch):
    """CuckooSearchHT hyperparameter search using CuckooSearch."""

    def __init__(self, *args, **kwargs):
        super().__init__(CuckooSearch, *args, **kwargs)


class SalpSearch(GenericSwarmSearch):
    """SalpSearch hyperparameter search using Salp."""

    def __init__(self, *args, **kwargs):
        super().__init__(Salp, *args, **kwargs)


class BeeSearch(GenericSwarmSearch):
    """BeeSearch hyperparameter search using Bee."""

    def __init__(self, *args, **kwargs):
        super().__init__(Bee, *args, **kwargs)


class FishSearch(GenericSwarmSearch):
    """FishSearch hyperparameter search using Fish."""

    def __init__(self, *args, **kwargs):
        super().__init__(Fish, *args, **kwargs)


class DFOSearch(GenericSwarmSearch):
    """DFOSearch hyperparameter search using DFO."""

    def __init__(self, *args, **kwargs):
        super().__init__(DFO, *args, **kwargs)


class MBOSearch(GenericSwarmSearch):
    """MBOSearch hyperparameter search using MBO."""

    def __init__(self, *args, **kwargs):
        super().__init__(MBO, *args, **kwargs)


class CSASearch(GenericSwarmSearch):
    """CSASearch hyperparameter search using CSA."""

    def __init__(self, *args, **kwargs):
        super().__init__(CSA, *args, **kwargs)


class AOASearch(GenericSwarmSearch):
    """AOASearch hyperparameter search using AOA."""

    def __init__(self, *args, **kwargs):
        super().__init__(AOA, *args, **kwargs)


class SOSSearch(GenericSwarmSearch):
    """SOSSearch hyperparameter search using SOS."""

    def __init__(self, *args, **kwargs):
        super().__init__(SOS, *args, **kwargs)


class DVBASearch(GenericSwarmSearch):
    """DVBASearch hyperparameter search using DVBA."""

    def __init__(self, *args, **kwargs):
        super().__init__(DVBA, *args, **kwargs)


class ABCOSearch(GenericSwarmSearch):
    """ABCOSearch hyperparameter search using ABCO."""

    def __init__(self, *args, **kwargs):
        super().__init__(ABCO, *args, **kwargs)


class GOASearch(GenericSwarmSearch):
    """GOASearch hyperparameter search using GOA."""

    def __init__(self, *args, **kwargs):
        super().__init__(GOA, *args, **kwargs)


class HUSSearch(GenericSwarmSearch):
    """HUSSearch hyperparameter search using HUS."""

    def __init__(self, *args, **kwargs):
        super().__init__(HUS, *args, **kwargs)


class JYSearch(GenericSwarmSearch):
    """JYSearch hyperparameter search using JY."""

    def __init__(self, *args, **kwargs):
        super().__init__(JY, *args, **kwargs)


class SPBOSearch(GenericSwarmSearch):
    """SPBOSearch hyperparameter search using SPBO."""

    def __init__(self, *args, **kwargs):
        super().__init__(SPBO, *args, **kwargs)


class RandomSearchHT(GenericSwarmSearch):
    """RandomSearchHT hyperparameter search using RandomSearch."""

    def __init__(self, *args, **kwargs):
        super().__init__(RandomSearch, *args, **kwargs)


class IGWOSearch(GenericSwarmSearch):
    """IGWOSearch hyperparameter search using IGWO."""

    def __init__(self, *args, **kwargs):
        super().__init__(IGWO, *args, **kwargs)


class IWOASearch(GenericSwarmSearch):
    """IWOASearch hyperparameter search using IWOA."""

    def __init__(self, *args, **kwargs):
        super().__init__(IWOA, *args, **kwargs)


class ACGWOSearch(GenericSwarmSearch):
    """ACGWOSearch hyperparameter search using ACGWO."""

    def __init__(self, *args, **kwargs):
        super().__init__(ACGWO, *args, **kwargs)


class MemeticSearch(GenericSwarmSearch):
    """MemeticSearch hyperparameter search using Memetic."""

    def __init__(self, *args, **kwargs):
        super().__init__(Memetic, *args, **kwargs)


class ClonalgSearch(GenericSwarmSearch):
    """ClonalgSearch hyperparameter search using Clonalg."""

    def __init__(self, *args, **kwargs):
        super().__init__(Clonalg, *args, **kwargs)


class AFSASearch(GenericSwarmSearch):
    """AFSASearch hyperparameter search using AFSA."""

    def __init__(self, *args, **kwargs):
        super().__init__(AFSA, *args, **kwargs)


class HSASearch(GenericSwarmSearch):
    """HSASearch hyperparameter search using HSA."""

    def __init__(self, *args, **kwargs):
        super().__init__(HSA, *args, **kwargs)


class DFO2Search(GenericSwarmSearch):
    """DFO2Search hyperparameter search using DFO2."""

    def __init__(self, *args, **kwargs):
        super().__init__(DFO2, *args, **kwargs)


__all__ = [
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
]
