from swarmtorch.base.generic_search import GenericSwarmSearch
from swarmtorch.bio_inspired.model_training import (
    ALO, BBO, MVO, SineCosine, MFO
)

class ALOSearch(GenericSwarmSearch):
    """ALOSearch hyperparameter search using ALO."""
    def __init__(self, *args, **kwargs):
        super().__init__(ALO, *args, **kwargs)

class BBOSearch(GenericSwarmSearch):
    """BBOSearch hyperparameter search using BBO."""
    def __init__(self, *args, **kwargs):
        super().__init__(BBO, *args, **kwargs)

class MVOSearch(GenericSwarmSearch):
    """MVOSearch hyperparameter search using MVO."""
    def __init__(self, *args, **kwargs):
        super().__init__(MVO, *args, **kwargs)

class SineCosineSearch(GenericSwarmSearch):
    """SineCosineSearch hyperparameter search using SineCosine."""
    def __init__(self, *args, **kwargs):
        super().__init__(SineCosine, *args, **kwargs)

class MFOSearch(GenericSwarmSearch):
    """MFOSearch hyperparameter search using MFO."""
    def __init__(self, *args, **kwargs):
        super().__init__(MFO, *args, **kwargs)

__all__ = [
    "ALOSearch", "BBOSearch", "MVOSearch", "SineCosineSearch", "MFOSearch"
]
