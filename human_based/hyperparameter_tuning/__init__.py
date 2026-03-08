from swarmtorch.base.generic_search import GenericSwarmSearch
from swarmtorch.human_based.model_training import (
    TLBO, HarmonySearch
)

class TLBOSearch(GenericSwarmSearch):
    """TLBOSearch hyperparameter search using TLBO."""
    def __init__(self, *args, **kwargs):
        super().__init__(TLBO, *args, **kwargs)

class HarmonySearchHT(GenericSwarmSearch):
    """HarmonySearchHT hyperparameter search using HarmonySearch."""
    def __init__(self, *args, **kwargs):
        super().__init__(HarmonySearch, *args, **kwargs)

__all__ = [
    "TLBOSearch", "HarmonySearchHT"
]
