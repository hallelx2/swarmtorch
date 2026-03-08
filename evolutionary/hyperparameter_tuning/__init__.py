from swarmtorch.base.generic_search import GenericSwarmSearch
from swarmtorch.evolutionary.model_training import (
    DE, GA, CEM, PFA, ARS, FDA, CA, PBIL
)

class DESearch(GenericSwarmSearch):
    """DESearch hyperparameter search using DE."""
    def __init__(self, *args, **kwargs):
        super().__init__(DE, *args, **kwargs)

class GASearch(GenericSwarmSearch):
    """GASearch hyperparameter search using GA."""
    def __init__(self, *args, **kwargs):
        super().__init__(GA, *args, **kwargs)

class CEMSearch(GenericSwarmSearch):
    """CEMSearch hyperparameter search using CEM."""
    def __init__(self, *args, **kwargs):
        super().__init__(CEM, *args, **kwargs)

class PFASearch(GenericSwarmSearch):
    """PFASearch hyperparameter search using PFA."""
    def __init__(self, *args, **kwargs):
        super().__init__(PFA, *args, **kwargs)

class ARSSearch(GenericSwarmSearch):
    """ARSSearch hyperparameter search using ARS."""
    def __init__(self, *args, **kwargs):
        super().__init__(ARS, *args, **kwargs)

class FDASearch(GenericSwarmSearch):
    """FDASearch hyperparameter search using FDA."""
    def __init__(self, *args, **kwargs):
        super().__init__(FDA, *args, **kwargs)

class CASearch(GenericSwarmSearch):
    """CASearch hyperparameter search using CA."""
    def __init__(self, *args, **kwargs):
        super().__init__(CA, *args, **kwargs)

class PBILSearch(GenericSwarmSearch):
    """PBILSearch hyperparameter search using PBIL."""
    def __init__(self, *args, **kwargs):
        super().__init__(PBIL, *args, **kwargs)

__all__ = [
    "DESearch", "GASearch", "CEMSearch", "PFASearch", "ARSSearch", "FDASearch", "CASearch", "PBILSearch"
]
