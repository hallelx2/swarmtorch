from swarmtorch.base.generic_search import GenericSwarmSearch
from swarmtorch.physics.model_training import SA, GSA, FPA


class SASearch(GenericSwarmSearch):
    """SASearch hyperparameter search using SA."""

    def __init__(self, *args, **kwargs):
        super().__init__(SA, *args, **kwargs)


class GSASearch(GenericSwarmSearch):
    """GSASearch hyperparameter search using GSA."""

    def __init__(self, *args, **kwargs):
        super().__init__(GSA, *args, **kwargs)


class FPASearch(GenericSwarmSearch):
    """FPASearch hyperparameter search using FPA."""

    def __init__(self, *args, **kwargs):
        super().__init__(FPA, *args, **kwargs)


__all__ = ["SASearch", "GSASearch", "FPASearch"]
