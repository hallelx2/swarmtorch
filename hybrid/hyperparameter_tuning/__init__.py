from swarmtorch.base.generic_search import GenericSwarmSearch
from swarmtorch.hybrid.model_training import (
    EHO,
    ChickenSwarm,
    SMA,
    CatSwarm,
    Cockroach,
    Coati,
    Gorilla,
    JSO,
    KHA,
)


class EHOSearch(GenericSwarmSearch):
    """EHOSearch hyperparameter search using EHO."""

    def __init__(self, *args, **kwargs):
        super().__init__(EHO, *args, **kwargs)


class ChickenSwarmSearch(GenericSwarmSearch):
    """ChickenSwarmSearch hyperparameter search using ChickenSwarm."""

    def __init__(self, *args, **kwargs):
        super().__init__(ChickenSwarm, *args, **kwargs)


class SMASearch(GenericSwarmSearch):
    """SMASearch hyperparameter search using SMA."""

    def __init__(self, *args, **kwargs):
        super().__init__(SMA, *args, **kwargs)


class CatSwarmSearch(GenericSwarmSearch):
    """CatSwarmSearch hyperparameter search using CatSwarm."""

    def __init__(self, *args, **kwargs):
        super().__init__(CatSwarm, *args, **kwargs)


class CockroachSearch(GenericSwarmSearch):
    """CockroachSearch hyperparameter search using Cockroach."""

    def __init__(self, *args, **kwargs):
        super().__init__(Cockroach, *args, **kwargs)


class CoatiSearch(GenericSwarmSearch):
    """CoatiSearch hyperparameter search using Coati."""

    def __init__(self, *args, **kwargs):
        super().__init__(Coati, *args, **kwargs)


class GorillaSearch(GenericSwarmSearch):
    """GorillaSearch hyperparameter search using Gorilla."""

    def __init__(self, *args, **kwargs):
        super().__init__(Gorilla, *args, **kwargs)


class JSOSearch(GenericSwarmSearch):
    """JSOSearch hyperparameter search using JSO."""

    def __init__(self, *args, **kwargs):
        super().__init__(JSO, *args, **kwargs)


class KHASearch(GenericSwarmSearch):
    """KHASearch hyperparameter search using KHA."""

    def __init__(self, *args, **kwargs):
        super().__init__(KHA, *args, **kwargs)


__all__ = [
    "EHOSearch",
    "ChickenSwarmSearch",
    "SMASearch",
    "CatSwarmSearch",
    "CockroachSearch",
    "CoatiSearch",
    "GorillaSearch",
    "JSOSearch",
    "KHASearch",
]
