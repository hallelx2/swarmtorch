import torch
from torch import nn
from swarmtorch.base.generic_search import GenericSwarmSearch
from swarmtorch.swarm.model_training.pso import PSO

def test_generic_search():
    def build_model(params):
        return nn.Linear(params['in_features'], 2)
        
    param_space = {
        'in_features': [10, 20, 30],
        'lr': (0.001, 0.1)
    }

    def train_fn(model, params):
        return abs(params['lr'] - 0.05) + abs(params['in_features'] - 20)
        
    search = GenericSwarmSearch(
        optimizer_class=PSO,
        model_fn=build_model,
        param_space=param_space,
        train_fn=train_fn,
        iterations=5,
        swarm_size=5,
        device="cpu",
        verbose=False
    )

    best_params = search.search()
    assert 'lr' in best_params
    assert 'in_features' in best_params
