import torch
import torch.nn as nn
from swarmtorch.base.generic_search import GenericSwarmSearch
from swarmtorch.swarm.model_training.pso import PSO

def build_model(params):
    return nn.Linear(params['in_features'], 2)
    
param_space = {
    'in_features': [10, 20, 30],
    'lr': (0.001, 0.1)
}

def train_fn(model, params):
    # dummy cost function that wants in_features=20 and lr=0.05
    return abs(params['lr'] - 0.05) + abs(params['in_features'] - 20)
    
search = GenericSwarmSearch(
    optimizer_class=PSO,
    model_fn=build_model,
    param_space=param_space,
    train_fn=train_fn,
    iterations=10,
    swarm_size=10,
    device="cpu",
    verbose=True
)

best_params = search.search()
print("BEST PARAMS:", best_params)
