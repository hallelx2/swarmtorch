# Autonomous Benchmarking Report
Generated on: Sun Mar  8 14:32:48 2026

## Part 1: Neural Network Weight Optimization
In this experiment, we compared metaheuristic swarm optimizers against traditional gradient-based methods (Adam, SGD) for training a non-linear classification model.

| Optimizer | Final Loss |
|-----------|------------|
| Adam      | 0.015321 |
| SGD       | 0.536656 |
| PSO       | 0.195314 |
| GWO       | 0.682548 |
| GA        | 0.328369 |
| DE        | 0.257196 |
| HHO       | 0.101170 |
| TLBO      | 0.622300 |

## Part 2: Hyperparameter Optimization (HPO)
We evaluated the efficiency of swarm-based hyperparameter tuning against Random Search.

| Searcher  | Best Accuracy |
|-----------|---------------|
| Random    | 0.9900 |
| PSO       | 0.9900 |
| GWO       | 0.9800 |
| WOA       | 0.9900 |
| HHO       | 0.9900 |
| GA        | 0.9850 |
| DE        | 0.9900 |

**Note:** Visualizations have been saved as `training_convergence.png` and `hpo_comparison.png`.