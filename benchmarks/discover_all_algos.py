import importlib
import os

categories = ['bio_inspired', 'evolutionary', 'human_based', 'physics', 'hybrid', 'swarm']

train_algos = {}
tune_algos = {}

for cat in categories:
    try:
        mod_train = importlib.import_module(f"swarmtorch.{cat}.model_training")
        train_algos[cat] = getattr(mod_train, "__all__", [])
    except Exception as e:
        train_algos[cat] = []
        
    try:
        mod_tune = importlib.import_module(f"swarmtorch.{cat}.hyperparameter_tuning")
        tune_algos[cat] = getattr(mod_tune, "__all__", [])
    except Exception as e:
        tune_algos[cat] = []

print("=== Training Optimizers Found ===")
total_train = 0
for cat, algos in train_algos.items():
    print(f"{cat}: {len(algos)} algorithms")
    total_train += len(algos)
print(f"Total Model Training Algorithms: {total_train}")

print("\n=== HPO Searchers Found ===")
total_tune = 0
for cat, algos in tune_algos.items():
    print(f"{cat}: {len(algos)} algorithms")
    total_tune += len(algos)
print(f"Total HPO Tuners: {total_tune}")

# Export for the benchmark script
with open("algo_registry.py", "w") as f:
    f.write(f"TRAIN_MAP = {train_algos}\n")
    f.write(f"TUNE_MAP = {tune_algos}\n")
