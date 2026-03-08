import torch
import torch.nn as nn
import importlib
import traceback

categories = ["bio_inspired", "evolutionary", "human_based", "physics", "hybrid", "swarm"]

def build_model(params):
    return nn.Linear(params['in_features'], 2)

param_space = {
    'in_features': [10, 20],
    'lr': (0.01, 0.1)
}

def train_fn(model, params):
    # Dummy training fitness: try to find lr=0.05 and in_features=20
    return abs(params['lr'] - 0.05) + abs(params['in_features'] - 20)

def main():
    success_count = 0
    fail_count = 0
    failed_optimizers = []

    for category in categories:
        print(f"\n--- Testing category: {category} ---")
        try:
            module = importlib.import_module(f"swarmtorch.{category}.hyperparameter_tuning")
            exports = getattr(module, '__all__', [])
            
            for search_name in exports:
                search_class = getattr(module, search_name)
                
                print(f"Testing {search_name}...", end=" ")
                try:
                    search_instance = search_class(
                        model_fn=build_model,
                        param_space=param_space,
                        train_fn=train_fn,
                        iterations=2,
                        swarm_size=5,
                        device="cpu", # Force CPU for quick testing
                        verbose=False
                    )
                    best_params = search_instance.search()
                    print("SUCCESS")
                    success_count += 1
                except Exception as e:
                    print(f"FAILED: {e}")
                    fail_count += 1
                    failed_optimizers.append((search_name, str(e)))
        except Exception as e:
            print(f"Failed to import category {category}: {e}")

    print(f"\n=== Testing Complete ===")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")

    if failed_optimizers:
        print("\nFailed Optimizers details:")
        for name, err in failed_optimizers:
            print(f"- {name}: {err}")

if __name__ == "__main__":
    main()
