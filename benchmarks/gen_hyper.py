import os
import ast

categories = [
    "bio_inspired",
    "evolutionary",
    "human_based",
    "physics",
    "hybrid",
    "swarm",
]

for category in categories:
    base_dir = f"swarmtorch/{category}"
    mt_init = f"{base_dir}/model_training/__init__.py"

    # Read __all__ from mt_init
    with open(mt_init, "r") as f:
        content = f.read()

    tree = ast.parse(content)
    all_exports = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    all_exports = [elt.value for elt in node.value.elts]
                    break

    # Generate hyperparameter_tuning dir
    ht_dir = f"{base_dir}/hyperparameter_tuning"
    os.makedirs(ht_dir, exist_ok=True)

    ht_init = f"{ht_dir}/__init__.py"

    with open(ht_init, "w") as f:
        f.write("from swarmtorch.base.generic_search import GenericSwarmSearch\n")
        f.write(f"from swarmtorch.{category}.model_training import (\n    ")
        f.write(", ".join(all_exports))
        f.write("\n)\n\n")

        search_exports = []
        for export in all_exports:
            search_name = f"{export}Search"
            if search_name == "CuckooSearchSearch":  # handle edge cases
                search_name = "CuckooSearchHT"
            if search_name == "RandomSearchSearch":
                search_name = "RandomSearchHT"
            if search_name == "HarmonySearchSearch":
                search_name = "HarmonySearchHT"

            search_exports.append(search_name)

            f.write(f"class {search_name}(GenericSwarmSearch):\n")
            f.write(f'    """{search_name} hyperparameter search using {export}."""\n')
            f.write("    def __init__(self, *args, **kwargs):\n")
            f.write(f"        super().__init__({export}, *args, **kwargs)\n\n")

        f.write("__all__ = [\n    ")
        f.write(", ".join(f'"{s}"' for s in search_exports))
        f.write("\n]\n")
