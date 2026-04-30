"""Stage 4.2 — Real NN training sweep.

Loads MNIST / CIFAR-10 via torchvision (downloading on first run) and
runs the curated paper algorithms across all four reference tasks.

Usage:
    python scripts/run_training.py \
        --output-dir results/training \
        --seeds 0 1 2 3 4 \
        --max-fe 3000

Pass ``--quick`` to run on synthetic in-memory datasets instead — useful
for verifying the pipeline without paying torchvision download time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from swarmtorch.benchmark import BenchmarkConfig
from swarmtorch.experiments import PAPER_ALGORITHMS, run_sweep
from swarmtorch.experiments.training import (
    make_cifar10_smallcnn_task,
    make_mnist_mlp_2layer_task,
    make_mnist_mlp_4layer_task,
    make_tabular_regression_task,
    synthetic_image_dataset,
    synthetic_tabular_dataset,
)


def _load_real_datasets():
    """Return (mnist_train, mnist_val, cifar_train, cifar_val) via torchvision."""
    from torchvision import datasets, transforms

    mnist_tx = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    mnist_tr = datasets.MNIST(root="data", train=True, download=True, transform=mnist_tx)
    mnist_val = datasets.MNIST(root="data", train=False, download=True, transform=mnist_tx)

    cifar_tx = transforms.ToTensor()
    cifar_tr = datasets.CIFAR10(root="data", train=True, download=True, transform=cifar_tx)
    cifar_val = datasets.CIFAR10(root="data", train=False, download=True, transform=cifar_tx)

    return mnist_tr, mnist_val, cifar_tr, cifar_val


def _quick_synthetic():
    mnist_tr = synthetic_image_dataset(n_samples=512, image_shape=(1, 28, 28), flat=True, seed=0)
    mnist_val = synthetic_image_dataset(n_samples=128, image_shape=(1, 28, 28), flat=True, seed=1)
    cifar_tr = synthetic_image_dataset(n_samples=512, image_shape=(3, 32, 32), flat=False, seed=2)
    cifar_val = synthetic_image_dataset(n_samples=128, image_shape=(3, 32, 32), flat=False, seed=3)
    return mnist_tr, mnist_val, cifar_tr, cifar_val


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/training"))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--max-fe", type=int, default=3000)
    p.add_argument("--swarm-size", type=int, default=30)
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=list(PAPER_ALGORITHMS),
    )
    p.add_argument("--quick", action="store_true", help="Use synthetic datasets only.")
    args = p.parse_args()

    mnist_tr, mnist_val, cifar_tr, cifar_val = (
        _quick_synthetic() if args.quick else _load_real_datasets()
    )
    tab_tr = synthetic_tabular_dataset(n_samples=2048, in_features=8, seed=10)
    tab_val = synthetic_tabular_dataset(n_samples=512, in_features=8, seed=11)

    tasks = [
        make_mnist_mlp_2layer_task(mnist_tr, mnist_val),
        make_mnist_mlp_4layer_task(mnist_tr, mnist_val),
        make_cifar10_smallcnn_task(cifar_tr, cifar_val),
        make_tabular_regression_task(tab_tr, tab_val),
    ]

    config = BenchmarkConfig(
        seeds=args.seeds,
        max_fe=args.max_fe,
        log_every=max(args.max_fe // 50, 1),
        output_dir=args.output_dir,
    )

    report = run_sweep(
        tasks=tasks,
        algorithm_names=args.algorithms,
        config=config,
        swarm_size=args.swarm_size,
        task_kind="training",
    )
    print(f"\nReport written to: {report}")


if __name__ == "__main__":
    main()
