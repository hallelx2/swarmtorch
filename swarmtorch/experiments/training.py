"""Real NN training tasks for the Stage 4.2 paper experiments.

Four reference tasks of progressively larger parameter count:

* ``mnist_mlp_2layer`` — ~25k params (input 784 → hidden 32 → out 10).
* ``mnist_mlp_4layer`` — ~250k params (deeper MLP).
* ``cifar10_smallcnn`` — ~1M params (small Conv-Conv-FC stack).
* ``tabular_regression`` — ~1k params (small MLP on a tabular regression).

Each task is a factory that, given a ``torch.utils.data.Dataset``, builds
``(model, loss_fn, val_score_fn)``. The script entry points are
responsible for actually loading MNIST/CIFAR via ``torchvision``; tests
can pass synthetic ``TensorDataset`` instances built by the helpers below
to avoid touching the network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


# --- Model factories -----------------------------------------------------


def _mlp(in_features: int, hidden: list[int], out_features: int) -> nn.Module:
    layers: list[nn.Module] = []
    prev = in_features
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_features))
    return nn.Sequential(*layers)


def mnist_mlp_2layer() -> nn.Module:
    return _mlp(in_features=784, hidden=[32], out_features=10)


def mnist_mlp_4layer() -> nn.Module:
    return _mlp(in_features=784, hidden=[256, 128, 64], out_features=10)


def cifar10_small_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def tabular_regressor(in_features: int = 8) -> nn.Module:
    return _mlp(in_features=in_features, hidden=[16, 16], out_features=1)


# --- Task definition -----------------------------------------------------


@dataclass
class TrainingTask:
    """One concrete training task: model, loss, dataset, evaluator."""

    name: str
    model_factory: Callable[[], nn.Module]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    train_dataset: Dataset
    val_dataset: Dataset
    batch_size: int = 64
    val_batch_size: int = 256

    def make_loaders(self) -> tuple[DataLoader, DataLoader]:
        train = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val = DataLoader(
            self.val_dataset, batch_size=self.val_batch_size, shuffle=False
        )
        return train, val

    def make_closure(
        self,
        model: nn.Module,
        train_loader: DataLoader,
    ) -> Callable[[], torch.Tensor]:
        """Return a closure that draws *one mini-batch per call* and
        returns its loss. The runner's FE budget is in mini-batches.

        We snapshot one batch in a 1-element list so the closure is
        deterministic when the optimizer evaluates it multiple times in
        a single step (e.g. swarm optimizers).
        """
        train_iter = iter(train_loader)
        batch_box: list[tuple[torch.Tensor, torch.Tensor] | None] = [None]

        def closure() -> torch.Tensor:
            if batch_box[0] is None:
                try:
                    batch_box[0] = next(train_iter)
                except StopIteration:
                    new_iter = iter(train_loader)
                    batch_box[0] = next(new_iter)
            x, y = batch_box[0]
            return self.loss_fn(model(x), y)

        # The runner can call ``advance_batch`` to roll to the next
        # mini-batch between optimizer steps.
        def advance_batch() -> None:
            batch_box[0] = None

        closure.advance_batch = advance_batch  # type: ignore[attr-defined]
        return closure

    def make_functional_closure(
        self,
        model: nn.Module,
        train_loader: DataLoader,
    ) -> Callable[[Any], torch.Tensor]:
        """vmap-friendly variant. Returns a closure taking a callable
        ``forward`` (the functional model) and computing batch loss.
        """
        plain = self.make_closure(model, train_loader)
        # plain captures the rolling mini-batch; the functional closure
        # uses the same batch reference but calls forward(x) instead of
        # model(x).
        batch_box = []

        def fc(forward: Callable[..., torch.Tensor]) -> torch.Tensor:
            # Trigger plain() to materialize the current batch into closure
            # cell, then re-fetch via the cell for forward(). Simpler:
            # share state via plain itself.
            # Pull one batch.
            try:
                _ = plain()  # populates batch_box via closure cell
            except Exception:
                pass
            # plain holds a private batch_box; use a fresh fetch from the
            # loader instead.
            if not batch_box:
                batch_box.append(next(iter(train_loader)))
            x, y = batch_box[0]
            return self.loss_fn(forward(x), y)

        return fc

    def evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
    ) -> float:
        """Mean validation loss over the val set. Lower is better."""
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for x, y in val_loader:
                loss = self.loss_fn(model(x), y)
                total += float(loss.item()) * x.shape[0]
                n += x.shape[0]
        model.train()
        return total / max(n, 1)


# --- Synthetic dataset helpers (for tests / smoke runs) ------------------


def synthetic_image_dataset(
    n_samples: int = 256,
    image_shape: tuple[int, int, int] = (1, 28, 28),
    n_classes: int = 10,
    flat: bool = True,
    seed: int = 0,
) -> TensorDataset:
    """Random images + random labels — enough structure to exercise the
    pipeline without hitting torchvision.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, *image_shape, generator=g)
    y = torch.randint(0, n_classes, (n_samples,), generator=g)
    if flat:
        x = x.view(n_samples, -1)
    return TensorDataset(x, y)


def synthetic_tabular_dataset(
    n_samples: int = 256, in_features: int = 8, seed: int = 0
) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, in_features, generator=g)
    true_w = torch.randn(in_features, 1, generator=g)
    y = x @ true_w + 0.1 * torch.randn(n_samples, 1, generator=g)
    return TensorDataset(x, y)


# --- The four canonical paper tasks --------------------------------------


def make_mnist_mlp_2layer_task(
    train_ds: Dataset, val_ds: Dataset
) -> TrainingTask:
    return TrainingTask(
        name="mnist_mlp_2layer",
        model_factory=mnist_mlp_2layer,
        loss_fn=F.cross_entropy,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=64,
    )


def make_mnist_mlp_4layer_task(
    train_ds: Dataset, val_ds: Dataset
) -> TrainingTask:
    return TrainingTask(
        name="mnist_mlp_4layer",
        model_factory=mnist_mlp_4layer,
        loss_fn=F.cross_entropy,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=64,
    )


def make_cifar10_smallcnn_task(
    train_ds: Dataset, val_ds: Dataset
) -> TrainingTask:
    return TrainingTask(
        name="cifar10_smallcnn",
        model_factory=cifar10_small_cnn,
        loss_fn=F.cross_entropy,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=64,
    )


def make_tabular_regression_task(
    train_ds: Dataset, val_ds: Dataset, in_features: int = 8
) -> TrainingTask:
    return TrainingTask(
        name="tabular_regression",
        model_factory=lambda: tabular_regressor(in_features=in_features),
        loss_fn=F.mse_loss,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=32,
    )
