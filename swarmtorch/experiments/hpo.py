"""Real HPO tasks for the Stage 4.3 paper experiments.

Each task is a tuple ``(model_fn, param_space, train_fn)`` accepted by
:class:`HyperparameterSearch` and the baselines in
``swarmtorch.baselines.hpo``. The XGBoost task is gated on the optional
``xgboost`` dependency so the package still imports without it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from swarmtorch.experiments.training import (
    cifar10_small_cnn,
    synthetic_image_dataset,
    synthetic_tabular_dataset,
)


@dataclass
class HPOTask:
    name: str
    model_fn: Callable[[dict], Any]
    param_space: dict[str, Any]
    train_fn: Callable[..., float]
    n_trials: int = 30
    meta: dict = field(default_factory=dict)


# --- 1. Small CNN on (a stand-in for) CIFAR-10 --------------------------


def make_cnn_hpo_task(
    train_ds: Dataset | None = None,
    val_ds: Dataset | None = None,
    epochs_per_trial: int = 1,
) -> HPOTask:
    """Tune lr, momentum, weight_decay, dropout for a small CNN.

    ``train_ds`` / ``val_ds`` default to a tiny synthetic image dataset
    so tests can run without torchvision; the real script wires up
    CIFAR-10.
    """
    train_ds = train_ds or synthetic_image_dataset(
        n_samples=128, image_shape=(3, 32, 32), n_classes=10, flat=False
    )
    val_ds = val_ds or synthetic_image_dataset(
        n_samples=64, image_shape=(3, 32, 32), n_classes=10, flat=False, seed=1
    )

    param_space: dict[str, Any] = {
        "lr": (1e-4, 1e-1),
        "momentum": (0.0, 0.99),
        "weight_decay": (1e-6, 1e-2),
        "dropout": (0.0, 0.5),
    }

    def model_fn(params: dict) -> nn.Module:
        # Insert dropout before the classifier head.
        model = cifar10_small_cnn()
        # Replace the last Linear with Dropout -> Linear.
        head = model[-1]
        assert isinstance(head, nn.Linear)
        model[-1] = nn.Sequential(nn.Dropout(p=params["dropout"]), head)
        return model

    def train_fn(model: nn.Module, params: dict, **kwargs: Any) -> float:
        report_callback = kwargs.get("report_callback")
        opt = torch.optim.SGD(
            model.parameters(),
            lr=params["lr"],
            momentum=params["momentum"],
            weight_decay=params["weight_decay"],
        )
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        for epoch in range(epochs_per_trial):
            model.train()
            for x, y in train_loader:
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
            score = _classification_val_loss(model, val_loader)
            if report_callback is not None:
                report_callback(score, epoch)
        return _classification_val_loss(model, val_loader)

    return HPOTask(
        name="cnn_hpo",
        model_fn=model_fn,
        param_space=param_space,
        train_fn=train_fn,
        n_trials=20,
    )


def _classification_val_loss(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            total += float(F.cross_entropy(model(x), y).item()) * x.shape[0]
            n += x.shape[0]
    return total / max(n, 1)


# --- 2. Tiny transformer on a toy text task -----------------------------


def make_tiny_transformer_hpo_task(
    train_ds: Dataset | None = None,
    val_ds: Dataset | None = None,
    epochs_per_trial: int = 1,
) -> HPOTask:
    """Tune lr, n_heads, n_layers, dim_ff for a small transformer.

    Treats characters of random binary strings as tokens — enough to
    exercise the pipeline and HPO searcher without bringing in HuggingFace.
    """
    vocab_size = 16
    seq_len = 24

    def _toy_text_dataset(n: int, seed: int = 0) -> Dataset:
        g = torch.Generator().manual_seed(seed)
        x = torch.randint(0, vocab_size, (n, seq_len), generator=g)
        # Target: parity of the first 4 tokens — 0/1 binary classification.
        y = (x[:, :4].sum(dim=1) % 2).long()
        from torch.utils.data import TensorDataset

        return TensorDataset(x, y)

    train_ds = train_ds or _toy_text_dataset(256, seed=0)
    val_ds = val_ds or _toy_text_dataset(128, seed=1)

    param_space: dict[str, Any] = {
        "lr": (1e-4, 1e-2),
        "n_heads": [1, 2, 4],
        "n_layers": [1, 2, 3],
        "dim_ff": [32, 64, 128],
    }

    def model_fn(params: dict) -> nn.Module:
        d_model = max(8, int(params["n_heads"]) * 8)
        return _TinyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=int(params["n_heads"]),
            n_layers=int(params["n_layers"]),
            dim_ff=int(params["dim_ff"]),
            n_classes=2,
        )

    def train_fn(model: nn.Module, params: dict, **kwargs: Any) -> float:
        report_callback = kwargs.get("report_callback")
        opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        for epoch in range(epochs_per_trial):
            model.train()
            for x, y in train_loader:
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
            score = _classification_val_loss(model, val_loader)
            if report_callback is not None:
                report_callback(score, epoch)
        return _classification_val_loss(model, val_loader)

    return HPOTask(
        name="tiny_transformer_hpo",
        model_fn=model_fn,
        param_space=param_space,
        train_fn=train_fn,
        n_trials=20,
    )


class _TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_ff: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.embed(x))
        return self.head(h.mean(dim=1))


# --- 3. XGBoost on a tabular task (optional) ----------------------------


def make_xgboost_hpo_task(
    train_ds: Dataset | None = None,
    val_ds: Dataset | None = None,
    in_features: int = 8,
) -> HPOTask:
    """Tune n_estimators, max_depth, eta, subsample for XGBoost regression.

    Stress-tests the *categorical* encoding (n_estimators, max_depth) +
    continuous (eta, subsample) mix in HPO methods.

    If ``xgboost`` is not installed, raises ``ImportError`` only when the
    user tries to construct this task — the rest of the experiments
    package still imports cleanly.
    """
    try:
        import xgboost  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without xgb
        raise ImportError(
            "XGBoost HPO task requires xgboost. Install with: pip install xgboost"
        ) from e

    train_ds = train_ds or synthetic_tabular_dataset(
        n_samples=512, in_features=in_features, seed=0
    )
    val_ds = val_ds or synthetic_tabular_dataset(
        n_samples=128, in_features=in_features, seed=1
    )

    param_space: dict[str, Any] = {
        "n_estimators": [50, 100, 200, 400],
        "max_depth": [3, 4, 5, 6, 7, 8, 10],
        "eta": (0.01, 0.5),
        "subsample": (0.5, 1.0),
    }

    def model_fn(params: dict) -> Any:
        import xgboost as xgb

        return xgb.XGBRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["eta"]),
            subsample=float(params["subsample"]),
            verbosity=0,
        )

    def train_fn(model: Any, params: dict, **kwargs: Any) -> float:
        # Materialize the tensor datasets once.
        x_tr = torch.cat([t for t, _ in train_ds]).reshape(len(train_ds), -1).numpy()
        y_tr = (
            torch.cat([y for _, y in train_ds]).reshape(len(train_ds), -1).numpy()
        )
        x_val = torch.cat([t for t, _ in val_ds]).reshape(len(val_ds), -1).numpy()
        y_val = (
            torch.cat([y for _, y in val_ds]).reshape(len(val_ds), -1).numpy()
        )
        model.fit(x_tr, y_tr.ravel())
        pred = model.predict(x_val)
        mse = float(((pred - y_val.ravel()) ** 2).mean())
        return mse

    return HPOTask(
        name="xgboost_hpo",
        model_fn=model_fn,
        param_space=param_space,
        train_fn=train_fn,
        n_trials=20,
    )
