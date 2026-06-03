"""Real-signal HPO tasks on sklearn datasets (CPU-only, no GPU/torchvision).

The synthetic HPO tasks in ``hpo.py`` train on random data, so their
objective landscape is essentially flat -- every hyperparameter setting
yields the same loss and all searchers tie trivially. That makes the
comparison underpowered.

These tasks use *real* datasets with genuine learnable signal, so good
hyperparameters measurably reduce validation loss and the searchers can
actually separate. All run on CPU in seconds, so the whole sweep is
fast and Friedman gets enough tasks (N) for real statistical power.

Objective is always validation log-loss (lower is better), consistent
with the minimization convention used everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_wine,
    make_classification,
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class HPOTask:
    name: str
    model_fn: Any
    param_space: dict
    train_fn: Any
    n_trials: int = 20
    meta: dict = field(default_factory=dict)


def _split(X, y, seed: int = 0):
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xval), ytr, yval


def _make_mlp_task(name: str, loader) -> HPOTask:
    """MLP classifier HPO on a real classification dataset."""
    data = loader()
    X, y = data.data, data.target
    Xtr, Xval, ytr, yval = _split(X, y)
    n_classes = len(np.unique(y))

    param_space = {
        "lr": (1e-4, 1e-1),                 # log-ish learning rate
        "hidden": [16, 32, 64, 128],        # categorical width
        "alpha": (1e-6, 1e-1),              # L2 regularization
        "n_layers": [1, 2, 3],              # categorical depth
    }

    def model_fn(params: dict):
        hidden = int(params["hidden"])
        n_layers = int(params["n_layers"])
        return MLPClassifier(
            hidden_layer_sizes=tuple([hidden] * n_layers),
            learning_rate_init=float(params["lr"]),
            alpha=float(params["alpha"]),
            max_iter=120,
            early_stopping=False,
            random_state=0,
        )

    def train_fn(model, params, **kwargs):
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xval)
        return float(log_loss(yval, proba, labels=list(range(n_classes))))

    return HPOTask(name=name, model_fn=model_fn, param_space=param_space, train_fn=train_fn)


def _make_gb_task() -> HPOTask:
    """Gradient-boosting HPO on digits -- mixed categorical + continuous space."""
    data = load_digits()
    X, y = data.data, data.target
    Xtr, Xval, ytr, yval = _split(X, y)
    n_classes = len(np.unique(y))

    param_space = {
        "n_estimators": [40, 80, 120],
        "max_depth": [2, 3, 4],
        "learning_rate": (0.01, 0.5),
        "subsample": (0.5, 1.0),
    }

    def model_fn(params: dict):
        return GradientBoostingClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            random_state=0,
        )

    def train_fn(model, params, **kwargs):
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xval)
        return float(log_loss(yval, proba, labels=list(range(n_classes))))

    return HPOTask(name="gb_digits", model_fn=model_fn, param_space=param_space, train_fn=train_fn)


def _make_rf_task() -> HPOTask:
    """Random-forest HPO on a synthetic-but-structured classification set."""
    X, y = make_classification(
        n_samples=1500, n_features=30, n_informative=12, n_redundant=6,
        n_classes=3, n_clusters_per_class=2, random_state=0,
    )
    Xtr, Xval, ytr, yval = _split(X, y)
    n_classes = len(np.unique(y))

    param_space = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 8, 12],
        "max_features": (0.1, 1.0),
        "min_samples_leaf": [1, 2, 4, 8],
    }

    def model_fn(params: dict):
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            max_features=float(params["max_features"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=0,
            n_jobs=1,
        )

    def train_fn(model, params, **kwargs):
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xval)
        return float(log_loss(yval, proba, labels=list(range(n_classes))))

    return HPOTask(name="rf_synth", model_fn=model_fn, param_space=param_space, train_fn=train_fn)


def make_real_hpo_tasks() -> list[HPOTask]:
    """Five real-signal HPO tasks across model families and space types."""
    return [
        _make_mlp_task("mlp_digits", load_digits),
        _make_mlp_task("mlp_breast_cancer", load_breast_cancer),
        _make_mlp_task("mlp_wine", load_wine),
        _make_gb_task(),
        _make_rf_task(),
    ]
