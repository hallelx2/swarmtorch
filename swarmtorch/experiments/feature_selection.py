"""Combinatorial feature selection: a genuinely gradient-free application.

Selecting K of N features to maximize classification accuracy is a
combinatorial problem with a piecewise-constant objective -- there is no
gradient with respect to the include/exclude decisions, so gradient
methods are inapplicable by construction. The search space C(N, K) is far
too large to enumerate (for N=50, K=10 it is ~1e10).

A candidate is a real score vector s in R^N; the selected feature set is
the top-K indices of s. Fitness is validation error of a logistic-
regression classifier trained on the selected features (lower is better).
This is exactly the kind of black-box objective metaheuristics target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


@dataclass
class FeatureSelectionProblem:
    name: str
    n_features: int
    k: int
    Xtr: np.ndarray
    ytr: np.ndarray
    Xval: np.ndarray
    yval: np.ndarray

    def make_module(self) -> nn.Module:
        """A dummy module whose single parameter vector holds feature scores."""
        m = nn.Module()
        m.x = nn.Parameter(torch.rand(self.n_features))
        return m

    def evaluate(self, scores: torch.Tensor) -> float:
        """Validation error of a classifier on the top-K scored features."""
        idx = torch.topk(scores.detach().flatten(), self.k).indices.cpu().numpy()
        clf = LogisticRegression(max_iter=200)
        clf.fit(self.Xtr[:, idx], self.ytr)
        acc = clf.score(self.Xval[:, idx], self.yval)
        return float(1.0 - acc)


def make_feature_selection_problem(
    n_features: int = 50,
    n_informative: int = 10,
    k: int = 10,
    n_samples: int = 800,
    seed: int = 0,
) -> FeatureSelectionProblem:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,
        n_repeated=0,
        n_classes=2,
        random_state=seed,
        shuffle=True,
    )
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    return FeatureSelectionProblem(
        name=f"featsel_n{n_features}_k{k}",
        n_features=n_features,
        k=k,
        Xtr=scaler.transform(Xtr),
        ytr=ytr,
        Xval=scaler.transform(Xval),
        yval=yval,
    )
