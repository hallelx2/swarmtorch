"""Tests for HyperparameterSearch categorical encode/decode round-trip (Stage 1.6)."""

import torch
from torch import nn

from swarmtorch.base.hyperparam_search import HyperparameterSearch


class _Concrete(HyperparameterSearch):
    def search(self):
        return {}


def _make(param_space):
    return _Concrete(
        model_fn=lambda p: nn.Linear(2, 2),
        param_space=param_space,
        train_fn=lambda m, p: 0.0,
        iterations=1,
        swarm_size=1,
        device="cpu",
        verbose=False,
    )


def test_categorical_round_trip_first_value():
    s = _make({"opt": ["adam", "sgd", "rmsprop"]})
    encoded = s._encode_params({"opt": "adam"})
    decoded = s._decode_params(encoded)
    assert decoded["opt"] == "adam"


def test_categorical_round_trip_last_value():
    s = _make({"opt": ["adam", "sgd", "rmsprop"]})
    encoded = s._encode_params({"opt": "rmsprop"})
    decoded = s._decode_params(encoded)
    # Was previously off-by-one — encode put rmsprop at 2/3=0.667, decode then
    # mapped 0.667 * 2 = 1.33 -> int(1.33) = 1 ("sgd"). Should now round-trip.
    assert decoded["opt"] == "rmsprop"


def test_categorical_round_trip_middle_value():
    s = _make({"opt": ["a", "b", "c", "d", "e"]})
    for v in ["a", "b", "c", "d", "e"]:
        encoded = s._encode_params({"opt": v})
        decoded = s._decode_params(encoded)
        assert decoded["opt"] == v, f"failed round-trip on {v!r}"


def test_singleton_categorical():
    s = _make({"opt": ["only"]})
    encoded = s._encode_params({"opt": "only"})
    decoded = s._decode_params(encoded)
    assert decoded["opt"] == "only"


def test_continuous_round_trip():
    s = _make({"lr": (0.001, 0.1)})
    encoded = s._encode_params({"lr": 0.05})
    decoded = s._decode_params(encoded)
    assert abs(decoded["lr"] - 0.05) < 1e-6


def test_decode_clamps_out_of_range():
    s = _make({"opt": ["a", "b", "c"]})
    decoded = s._decode_params(torch.tensor([1.5]))
    assert decoded["opt"] == "c"
    decoded = s._decode_params(torch.tensor([-0.3]))
    assert decoded["opt"] == "a"
