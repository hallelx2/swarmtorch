"""Gradient-optimizer baselines exposed through the SwarmOptimizer-style API.

The benchmark harness treats every candidate as something with a
``step(closure)`` method that returns a loss tensor. PyTorch's gradient
optimizers already satisfy that contract, but they require a ``backward``
call before ``step``. This module adds the missing piece — a thin wrapper
that calls ``closure``, runs ``backward``, then ``step`` — so gradient
methods can be benchmarked alongside swarm/evolutionary optimizers under
identical FE-budget rules.

The wrapper also provides a no-op ``set_functional_closure`` so user code
that opts into the vmap fast path for swarmtorch optimizers can pass these
baselines through unchanged.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch.optim import SGD, Adam, AdamW, LBFGS, Optimizer, RMSprop


class GradientBaseline:
    """Adapter that exposes a ``torch.optim.Optimizer`` through the
    swarmtorch closure-style API.

    Args:
        params: Parameters to optimize.
        optimizer_cls: A subclass of ``torch.optim.Optimizer``.
        device: Device tag (kept for symmetry with ``SwarmOptimizer``).
        **opt_kwargs: Forwarded to ``optimizer_cls``.

    Example:
        >>> opt = AdamBaseline(model.parameters(), lr=1e-3)
        >>> for _ in range(epochs):
        ...     def closure():
        ...         return loss_fn(model(x), y)
        ...     opt.step(closure)
    """

    uses_gradients: bool = True

    def __init__(
        self,
        params: Any,
        optimizer_cls: type[Optimizer],
        device: str = "cpu",
        **opt_kwargs: Any,
    ) -> None:
        self.optimizer_cls = optimizer_cls
        self.opt_kwargs = opt_kwargs
        self.device = torch.device(device)
        self._opt: Optimizer = optimizer_cls(params, **opt_kwargs)

    @property
    def param_groups(self) -> list[dict]:
        return self._opt.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._opt.zero_grad(set_to_none=set_to_none)

    def set_functional_closure(self, *_args: Any, **_kwargs: Any) -> None:
        """No-op — gradient baselines do not use the vmap fast path."""
        return None

    def step(
        self, closure: Callable[[], torch.Tensor] | None = None
    ) -> torch.Tensor | None:
        """Compute gradients via ``closure`` and take one optimizer step.

        ``LBFGS`` natively requires the closure to be passed through to
        ``self._opt.step`` because it re-evaluates the loss multiple times
        per step; we forward to that path automatically.
        """
        if closure is None:
            return self._opt.step()

        if isinstance(self._opt, LBFGS):
            def _closure_with_backward() -> torch.Tensor:
                self._opt.zero_grad()
                loss = closure()
                loss.backward()
                return loss

            return self._opt.step(_closure_with_backward)

        self._opt.zero_grad()
        loss = closure()
        loss.backward()
        self._opt.step()
        return loss.detach()

    def state_dict(self) -> dict:
        return self._opt.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._opt.load_state_dict(state_dict)


class AdamBaseline(GradientBaseline):
    def __init__(self, params: Any, lr: float = 1e-3, device: str = "cpu", **kwargs: Any):
        super().__init__(params, Adam, device=device, lr=lr, **kwargs)


class AdamWBaseline(GradientBaseline):
    def __init__(self, params: Any, lr: float = 1e-3, device: str = "cpu", **kwargs: Any):
        super().__init__(params, AdamW, device=device, lr=lr, **kwargs)


class SGDBaseline(GradientBaseline):
    def __init__(
        self,
        params: Any,
        lr: float = 1e-2,
        momentum: float = 0.9,
        device: str = "cpu",
        **kwargs: Any,
    ):
        super().__init__(params, SGD, device=device, lr=lr, momentum=momentum, **kwargs)


class RMSpropBaseline(GradientBaseline):
    def __init__(self, params: Any, lr: float = 1e-3, device: str = "cpu", **kwargs: Any):
        super().__init__(params, RMSprop, device=device, lr=lr, **kwargs)


class LBFGSBaseline(GradientBaseline):
    def __init__(
        self,
        params: Any,
        lr: float = 1.0,
        max_iter: int = 20,
        device: str = "cpu",
        **kwargs: Any,
    ):
        super().__init__(
            params, LBFGS, device=device, lr=lr, max_iter=max_iter, **kwargs
        )
