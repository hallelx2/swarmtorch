from collections.abc import Callable
from typing import Any

import torch
from torch.optim import Optimizer


class SwarmOptimizer(Optimizer):
    """Base class for swarm-based optimization algorithms.

    Inherits from torch.optim.Optimizer to leverage parameter group management,
    state dict, and compatibility with PyTorch ecosystem.

    Note: These optimizers are gradient-free - they use loss values as fitness
    rather than gradients to update weights.
    """

    uses_gradients: bool = False

    def __init__(
        self,
        params: Any,
        swarm_size: int = 30,
        device: str = "cpu",
        init_strategy: str = "model",
        init_sigma: float = 0.1,
        **kwargs: Any,
    ) -> None:
        if init_strategy not in {"model", "uniform", "gaussian"}:
            raise ValueError(
                f"init_strategy must be one of 'model', 'uniform', 'gaussian'; "
                f"got {init_strategy!r}"
            )
        defaults = dict(
            swarm_size=swarm_size,
            device=device,
            init_strategy=init_strategy,
            init_sigma=init_sigma,
            **kwargs,
        )
        super().__init__(params, defaults)
        self._initialized = False
        self._swarm_initialized = False
        self.device = torch.device(device)
        self.init_strategy = init_strategy
        self.init_sigma = init_sigma

    def _init_positions(self, d: int) -> torch.Tensor:
        """Sample initial swarm positions of shape (swarm_size, d).

        The default ``init_strategy="model"`` seeds the swarm around the
        model's current weights — i.e. whatever Kaiming/Xavier initialization
        ``torch.nn`` already applied — with a small Gaussian perturbation
        scaled by the existing weight std. ``positions[0]`` is set to the
        unmodified model weights, so the swarm is guaranteed to contain the
        user's starting point (CMA-ES style).

        ``"uniform"`` reproduces the legacy ``torch.rand`` behavior in
        ``[0, 1)`` for backwards reproducibility of older experiments.
        ``"gaussian"`` draws zero-mean Gaussian samples with std
        ``init_sigma``.
        """
        n = getattr(
            self, "swarm_size", getattr(self, "population_size", None)
        ) or self.defaults["swarm_size"]
        device = self.device
        if self.init_strategy == "uniform":
            return torch.rand(n, d, device=device)
        if self.init_strategy == "gaussian":
            return torch.randn(n, d, device=device) * self.init_sigma
        # "model"
        init_params = self._get_params().to(device)
        scale = init_params.std().clamp(min=1e-3) * self.init_sigma
        positions = init_params.unsqueeze(0).expand(n, d).clone()
        positions = positions + torch.randn(n, d, device=device) * scale
        positions[0] = init_params
        return positions

    def _get_param_shape(self) -> torch.Size:
        """Get the shape of flattened parameters."""
        total_params = 0
        for group in self.param_groups:
            for p in group["params"]:
                total_params += p.numel()
        return torch.Size([total_params])

    def _set_params(self, flat_params: torch.Tensor) -> None:
        """Set model parameters from a flattened tensor.

        Args:
            flat_params: 1-D tensor containing all parameter values.
        """
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                p.data.copy_(flat_params[idx : idx + numel].reshape(p.shape))
                idx += numel

    def _get_params(self) -> torch.Tensor:
        """Get all model parameters as a single flattened tensor.

        Returns:
            1-D tensor of concatenated, flattened parameters.
        """
        return torch.cat(
            [p.data.flatten() for group in self.param_groups for p in group["params"]]
        )

    def _evaluate_fitness(
        self,
        particles: torch.Tensor,
        closure: Any = None,
    ) -> torch.Tensor:
        """Evaluate fitness for every particle.

        Three evaluation paths, in priority order:

        1. **Functional vmap path** — set when the user calls
           :meth:`set_functional_closure` with a model and a loss function.
           All ``n`` particles are evaluated in a single vectorized forward
           pass via :func:`torch.func.functional_call` + :func:`torch.vmap`.
           This is the fast path and is what makes ``swarmtorch`` actually
           benefit from being on top of PyTorch.
        2. **Legacy loop path** — falls back to the historical
           per-particle ``_set_params`` + ``closure()`` loop when only a
           plain closure is available. Slower (one forward pass per
           particle) but compatible with closures that mutate global state.
        3. **Error** — raises if neither is configured.

        Args:
            particles: 2-D tensor of shape ``(n, d)``.
            closure: Optional plain closure (used only by the legacy path).

        Returns:
            1-D tensor of fitness values, one per particle.
        """
        fc = getattr(self, "_functional_closure", None)
        if fc is not None:
            with torch.no_grad():
                return torch.vmap(fc)(particles).detach()

        if closure is None:
            raise ValueError(
                f"{type(self).__name__} requires either a functional closure "
                "(via set_functional_closure) or a plain closure to evaluate "
                "fitness."
            )
        fitness = torch.zeros(particles.shape[0], device=self.device)
        for i in range(particles.shape[0]):
            self._set_params(particles[i])
            fitness[i] = closure().detach()
        return fitness

    def set_functional_closure(
        self,
        model: torch.nn.Module | None,
        loss_fn: Callable[[Callable[..., torch.Tensor]], torch.Tensor] | None,
    ) -> None:
        """Enable vectorized fitness evaluation for ``model``.

        Once configured, every call to :meth:`step` evaluates all
        particles in a single batched forward pass via
        :func:`torch.func.functional_call` + :func:`torch.vmap`, instead
        of the per-particle Python loop. This is the fast path that
        makes ``swarmtorch`` actually benefit from being on PyTorch.

        ``loss_fn`` is given a *callable model* — call it just like the
        original ``model`` to get its output for the current particle's
        weights — and must return a scalar loss::

            def loss_fn(forward):
                return F.cross_entropy(forward(x), y)

            optimizer.set_functional_closure(model, loss_fn)

        Refresh ``loss_fn`` between batches by calling this method
        again. Pass ``None`` for either argument to disable the fast
        path and fall back to the legacy per-particle loop.
        """
        if model is None or loss_fn is None:
            self._functional_closure = None
            self._functional_model = None
            return

        from torch.func import functional_call

        param_specs: list[tuple[str, torch.Size, int]] = [
            (name, p.shape, p.numel()) for name, p in model.named_parameters()
        ]
        buffers = dict(model.named_buffers())

        def _functional_closure(flat_params: torch.Tensor) -> torch.Tensor:
            params: dict[str, torch.Tensor] = {}
            idx = 0
            for name, shape, numel in param_specs:
                params[name] = flat_params[idx : idx + numel].view(shape)
                idx += numel
            full_state = {**params, **buffers}

            def forward(*args, **kwargs):
                return functional_call(model, full_state, args, kwargs)

            return loss_fn(forward)

        self._functional_closure = _functional_closure
        self._functional_model = model

    def _init_swarm(self) -> None:
        """Initialize swarm particles. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _init_swarm")

    def _update_positions(self) -> None:
        """Update particle positions. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _update_positions")

    def step(self, closure: Any = None) -> Any:
        """Perform one optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                Required for swarm optimizers since they need loss values.

        Returns:
            The best fitness observed so far (a 0-D tensor) if available,
            otherwise ``None``. The closure is *not* invoked an extra time
            here — ``_update_positions`` evaluates it once per particle.
        """
        # Stash the closure so subclasses can pull it from
        # ``self._current_closure`` inside ``_update_positions``.
        # If the user has registered a functional closure (vmap fast path)
        # but didn't supply a plain closure, install a passthrough
        # sentinel so the existing ``if closure is None: return`` guards
        # in subclass ``_update_positions`` don't short-circuit. The
        # actual fitness evaluation still goes through the vmap path
        # inside ``_evaluate_fitness``.
        if closure is None and getattr(self, "_functional_closure", None) is not None:
            self._current_closure = self._functional_closure
        else:
            self._current_closure = closure

        if not self._swarm_initialized:
            self._init_swarm()
            self._swarm_initialized = True

        self._update_positions()

        for attr in ("global_best_fitness", "best_fitness", "alpha_fitness"):
            value = getattr(self, attr, None)
            if value is not None:
                return value
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero out gradients for all parameter groups."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def _swarm_state(self) -> dict:
        """Return swarm-specific tensors that should be persisted.

        Subclasses override to list their own state (``positions``,
        ``velocities``, ``personal_best_*``, ``alpha_position``, etc.). The
        default implementation captures the common tensors used by most
        subclasses so that state-dict round-trips work without requiring
        every subclass to override.
        """
        keys = (
            "positions",
            "velocities",
            "fitness",
            "personal_best_positions",
            "personal_best_fitness",
            "global_best_position",
            "global_best_fitness",
            "best_position",
            "best_fitness",
            "alpha_position",
            "alpha_fitness",
            "beta_position",
            "delta_position",
            "population",
            "iteration_count",
            "mean",
            "std",
            "ant_lions",
            "flames",
            "flags",
            "samples",
        )
        out: dict = {}
        for k in keys:
            if hasattr(self, k):
                out[k] = getattr(self, k)
        return out

    def _load_swarm_state(self, state: dict) -> None:
        """Restore swarm-specific tensors saved by ``_swarm_state``."""
        for k, v in state.items():
            setattr(self, k, v)

    def state_dict(self) -> dict:
        """Return the state of the optimizer as a dict."""
        state = super().state_dict()
        state["_swarm_initialized"] = self._swarm_initialized
        state["_swarm_state"] = self._swarm_state()
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the optimizer state."""
        self._swarm_initialized = state_dict.pop("_swarm_initialized", False)
        swarm_state = state_dict.pop("_swarm_state", None)
        super().load_state_dict(state_dict)
        if swarm_state:
            self._load_swarm_state(swarm_state)
