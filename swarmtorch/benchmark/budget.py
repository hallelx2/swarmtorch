"""Function-evaluation (FE) budget enforcement.

Every fair benchmark of optimization algorithms must equalize the number
of times the loss function is evaluated. PSO with swarm=30 doing 100
iterations evaluates the loss 3000 times; comparing it to Adam taking 100
gradient steps (which only evaluate the loss 100 times) is meaningless.

This module provides :class:`FEBudgetTracker`, which counts every
particle evaluation that flows through a closure or functional closure.
When the running count exceeds the configured budget, it sets a ``done``
flag that the runner checks between optimizer steps. We don't raise
mid-step because that would leave PyTorch's vmap machinery in an
awkward state; the runner reads ``tracker.done`` after each ``step()``
and stops cleanly.
"""

from collections.abc import Callable

import torch


class BudgetExceeded(RuntimeError):
    """Raised explicitly by ``raise_if_done`` when callers want a hard stop."""


class FEBudgetTracker:
    """Counts function evaluations against a budget.

    Args:
        max_fe: Maximum number of single-particle function evaluations
            allowed for the run. Each plain closure call contributes 1;
            each vmap'd functional closure call contributes one per
            particle in the batch.
        log_every: If set, record (fe_count, best_so_far) pairs into
            ``self.trajectory`` every ``log_every`` evaluations. The
            runner uses this to draw convergence curves.
    """

    def __init__(self, max_fe: int, log_every: int | None = None) -> None:
        if max_fe <= 0:
            raise ValueError(f"max_fe must be > 0, got {max_fe}")
        self.max_fe = int(max_fe)
        self.log_every = log_every
        self.fe_count: int = 0
        self.best_so_far: float = float("inf")
        self.trajectory: list[tuple[int, float]] = []
        self._next_log_at: int = log_every if log_every else None  # type: ignore[assignment]

    @property
    def done(self) -> bool:
        return self.fe_count >= self.max_fe

    @property
    def remaining(self) -> int:
        return max(0, self.max_fe - self.fe_count)

    def raise_if_done(self) -> None:
        if self.done:
            raise BudgetExceeded(
                f"FE budget {self.max_fe} exhausted (used {self.fe_count})."
            )

    def _observe(self, n: int, scores: torch.Tensor | None) -> None:
        self.fe_count += int(n)
        if scores is not None and scores.numel() > 0:
            current = float(scores.detach().min().item())
            if current < self.best_so_far:
                self.best_so_far = current
        if self.log_every is not None and self._next_log_at is not None:
            while self.fe_count >= self._next_log_at:
                self.trajectory.append((self._next_log_at, self.best_so_far))
                self._next_log_at += self.log_every

    def wrap_closure(
        self,
        closure: Callable[[], torch.Tensor],
    ) -> Callable[[], torch.Tensor]:
        """Wrap a plain closure so each call increments the counter by 1."""

        def wrapped() -> torch.Tensor:
            loss = closure()
            self._observe(1, loss.detach().reshape(-1))
            return loss

        return wrapped

    def wrap_functional_closure(
        self,
        functional_closure: Callable[[torch.Tensor], torch.Tensor],
        n_particles: int,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Wrap a vmap-style functional closure.

        ``n_particles`` is the swarm/population size — one outer call to
        ``vmap(functional_closure)(particles)`` evaluates the loss for
        every particle in the batch, so we credit ``n_particles``
        evaluations per call.

        We can't wrap the per-particle ``functional_closure`` directly
        because vmap traces it once and reuses the trace; the counter
        increment must happen at the outer-call level. So we return a
        thin shim that the runner inserts at that level.
        """

        def wrapped_outer(particles: torch.Tensor) -> torch.Tensor:
            scores = torch.vmap(functional_closure)(particles)
            self._observe(int(particles.shape[0]), scores)
            return scores

        return wrapped_outer

    def observe_external(self, n: int, current_best: float | None = None) -> None:
        """Manual hook for paths that don't go through wrap_closure
        (e.g. gradient baselines that call ``loss.backward()`` once per
        step). Add ``n`` evaluations and optionally update best-so-far.
        """
        scores = (
            torch.tensor([current_best])
            if current_best is not None
            else None
        )
        self._observe(n, scores)
