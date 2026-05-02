"""Batched-game interface: N games as (N, ...) tensors.

Companion to `Game` in `base.py`. A `BatchedGame` mirrors that interface but
operates on N games simultaneously, with state stored in torch tensors that
share a leading batch dim. Implementations are expected to be GPU-friendly
and (eventually) torch.compile-able — meaning state is pure tensors, no
Python-side per-game branching.

The flat action-index space is the same as the corresponding single-state
`Game` so a `BatchedGame` is a drop-in substitute at the env-interface
boundary of self-play / MCTS-root.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BatchedGameState:
    """Marker base class. Subclasses are typically dataclasses holding
    `(N, …)` tensors that all share device + dtype-family.

    Required interface: `n` (batch size) and `device`. No other contract —
    each game adds its own fields (e.g. piece bitboards for chess).
    """

    @property
    def n(self) -> int:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        raise NotImplementedError


class BatchedGame(ABC):
    """Minimal batched-game interface.

    Sign convention for `step_batch` rewards matches `Game.step`: reward is
    from the perspective of the player who just moved (mover-POV).
    """

    board_size: tuple[int, int]
    action_space_size: int
    num_planes: int  # observation channels per ply (network sees num_planes * history_frames)

    @abstractmethod
    def reset_batch(
        self, n: int, *, device: str | torch.device = "cpu",
    ) -> BatchedGameState:
        """Return initial state for n games on `device`."""

    @abstractmethod
    def legal_mask(self, state: BatchedGameState) -> torch.Tensor:
        """Return `(N, action_space_size)` bool mask of legal actions."""

    @abstractmethod
    def step_batch(
        self,
        state: BatchedGameState,
        actions: torch.Tensor,
    ) -> tuple[BatchedGameState, torch.Tensor, torch.Tensor]:
        """Apply one action per game.

        Args:
            actions: `(N,)` int64 — must be legal at each game.

        Returns:
            (new_state, rewards `(N,)` float32, done `(N,)` bool).
            Rewards are mover-POV (matches `Game.step`).
        """

    @abstractmethod
    def to_tensor_batch(self, state: BatchedGameState) -> torch.Tensor:
        """Return observation tensor of shape `(N, num_planes, H, W)`."""
