"""Abstract base class for all games."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class GameState:
    """Generic game state container."""
    board: Any  # Game-specific board representation
    current_player: int  # 1 or -1
    done: bool = False
    winner: int = 0  # 1, -1, or 0 (draw)


class Game(ABC):
    """Minimal interface all games must implement."""

    board_size: tuple[int, int]  # (H, W)
    action_space_size: int  # max possible actions
    num_planes: int  # observation tensor channels

    @abstractmethod
    def reset(self) -> GameState:
        """Return initial game state."""

    @abstractmethod
    def step(self, state: GameState, action: int) -> tuple[GameState, float, bool]:
        """Apply action, return (new_state, reward, done).

        Reward is from the perspective of the player who just moved.
        """

    @abstractmethod
    def legal_actions(self, state: GameState) -> list[int]:
        """Return list of legal action indices."""

    @abstractmethod
    def to_tensor(self, state: GameState) -> torch.Tensor:
        """Convert state to observation tensor of shape (C, H, W)."""

    @abstractmethod
    def render(self, state: GameState) -> str:
        """Return text display of current state."""

    def clone_state(self, state: GameState) -> GameState:
        """Deep copy a game state. Override for complex states."""
        import copy
        return copy.deepcopy(state)
