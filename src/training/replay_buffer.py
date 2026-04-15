"""Replay buffer for MuZero training."""

import random
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class GameHistory:
    """Stores a complete game trajectory."""
    observations: list[torch.Tensor] = field(default_factory=list)  # (C, H, W) tensors
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    policies: list[list[float]] = field(default_factory=list)  # MCTS visit count distributions
    root_values: list[float] = field(default_factory=list)  # MCTS root values
    game_outcome: float = 0.0  # final game outcome from player 1's perspective
    game_name: str = ""  # for multi-game training

    def __len__(self) -> int:
        return len(self.observations)

    def make_target(self, state_index: int, num_unroll_steps: int,
                    td_steps: int, discount: float, action_space_size: int):
        """Create training target for a given position.

        Returns:
            target_observations: observation at state_index
            target_actions: K future actions (padded with 0 if game ended)
            target_values: K+1 values (bootstrapped or game outcome)
            target_rewards: K+1 rewards
            target_policies: K+1 policy targets
        """
        # Value target: use game outcome (td_steps=-1) or n-step TD
        values = []
        rewards = []
        policies = []
        actions = []

        for i in range(num_unroll_steps + 1):
            idx = state_index + i

            if idx < len(self):
                # Compute value target
                if td_steps == -1:
                    # Use final game outcome, adjusted for player perspective
                    # At even positions, it's from player 1's perspective
                    # At odd positions, negate
                    sign = 1.0 if (idx % 2 == 0) else -1.0
                    value = sign * self.game_outcome
                else:
                    # N-step TD bootstrap
                    bootstrap_idx = idx + td_steps
                    value = 0.0
                    for j in range(idx, min(bootstrap_idx, len(self))):
                        r = self.rewards[j] if j < len(self.rewards) else 0.0
                        value += (discount ** (j - idx)) * r
                    if bootstrap_idx < len(self):
                        sign = 1.0 if ((bootstrap_idx % 2) == (idx % 2)) else -1.0
                        value += (discount ** td_steps) * sign * self.root_values[bootstrap_idx]

                values.append(value)
                rewards.append(self.rewards[idx] if idx < len(self.rewards) else 0.0)

                # Pad policy to action_space_size
                policy = self.policies[idx] if idx < len(self.policies) else []
                if len(policy) < action_space_size:
                    policy = policy + [0.0] * (action_space_size - len(policy))
                policies.append(policy[:action_space_size])
            else:
                # Past end of game: absorbing state
                values.append(0.0)
                rewards.append(0.0)
                policies.append([1.0 / action_space_size] * action_space_size)

            # Action leading to next state
            if i < num_unroll_steps:
                act_idx = state_index + i
                if act_idx < len(self.actions):
                    actions.append(self.actions[act_idx])
                else:
                    actions.append(0)  # padding action

        return (
            self.observations[state_index],
            actions,
            values,
            rewards,
            policies,
        )


class ReplayBuffer:
    """Fixed-size replay buffer storing complete game histories."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: list[GameHistory] = []
        self.total_games = 0

    def save_game(self, game_history: GameHistory):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(game_history)
        self.total_games += 1

    def sample_batch(
        self,
        batch_size: int,
        num_unroll_steps: int,
        td_steps: int,
        discount: float,
        action_space_size: int,
    ) -> dict:
        """Sample a batch of positions from stored games.

        Returns dict with batched tensors ready for training.
        """
        observations = []
        actions_batch = []
        values_batch = []
        rewards_batch = []
        policies_batch = []

        for _ in range(batch_size):
            game = random.choice(self.buffer)
            pos = random.randint(0, len(game) - 1)

            obs, actions, values, rewards, policies = game.make_target(
                pos, num_unroll_steps, td_steps, discount, action_space_size
            )

            observations.append(obs)
            actions_batch.append(actions)
            values_batch.append(values)
            rewards_batch.append(rewards)
            policies_batch.append(policies)

        return {
            "observations": torch.stack(observations),
            "actions": torch.tensor(actions_batch, dtype=torch.long),
            "target_values": torch.tensor(values_batch, dtype=torch.float32),
            "target_rewards": torch.tensor(rewards_batch, dtype=torch.float32),
            "target_policies": torch.tensor(policies_batch, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.buffer)
