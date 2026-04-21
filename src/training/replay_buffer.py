"""Replay buffer for MuZero training."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class GameHistory:
    """Stores a complete game trajectory."""
    observations: list[torch.Tensor] = field(default_factory=list)  # (C, H, W) tensors
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)  # MCTS visit count distributions (fp32)
    root_values: list[float] = field(default_factory=list)  # MCTS root values
    legal_actions_list: list[list[int]] = field(default_factory=list)  # legal actions at each step
    game_outcome: float = 0.0  # final game outcome from player 1's perspective
    game_name: str = ""  # for multi-game training
    # Per-step value targets from an external source (e.g., Stockfish eval), already
    # converted to side-to-move POV in [-1, +1]. Empty for self-play games (default).
    # When populated, make_target prefers these over the td_steps/game_outcome paths.
    external_values: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.observations)

    def make_target(self, state_index: int, num_unroll_steps: int,
                    td_steps: int, discount: float, action_space_size: int):
        """Create training target for a given position.

        Returns:
            target_observations: list of K+1 obs at state_index..state_index+K
                (past game end: last valid obs is reused; the mask zeros the loss)
            target_obs_mask: list of K+1 floats — 1.0 where the index is within
                the game, 0.0 past game end. Used by EfficientZero consistency loss.
            target_actions: K future actions (padded with 0 if game ended)
            target_values: K+1 values (bootstrapped or game outcome)
            target_rewards: K+1 rewards
            target_policies: K+1 policy targets
        """
        values = []
        rewards = []
        policies = []
        actions = []
        obs_list = []
        obs_mask = []
        last_obs = self.observations[min(state_index, len(self) - 1)]

        for i in range(num_unroll_steps + 1):
            idx = state_index + i

            if idx < len(self):
                if self.external_values and idx < len(self.external_values):
                    # Warmstart path: external targets are already side-to-move-relative.
                    value = self.external_values[idx]
                elif td_steps == -1:
                    sign = 1.0 if (idx % 2 == 0) else -1.0
                    value = sign * self.game_outcome
                else:
                    bootstrap_idx = idx + td_steps
                    value = 0.0
                    for j in range(idx, min(bootstrap_idx, len(self))):
                        r = self.rewards[j] if j < len(self.rewards) else 0.0
                        value += (discount ** (j - idx)) * r
                    # root_values has one fewer entry than observations (no root_value
                    # for the terminal observation), so guard against that off-by-one.
                    if bootstrap_idx < len(self.root_values):
                        sign = 1.0 if ((bootstrap_idx % 2) == (idx % 2)) else -1.0
                        value += (discount ** td_steps) * sign * self.root_values[bootstrap_idx]

                values.append(value)
                rewards.append(self.rewards[idx] if idx < len(self.rewards) else 0.0)

                policy = self.policies[idx] if idx < len(self.policies) else None
                if policy is None or len(policy) == 0:
                    policy = np.zeros(action_space_size, dtype=np.float32)
                else:
                    policy = np.asarray(policy, dtype=np.float32)
                    if len(policy) < action_space_size:
                        padded = np.zeros(action_space_size, dtype=np.float32)
                        padded[: len(policy)] = policy
                        policy = padded
                    elif len(policy) > action_space_size:
                        policy = policy[:action_space_size]
                policies.append(policy)
                last_obs = self.observations[idx]
                obs_list.append(last_obs)
                obs_mask.append(1.0)
            else:
                values.append(0.0)
                rewards.append(0.0)
                policies.append(np.full(action_space_size, 1.0 / action_space_size, dtype=np.float32))
                obs_list.append(last_obs)
                obs_mask.append(0.0)

            if i < num_unroll_steps:
                act_idx = state_index + i
                if act_idx < len(self.actions):
                    actions.append(self.actions[act_idx])
                else:
                    actions.append(0)

        return (
            obs_list,
            obs_mask,
            actions,
            values,
            rewards,
            policies,
        )


class ReplayBuffer:
    """Fixed-size replay buffer with Prioritized Experience Replay (PER).

    Games are sampled proportional to priority^alpha. Importance sampling
    weights correct for the sampling bias introduced by prioritization.
    Priorities are updated after each training step using TD errors.

    With alpha=0 or beta=1, degrades gracefully to uniform sampling.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: list[GameHistory] = []
        self._priorities: list[float] = []
        self.total_games = 0

    def save_game(self, game_history: GameHistory):
        # New games get max current priority so they're sampled at least once
        max_p = max(self._priorities) if self._priorities else 1.0
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
            self._priorities.pop(0)
        self.buffer.append(game_history)
        self._priorities.append(max_p)
        self.total_games += 1

    def sample_batch(
        self,
        batch_size: int,
        num_unroll_steps: int,
        td_steps: int,
        discount: float,
        action_space_size: int,
        alpha: float = 0.0,
        beta: float = 1.0,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Sample a batch of positions from stored games using PER.

        Returns:
            batch: dict of batched tensors
            game_indices: indices into self.buffer for priority updates
            is_weights: importance sampling weights to correct for PER bias
        """
        n = len(self.buffer)
        priorities = np.array(self._priorities[:n], dtype=np.float64)
        # Repair any non-finite priorities from past NaN/Inf TD errors.
        priorities = np.where(np.isfinite(priorities) & (priorities > 0), priorities, 1.0)
        probs = priorities ** alpha
        probs /= probs.sum()

        game_indices = np.random.choice(n, size=batch_size, p=probs)

        # Importance sampling weights: w_i = (1 / (N * P(i)))^beta, normalised
        weights = (n * probs[game_indices]) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        target_observations = []
        target_obs_masks = []
        actions_batch = []
        values_batch = []
        rewards_batch = []
        policies_batch = []

        for g_idx in game_indices:
            game = self.buffer[g_idx]
            pos = np.random.randint(0, len(game))
            obs_list, obs_mask, actions, values, rewards, policies = game.make_target(
                pos, num_unroll_steps, td_steps, discount, action_space_size
            )
            target_observations.append(torch.stack(obs_list))  # (K+1, C, H, W)
            target_obs_masks.append(obs_mask)
            actions_batch.append(actions)
            values_batch.append(values)
            rewards_batch.append(rewards)
            policies_batch.append(policies)

        target_observations_t = torch.stack(target_observations)  # (B, K+1, C, H, W)

        batch = {
            # Root obs (k=0) kept at "observations" for back-compat with callers that
            # only need the initial inference input.
            "observations": target_observations_t[:, 0],
            "target_observations": target_observations_t,
            "target_obs_mask": torch.tensor(target_obs_masks, dtype=torch.float32),
            "actions": torch.tensor(actions_batch, dtype=torch.long),
            "target_values": torch.tensor(values_batch, dtype=torch.float32),
            "target_rewards": torch.tensor(rewards_batch, dtype=torch.float32),
            "target_policies": torch.from_numpy(np.stack(policies_batch)),
        }
        return batch, game_indices, weights

    def update_priorities(self, game_indices: np.ndarray, td_errors: np.ndarray, epsilon: float = 1e-6):
        """Update game priorities from TD errors computed during training."""
        for idx, err in zip(game_indices, td_errors):
            if 0 <= idx < len(self._priorities):
                err = float(abs(err))
                # Guard against NaN/Inf from divergent training steps poisoning the buffer.
                if not np.isfinite(err):
                    err = 1.0
                self._priorities[idx] = err + epsilon

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump({
                "buffer": self.buffer,
                "priorities": self._priorities,
                "total_games": self.total_games,
            }, f)

    def load(self, path: str | Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.buffer = data["buffer"]
        self._priorities = data["priorities"]
        self.total_games = data["total_games"]

    def load_warmstart_games(self, paths: list[Path | str]) -> int:
        """Load pickled shards of GameHistory objects into the buffer.

        Each shard is expected to be a pickle of list[GameHistory]. Games are
        inserted via save_game so priorities are seeded to the current max.
        Returns the number of games loaded.
        """
        loaded = 0
        for p in paths:
            with open(p, "rb") as f:
                shard = pickle.load(f)
            if not isinstance(shard, list):
                raise TypeError(f"{p}: expected list[GameHistory], got {type(shard)}")
            for g in shard:
                self.save_game(g)
                loaded += 1
        return loaded

    def sample_games_for_reanalyze(self, n: int) -> tuple[np.ndarray, list]:
        """Sample n games uniformly (ignoring PER priorities) for reanalyze.

        Uniform sampling avoids double-biasing: PER already up-weights high-error
        positions during training; using it here too would over-concentrate reanalyze
        on the same games. See muzero-general replay_buffer.py (force_uniform=True).

        Warmstart games (``external_values`` populated) are excluded: their policy
        targets are supervised one-hots from Stockfish, and their value targets come
        from external_values directly — reanalyze would strictly degrade both.
        """
        eligible = [i for i, g in enumerate(self.buffer) if not g.external_values]
        if not eligible:
            return np.array([], dtype=np.int64), []
        n = min(n, len(eligible))
        indices = np.random.choice(eligible, size=n, replace=False)
        return indices, [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)
