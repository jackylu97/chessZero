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

    def to_compact_dict(self) -> dict:
        """Pack into a minimal dict for storage: drops observations and
        legal_actions_list (reconstructable by replaying actions through the
        game), and packs policies sparsely.

        One-hot policies (Stockfish shards) collapse to a single int per step.
        Soft policies (self-play) keep per-step (nonzero_indices, values)
        pairs — typically ~30-40 nonzeros for chess vs 4672 dense entries.

        Callers round-trip via ``GameHistory.from_compact_dict(d, game)``.
        """
        n = len(self.actions)

        one_hot = (
            len(self.policies) == n
            and all(
                p is not None and len(p) > 0
                and np.count_nonzero(p) == 1
                and float(p.max()) > 0.999
                for p in self.policies
            )
        )

        if one_hot:
            policies_mode = "onehot"
            policies_data = np.fromiter(
                (int(np.argmax(p)) for p in self.policies),
                dtype=np.int32,
                count=n,
            )
        else:
            policies_mode = "sparse"
            policies_data = []
            for p in self.policies:
                if p is None or len(p) == 0:
                    policies_data.append(
                        (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32))
                    )
                    continue
                nz = np.nonzero(p)[0].astype(np.int32)
                vals = np.asarray(p, dtype=np.float32)[nz]
                policies_data.append((nz, vals))

        d = {
            "format_version": 3,
            "game_name": self.game_name,
            "actions": np.asarray(self.actions, dtype=np.int32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "root_values": np.asarray(self.root_values, dtype=np.float32),
            "game_outcome": float(self.game_outcome),
            "policies_mode": policies_mode,
            "policies_data": policies_data,
        }
        if self.external_values:
            d["external_values"] = np.asarray(self.external_values, dtype=np.float32)
        return d

    @classmethod
    def from_compact_dict(cls, d: dict, game) -> "GameHistory":
        """Reconstruct a GameHistory by replaying ``d['actions']`` through ``game``.

        Rebuilds observations + legal_actions_list; expands sparse policies back
        to dense fp32 arrays of width ``game.action_space_size`` (what
        ``make_target`` expects). ``game`` must be a fresh instance whose
        ``reset()`` returns the same starting state as the original game.
        """
        version = d.get("format_version")
        if version != 3:
            raise ValueError(f"Unknown compact format version: {version!r}")

        gh = cls(game_name=d.get("game_name", ""))
        gh.actions = [int(a) for a in d["actions"]]
        gh.rewards = [float(r) for r in d["rewards"]]
        gh.root_values = [float(v) for v in d["root_values"]]
        gh.game_outcome = float(d["game_outcome"])
        ev = d.get("external_values")
        if ev is not None and len(ev) > 0:
            gh.external_values = [float(v) for v in ev]

        state = game.reset()
        gh.observations.append(game.to_tensor(state))
        for a in gh.actions:
            gh.legal_actions_list.append(game.legal_actions(state))
            state, _, _ = game.step(state, a)
            gh.observations.append(game.to_tensor(state))

        action_space_size = game.action_space_size
        mode = d["policies_mode"]
        if mode == "onehot":
            indices = d["policies_data"]
            policies = []
            for idx in indices:
                p = np.zeros(action_space_size, dtype=np.float32)
                p[int(idx)] = 1.0
                policies.append(p)
            gh.policies = policies
        elif mode == "sparse":
            policies = []
            for idxs, vals in d["policies_data"]:
                p = np.zeros(action_space_size, dtype=np.float32)
                if len(idxs) > 0:
                    p[np.asarray(idxs, dtype=np.int64)] = np.asarray(vals, dtype=np.float32)
                policies.append(p)
            gh.policies = policies
        else:
            raise ValueError(f"Unknown policies_mode: {mode!r}")

        return gh

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


def _iter_shard_games(path: str | Path, game=None):
    """Yield GameHistory objects from a Stockfish shard, legacy or compact.

    Legacy shard: a single pickled ``list[GameHistory]``.
    Compact (v2) shard: a header dict followed by N compact-dict pickles;
    each record is decoded via ``GameHistory.from_compact_dict(d, game)``.

    Used by both ``ReplayBuffer.load_warmstart_games`` and the trainer's
    per-shard injection loader.
    """
    with open(path, "rb") as f:
        first = pickle.load(f)
        if isinstance(first, list):
            for g in first:
                yield g
        elif isinstance(first, dict) and first.get("version") == 2:
            if game is None:
                raise ValueError(
                    f"{path}: compact (v2) shard requires `game` for reconstruction"
                )
            for _ in range(first["n_records"]):
                d = pickle.load(f)
                yield GameHistory.from_compact_dict(d, game)
        else:
            raise TypeError(
                f"{path}: unrecognized shard format "
                f"(got {type(first).__name__}, expected list or version-2 header dict)"
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

    def save(self, path: str | Path, filter_fn=None, max_games: int | None = None,
             format_version: int = 3):
        """Stream-pickle buffer to path, one record per game.

        Writes a header dict first, then one record pickle per game. Peak
        memory is ~one game instead of the whole buffer, which matters when
        the buffer is multi-GB: the old single-dump path allocated the
        entire serialized payload in RAM before writing and OOM'd the host.

        format_version:
            2 — legacy streaming ``(GameHistory, priority)`` pairs.
            3 — compact streaming ``(compact_dict, priority)`` pairs (default).
                Drops observations + legal_actions_list (replayable on load),
                sparsifies policies. Typically ~100× smaller than v2.
                Requires ``game`` when loading.

        filter_fn: optional ``(GameHistory) -> bool``. Only matching games
        are written; priorities stay aligned.
        max_games: if set, keep only the most recent ``max_games`` (post-filter)
        records. total_games is preserved regardless so the training-progress
        counter survives capped saves.
        """
        if format_version not in (2, 3):
            raise ValueError(f"Unsupported format_version {format_version}; expected 2 or 3")
        if filter_fn is None:
            pairs = list(zip(self.buffer, self._priorities))
        else:
            pairs = [(g, p) for g, p in zip(self.buffer, self._priorities) if filter_fn(g)]
        if max_games is not None and len(pairs) > max_games:
            pairs = pairs[-max_games:]
        header = {
            "version": format_version,
            "total_games": self.total_games,
            "n_records": len(pairs),
        }
        with open(path, "wb") as f:
            pickle.dump(header, f)
            if format_version == 3:
                for game, priority in pairs:
                    pickle.dump((game.to_compact_dict(), priority), f)
            else:
                for game, priority in pairs:
                    pickle.dump((game, priority), f)

    def load(self, path: str | Path, game=None):
        """Load buffer from path. Dispatches on header version.

        Supported:
          * v1 (legacy) — single pickled dict ``{buffer, priorities, total_games}``.
          * v2 — streaming header + ``(GameHistory, priority)`` pairs.
          * v3 — streaming header + ``(compact_dict, priority)`` pairs;
            requires ``game`` so ``GameHistory.from_compact_dict`` can replay
            actions to rebuild observations.
        """
        with open(path, "rb") as f:
            first = pickle.load(f)
            if isinstance(first, dict) and first.get("version") in (2, 3):
                version = first["version"]
                if version == 3 and game is None:
                    raise ValueError(
                        f"{path}: v3 (compact) buffer requires a `game` argument "
                        f"to reconstruct observations."
                    )
                self.buffer = []
                self._priorities = []
                self.total_games = first["total_games"]
                for _ in range(first["n_records"]):
                    record, priority = pickle.load(f)
                    if version == 3:
                        record = GameHistory.from_compact_dict(record, game)
                    self.buffer.append(record)
                    self._priorities.append(priority)
            else:
                # Legacy format: single pickled dict {"buffer", "priorities", "total_games"}
                data = first
                self.buffer = data["buffer"]
                self._priorities = data["priorities"]
                self.total_games = data["total_games"]

    def load_warmstart_games(self, paths: list[Path | str], game=None) -> int:
        """Load pickled shards of GameHistory objects into the buffer.

        Supported shard layouts:
          * Legacy — pickle of ``list[GameHistory]``.
          * Compact (v2) — header dict ``{"version": 2, "n_records": N}``
            followed by N compact-dict pickles. Requires ``game`` for
            reconstruction.

        Games are inserted via save_game so priorities are seeded to the
        current max. Returns the number of games loaded.
        """
        loaded = 0
        for p in paths:
            for g in _iter_shard_games(p, game):
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
