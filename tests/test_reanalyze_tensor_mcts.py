"""Reanalyze with ``reanalyze_use_tensor_mcts`` flag.

Verifies:
1. Default flag (False) routes through BatchedMCTS (existing behavior).
2. Flag-on routes through TensorMCTS.
3. Both paths update ``game.policies`` and ``game.root_values`` in-place
   on the same input games (correctness — the freshened targets are sane).
4. The flag is rejected when combined with ``use_gumbel=True``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import MuZeroConfig
from src.games.tictactoe import TicTacToe
from src.model.muzero_net import MuZeroNetwork
from src.training.replay_buffer import GameHistory, ReplayBuffer


def _tiny_config(**overrides) -> MuZeroConfig:
    cfg = MuZeroConfig(
        game="tictactoe",
        hidden_planes=8,
        num_residual_blocks=1,
        latent_h=3, latent_w=3,
        fc_hidden=8,
        num_simulations=4,
        batch_size=2,
        training_steps=2,
        replay_buffer_size=16,
        min_buffer_size=1,
        num_self_play_games=1,
        num_parallel_games=2,
        use_amp=False,
        device="cpu",
        reanalyze_interval=1,
        reanalyze_batch_size=2,
        eval_interval=999999,
        sample_k=5,
        history_frames=1,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_net(cfg: MuZeroConfig, game: TicTacToe) -> MuZeroNetwork:
    net = MuZeroNetwork(
        observation_channels=game.num_planes * cfg.history_frames,
        action_space_size=game.action_space_size,
        hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks,
        latent_h=cfg.latent_h, latent_w=cfg.latent_w,
        input_h=game.board_size[0], input_w=game.board_size[1],
        fc_hidden=cfg.fc_hidden,
        value_support_size=cfg.value_support_size,
        reward_support_size=cfg.reward_support_size,
        action_embed_dim=cfg.action_embed_dim,
    )
    return net.eval()


def _populate_buffer(buf: ReplayBuffer, game: TicTacToe, num_games: int = 2):
    """Add num_games tic-tac-toe games to the buffer with non-trivial positions."""
    A = game.action_space_size
    for g_idx in range(num_games):
        gh = GameHistory(game_name="tictactoe")
        state = game.reset()
        for _ in range(3):
            if state.done:
                break
            obs = game.to_tensor(state)
            gh.observations.append(obs)
            legals = game.legal_actions(state)
            action = int(legals[g_idx % len(legals)])
            gh.actions.append(action)
            gh.policies.append(np.full(A, 1.0 / A, dtype=np.float32))
            gh.root_values.append(0.0)
            gh.legal_actions_list.append(list(legals))
            state, reward, done = game.step(state, action)
            gh.rewards.append(float(reward))
        gh.game_outcome = float(state.winner) if state.done else 0.0
        buf.save_game(gh)


def _run_reanalyze_and_capture(cfg, game, net, buf, device="cpu"):
    """Run a single _reanalyze pass; return snapshots of policies+root_values per game."""
    from src.training.trainer import MuZeroTrainer
    trainer = MuZeroTrainer.__new__(MuZeroTrainer)
    trainer.config = cfg
    trainer.game = game
    trainer.network = net
    trainer.replay_buffer = buf
    trainer.device = device
    trainer.global_step = 100
    # _reanalyze writes to writer; stub it
    class _NoopWriter:
        def add_scalar(self, *a, **kw): pass
    trainer.writer = _NoopWriter()
    trainer._reanalyze()
    return [
        (g.policies[0].copy(), float(g.root_values[0]))
        for g in buf.buffer
    ]


def test_default_uses_batched_mcts(monkeypatch):
    """Default flag (off) → BatchedMCTS code path."""
    game = TicTacToe()
    cfg = _tiny_config()
    net = _make_net(cfg, game)
    buf = ReplayBuffer(max_size=cfg.replay_buffer_size)
    _populate_buffer(buf, game, num_games=2)

    # Sentinel: track which class was constructed
    constructed: list[str] = []
    from src.mcts.mcts import BatchedMCTS as _BMCTS
    orig_init = _BMCTS.__init__
    def _spy(self, *a, **kw):
        constructed.append("BatchedMCTS")
        return orig_init(self, *a, **kw)
    monkeypatch.setattr(_BMCTS, "__init__", _spy)

    _run_reanalyze_and_capture(cfg, game, net, buf, device="cpu")
    assert constructed == ["BatchedMCTS"], f"expected BatchedMCTS, got {constructed!r}"


def test_flag_on_uses_tensor_mcts(monkeypatch):
    """Flag on → TensorMCTS code path."""
    game = TicTacToe()
    cfg = _tiny_config(reanalyze_use_tensor_mcts=True)
    net = _make_net(cfg, game)
    buf = ReplayBuffer(max_size=cfg.replay_buffer_size)
    _populate_buffer(buf, game, num_games=2)

    constructed: list[str] = []
    from src.mcts.tensor_mcts import TensorMCTS as _TMCTS
    orig_init = _TMCTS.__init__
    def _spy(self, *a, **kw):
        constructed.append("TensorMCTS")
        return orig_init(self, *a, **kw)
    monkeypatch.setattr(_TMCTS, "__init__", _spy)

    _run_reanalyze_and_capture(cfg, game, net, buf, device="cpu")
    assert constructed == ["TensorMCTS"], f"expected TensorMCTS, got {constructed!r}"


def test_both_paths_update_policies_and_values():
    """Both backends must mutate policies + root_values to fresh, valid distributions."""
    game = TicTacToe()
    A = game.action_space_size

    for flag, name in [(False, "BatchedMCTS"), (True, "TensorMCTS")]:
        cfg = _tiny_config(reanalyze_use_tensor_mcts=flag)
        net = _make_net(cfg, game)
        buf = ReplayBuffer(max_size=cfg.replay_buffer_size)
        _populate_buffer(buf, game, num_games=2)

        # Snapshot the initial uniform policy for comparison.
        initial = [g.policies[0].copy() for g in buf.buffer]

        snaps = _run_reanalyze_and_capture(cfg, game, net, buf, device="cpu")

        for i, (policy, root_v) in enumerate(snaps):
            # Each policy must still be a valid distribution.
            assert policy.shape == (A,), f"[{name}] policy shape mismatch"
            assert np.all(policy >= 0), f"[{name}] negative policy"
            s = float(policy.sum())
            assert abs(s - 1.0) < 1e-3, f"[{name}] policy doesn't sum to 1: {s}"
            # Reanalyze should produce a non-trivial change vs the initial uniform.
            # (With 4 sims and a fresh net, the visit distribution should differ.)
            assert not np.allclose(policy, initial[i], atol=1e-4), (
                f"[{name}] policy didn't change from initial uniform"
            )
            # Root value should be a finite scalar in [-1, 1].
            assert np.isfinite(root_v) and -1.0 <= root_v <= 1.0, (
                f"[{name}] bad root_v: {root_v}"
            )


def test_flag_rejects_use_gumbel():
    """reanalyze_use_tensor_mcts=True + use_gumbel=True must raise."""
    game = TicTacToe()
    cfg = _tiny_config(reanalyze_use_tensor_mcts=True, use_gumbel=True)
    net = _make_net(cfg, game)
    buf = ReplayBuffer(max_size=cfg.replay_buffer_size)
    _populate_buffer(buf, game, num_games=2)
    with pytest.raises(NotImplementedError, match="Gumbel"):
        _run_reanalyze_and_capture(cfg, game, net, buf, device="cpu")


def test_flag_on_with_amp_dtype_set():
    """Cover the truthy ``amp_str`` branch of the dtype dispatch.

    The chess production preset has ``tensor_mcts_amp_dtype='float16'``;
    without this test the truthy ternary branch of
    ``amp_dtype = _dtype_map[amp_str] if amp_str else None`` is uncovered.
    On CPU, TensorMCTS internally nullifies the amp_dtype (no autocast
    available), but the dispatch code still runs.
    """
    game = TicTacToe()
    cfg = _tiny_config(
        reanalyze_use_tensor_mcts=True,
        tensor_mcts_amp_dtype="float16",
    )
    net = _make_net(cfg, game)
    buf = ReplayBuffer(max_size=cfg.replay_buffer_size)
    _populate_buffer(buf, game, num_games=2)
    # Must run without raising; produces valid policies + values.
    snaps = _run_reanalyze_and_capture(cfg, game, net, buf, device="cpu")
    for policy, root_v in snaps:
        assert abs(float(policy.sum()) - 1.0) < 1e-3
        assert -1.0 <= root_v <= 1.0
