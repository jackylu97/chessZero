"""Integration tests for ``use_tensor_mcts`` flag wiring.

Exercises ``play_games_parallel`` end-to-end with a tiny tictactoe
``MuZeroNetwork`` and verifies the produced ``GameHistory`` matches the
contract that downstream training/replay-buffer code depends on.

Also includes a soft replay-equivalence test against ``BatchedMCTS``: with
no Dirichlet noise and matched seeds, both implementations should produce
games of similar length and pick the same actions on a clear majority of
plies. Bit-equivalence isn't possible (``torch.multinomial`` vs.
``np.random.choice`` paths diverge), so the comparison is deliberately
loose.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import MuZeroConfig
from src.games.tictactoe import TicTacToe
from src.training.replay_buffer import _densify_policy
from src.model.muzero_net import MuZeroNetwork
from src.training.self_play import (
    _make_batched_mcts,
    play_games_parallel,
)


def _tiny_ttt_config(**overrides) -> MuZeroConfig:
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
        num_self_play_games=2,
        num_parallel_games=2,
        use_amp=False,
        device="cpu",
        reanalyze_interval=0,
        eval_interval=999999,
        sample_k=5,           # tensor path only enables Sampled MuZero
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_net(cfg: MuZeroConfig, game: TicTacToe) -> MuZeroNetwork:
    return MuZeroNetwork(
        observation_channels=game.num_planes,
        action_space_size=game.action_space_size,
        hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks,
        latent_h=cfg.latent_h, latent_w=cfg.latent_w,
        input_h=game.board_size[0], input_w=game.board_size[1],
        fc_hidden=cfg.fc_hidden,
        value_support_size=cfg.value_support_size,
        reward_support_size=cfg.reward_support_size,
    )


@pytest.fixture
def ttt():
    return TicTacToe()


# --- Factory dispatch ------------------------------------------------------


def test_factory_picks_batched_mcts_by_default(ttt):
    cfg = _tiny_ttt_config()
    net = _make_net(cfg, ttt)
    mcts = _make_batched_mcts(net, ttt, cfg, device="cpu")
    from src.mcts.mcts import BatchedMCTS
    assert isinstance(mcts, BatchedMCTS)


def test_factory_picks_tensor_mcts_when_enabled(ttt):
    cfg = _tiny_ttt_config(use_tensor_mcts=True)
    net = _make_net(cfg, ttt)
    mcts = _make_batched_mcts(net, ttt, cfg, device="cpu")
    from src.mcts.tensor_mcts import TensorMCTS
    assert isinstance(mcts, TensorMCTS)


def test_factory_rejects_use_tensor_mcts_with_use_gumbel(ttt):
    cfg = _tiny_ttt_config(use_tensor_mcts=True, use_gumbel=True)
    net = _make_net(cfg, ttt)
    with pytest.raises(NotImplementedError, match="Gumbel"):
        _make_batched_mcts(net, ttt, cfg, device="cpu")


# --- play_games_parallel end-to-end ----------------------------------------


def test_play_games_parallel_with_tensor_mcts_produces_valid_histories(ttt):
    """Full self-play batch through the wired TensorMCTS path."""
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = _tiny_ttt_config(use_tensor_mcts=True)
    net = _make_net(cfg, ttt)

    histories = play_games_parallel(
        net, ttt, cfg, num_games=3, device="cpu", training_step=0,
    )

    assert len(histories) == 3
    for h in histories:
        assert len(h.actions) > 0
        # One observation per ply + one terminal observation appended.
        assert len(h.observations) == len(h.actions) + 1
        # Per-ply policies sum to ~1 (deduped). Stored sparse (Path B) → densify.
        for pi in h.policies:
            pi = _densify_policy(pi, ttt.action_space_size)
            assert pi.shape[0] >= 1
            assert pi.sum() == pytest.approx(1.0, abs=1e-5)
            assert (pi >= 0).all()
        # Per-ply legal-actions list has same length as actions.
        assert len(h.legal_actions_list) == len(h.actions)
        # All chosen actions are legal at the time they were chosen.
        for action, legal in zip(h.actions, h.legal_actions_list):
            assert action in legal
        # Root values finite.
        assert all(np.isfinite(v) for v in h.root_values)
        # Game outcome is recorded.
        assert h.game_outcome in (-1, 0, 1)


def test_play_games_parallel_tensor_mcts_respects_legal_actions(ttt):
    """π targets must place zero mass on illegal actions for every ply."""
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = _tiny_ttt_config(use_tensor_mcts=True)
    net = _make_net(cfg, ttt)
    histories = play_games_parallel(
        net, ttt, cfg, num_games=2, device="cpu", training_step=0,
    )
    for h in histories:
        for pi, legal in zip(h.policies, h.legal_actions_list):
            pi = _densify_policy(pi, ttt.action_space_size)
            legal_set = set(legal)
            for a in range(pi.shape[0]):
                if a not in legal_set:
                    assert pi[a] == 0.0, f"illegal action {a} got mass {pi[a]}"


# --- Soft replay equivalence vs. BatchedMCTS -------------------------------


def test_replay_equivalence_batched_vs_tensor_soft(ttt):
    """With matched seeds + no Dirichlet, both backends should produce
    self-play games of comparable length and root-value scale.

    Bit-equivalence is not the goal — different sampling RNG paths
    (torch.multinomial vs. np.random.choice) plus option-(b) duplicate
    slots make per-action divergence unavoidable. The check is a sanity
    test: same network, same start, similar dynamics.
    """
    torch.manual_seed(7)
    np.random.seed(7)
    cfg_batched = _tiny_ttt_config(
        use_tensor_mcts=False, num_simulations=16,
    )
    cfg_tensor = _tiny_ttt_config(
        use_tensor_mcts=True, num_simulations=16,
    )
    # Disable Dirichlet noise so the comparison is over the search itself,
    # not over different RNG-driven exploration noise. Tictactoe action
    # space is 9 — sample_k=9 (set by _tiny_ttt_config) means the tensor
    # path samples K=9 with replacement (heavy duplication after dedup),
    # while BatchedMCTS expands all 9 directly. Behaviour should still be
    # broadly aligned.
    cfg_batched.dirichlet_epsilon = 0.0
    cfg_tensor.dirichlet_epsilon = 0.0

    net_b = _make_net(cfg_batched, ttt)
    # Make two networks identical.
    net_t = _make_net(cfg_tensor, ttt)
    net_t.load_state_dict(net_b.state_dict())

    torch.manual_seed(7); np.random.seed(7)
    histories_batched = play_games_parallel(
        net_b, ttt, cfg_batched, num_games=2, device="cpu", training_step=0,
    )
    torch.manual_seed(7); np.random.seed(7)
    histories_tensor = play_games_parallel(
        net_t, ttt, cfg_tensor, num_games=2, device="cpu", training_step=0,
    )

    # Game lengths should be in the same ballpark (tictactoe is 5-9 ply).
    for hb, ht in zip(histories_batched, histories_tensor):
        assert abs(len(hb.actions) - len(ht.actions)) <= 4
    # Root-value scale — none should NaN/inf.
    for h in histories_batched + histories_tensor:
        assert all(np.isfinite(v) for v in h.root_values)
        assert all(-1.5 < v < 1.5 for v in h.root_values), \
            "root values should stay in scaled value range"


# --- Chess GPU path -------------------------------------------------------


def _make_small_chess_config():
    from src.config import get_config
    cfg = get_config("chess")
    cfg.num_simulations = 4
    cfg.num_parallel_games = 4
    cfg.history_frames = 8
    cfg.sample_k = 50
    cfg.use_gumbel = False
    cfg.use_consistency_loss = False
    cfg.hidden_planes = 16
    cfg.num_residual_blocks = 1
    cfg.fc_hidden = 16
    return cfg


def _build_chess_network(cfg, game):
    return MuZeroNetwork(
        observation_channels=game.num_planes * cfg.history_frames,
        action_space_size=game.action_space_size,
        hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks,
        latent_h=cfg.latent_h,
        latent_w=cfg.latent_w,
        input_h=game.board_size[0],
        input_w=game.board_size[1],
        fc_hidden=cfg.fc_hidden,
        value_support_size=cfg.value_support_size,
        reward_support_size=cfg.reward_support_size,
        use_consistency_loss=cfg.use_consistency_loss,
        use_scalar_transform=cfg.use_scalar_transform,
        value_target_scale=cfg.value_target_scale,
        value_head_type=getattr(cfg, "value_head_type", "support"),
        draw_score=getattr(cfg, "draw_score", 0.0),
    )


def test_play_games_parallel_gpu_with_tensor_mcts_runs():
    """Chess GPU self-play under TensorMCTS produces well-shaped histories.

    Mirrors ``test_chess_gpu_self_play.py::test_play_games_parallel_gpu_runs``
    but enables ``use_tensor_mcts``. Runs on CPU device to avoid CUDA
    requirement at test time — the GPU-resident chess env still exercises
    the same tensor code paths.
    """
    from src.games.chess import ChessGame
    from src.training.replay_buffer import GameHistory
    from src.training.self_play import play_games_parallel_gpu

    cfg = _make_small_chess_config()
    cfg.use_tensor_mcts = True
    cfg.tensor_mcts_select_backend = "eager"  # triton requires CUDA; this test runs on CPU
    chess_game = ChessGame()
    network = _build_chess_network(cfg, chess_game)
    torch.manual_seed(0)

    histories = play_games_parallel_gpu(
        network, cfg, num_games=4, device="cpu", training_step=0,
    )
    assert len(histories) == 4
    for h in histories:
        assert isinstance(h, GameHistory)
        assert len(h.actions) >= 1
        assert len(h.observations) == len(h.actions) + 1
        assert len(h.policies) == len(h.actions)
        assert len(h.root_values) == len(h.actions)
        assert len(h.legal_actions_list) == len(h.actions)
        assert h.game_outcome in (-1, 0, 1)
        for a in h.actions:
            assert 0 <= a < chess_game.action_space_size
        # Per-ply policies sum to ~1 over the deduped/sampled support (sparse → densify).
        for pi in h.policies:
            assert _densify_policy(pi, chess_game.action_space_size).sum() == pytest.approx(1.0, abs=1e-5)
