"""End-to-end check that play_games_parallel_gpu produces valid GameHistory
records and that, given the same actions, the GPU env path keeps state +
observations + legal masks identical to ChessGame.

The phase-1-to-3 cross-validation already proves the underlying primitives
match. This test exercises the full self-play loop wiring.
"""
import random

import chess
import torch

from src.config import get_config
from src.games.chess import ChessGame
from src.games.chess_gpu import GpuChessGame
from src.games.base import GameState
from src.model.muzero_net import MuZeroNetwork
from src.training.replay_buffer import GameHistory, stack_with_history
from src.training.self_play import play_games_parallel_gpu


def _make_small_chess_config():
    """Trim the chess preset for fast self-play in tests."""
    cfg = get_config("chess")
    cfg.num_simulations = 4              # tiny — we're testing wiring, not search quality
    cfg.num_parallel_games = 4
    cfg.history_frames = 8
    cfg.sample_k = 50
    cfg.use_gumbel = False
    cfg.use_consistency_loss = False     # don't need projection heads for this test
    # The chess preset selects the 'triton' PUCT backend, which requires CUDA.
    # These tests run on CPU (device="cpu"), so use the CPU-compatible 'eager'
    # backend — this test checks history/replay equivalence, not search speed.
    cfg.tensor_mcts_select_backend = "eager"
    return cfg


def _build_network(cfg, game):
    """Tiny untrained network — we just need forward to work."""
    cfg.hidden_planes = 16
    cfg.num_residual_blocks = 1
    cfg.fc_hidden = 16
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
        proj_hid=cfg.proj_hid,
        proj_out=cfg.proj_out,
        pred_hid=cfg.pred_hid,
        pred_out=cfg.pred_out,
        use_scalar_transform=cfg.use_scalar_transform,
        value_target_scale=cfg.value_target_scale,
        value_head_type=getattr(cfg, "value_head_type", "support"),
        draw_score=getattr(cfg, "draw_score", 0.0),
    )


def test_play_games_parallel_gpu_runs():
    """Smoke test: 4 games run end-to-end and produce well-shaped histories."""
    cfg = _make_small_chess_config()
    chess_game = ChessGame()
    network = _build_network(cfg, chess_game)
    torch.manual_seed(0)
    histories = play_games_parallel_gpu(
        network, cfg, num_games=4, device="cpu", training_step=0,
    )
    assert len(histories) == 4
    for h in histories:
        # At least one move played (a 4-sim untrained MCTS may still finish a game
        # quickly via random-ish play, but it always plays at least one move).
        assert isinstance(h, GameHistory)
        assert len(h.actions) >= 1
        assert len(h.observations) == len(h.actions) + 1
        assert len(h.policies) == len(h.actions)
        assert len(h.root_values) == len(h.actions)
        assert len(h.legal_actions_list) == len(h.actions)
        assert len(h.rewards) == len(h.actions)
        assert h.game_outcome in (-1, 0, 1)
        # Every action recorded must be a valid index.
        for a in h.actions:
            assert 0 <= a < chess_game.action_space_size


def test_history_replay_equivalence():
    """For a small set of GPU-self-played games, replay the same action
    sequences through ChessGame and verify the per-ply observations + legal
    sets line up. This is a stronger check than the smoke test above —
    it asserts that the histories the GPU path emits are *consumable* by
    the existing CPU replay code (which is what reanalyze + buffer use).
    """
    cfg = _make_small_chess_config()
    chess_game = ChessGame()
    network = _build_network(cfg, chess_game)
    torch.manual_seed(1)

    histories = play_games_parallel_gpu(
        network, cfg, num_games=4, device="cpu", training_step=0,
    )

    # Replay each history through ChessGame.
    for hi, h in enumerate(histories):
        cpu_state = chess_game.reset()
        for ply, action in enumerate(h.actions):
            # The recorded observation must equal what ChessGame would have produced.
            cpu_obs = chess_game.to_tensor(cpu_state)
            assert torch.equal(cpu_obs, h.observations[ply]), (
                f"game {hi} ply {ply}: stored observation diverges from ChessGame.to_tensor"
            )
            # The recorded legal-action set must equal what ChessGame would have produced.
            cpu_legals = set(chess_game.legal_actions(cpu_state))
            stored_legals = set(h.legal_actions_list[ply])
            assert cpu_legals == stored_legals, (
                f"game {hi} ply {ply}: stored legal set diverges from ChessGame.legal_actions"
            )
            assert action in cpu_legals, (
                f"game {hi} ply {ply}: stored action {action} not in ChessGame's legal set"
            )
            # Step CPU oracle.
            cpu_state, reward, done = chess_game.step(cpu_state, action)
            # Reward should match the reward the GPU path recorded.
            assert reward == h.rewards[ply], (
                f"game {hi} ply {ply}: reward {reward} (CPU) vs {h.rewards[ply]} (GPU)"
            )
            if done:
                # GPU path also marks done at this ply.
                assert ply == len(h.actions) - 1
                # game_outcome must match CPU oracle's winner.
                assert h.game_outcome == cpu_state.winner

        # Final observation matches.
        cpu_final_obs = chess_game.to_tensor(cpu_state)
        assert torch.equal(cpu_final_obs, h.observations[-1]), (
            f"game {hi}: final observation diverges"
        )
