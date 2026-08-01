"""Reanalyze with ``reanalyze_use_tensor_mcts`` flag.

Verifies:
1. Default flag (False) routes through BatchedMCTS (existing behavior).
2. Flag-on routes through TensorMCTS.
3. Both paths update ``game.policies`` and ``game.root_values`` in-place
   on the same input games (correctness — the freshened targets are sane).
4. Tensor + Gumbel routes ``run_batch_gpu`` (the tensor Gumbel root) and
   produces valid legal-masked π' targets (2026-07-15; previously hard-failed).
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
    import threading
    from src.training.trainer import MuZeroTrainer
    trainer = MuZeroTrainer.__new__(MuZeroTrainer)
    trainer.config = cfg
    trainer.game = game
    trainer.network = net
    trainer.replay_buffer = buf
    trainer.device = device
    trainer.global_step = 100
    # _reanalyze serializes its in-place writes against the prefetch thread.
    trainer._buffer_lock = threading.Lock()
    # _reanalyze writes to writer; stub it
    class _NoopWriter:
        def add_scalar(self, *a, **kw): pass
    trainer.writer = _NoopWriter()
    trainer._reanalyze()
    # Reanalyze stores policies SPARSIFIED (indices, values); densify for asserts.
    from src.training.replay_buffer import _densify_policy
    A = game.action_space_size
    return [
        (_densify_policy(g.policies[0], A).copy(), float(g.root_values[0]))
        for g in buf.buffer
    ]


def test_flag_off_uses_batched_mcts(monkeypatch):
    """``reanalyze_use_tensor_mcts=False`` → BatchedMCTS code path (legacy/fallback)."""
    game = TicTacToe()
    cfg = _tiny_config(reanalyze_use_tensor_mcts=False)
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


def test_default_uses_tensor_mcts(monkeypatch):
    """Default flag (now True since 2026-05-08) → TensorMCTS code path."""
    game = TicTacToe()
    cfg = _tiny_config()  # uses default reanalyze_use_tensor_mcts (True)
    assert cfg.reanalyze_use_tensor_mcts is True, "default should be True"
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


def test_tensor_gumbel_reanalyze():
    """reanalyze_use_tensor_mcts=True + use_gumbel=True routes run_batch_gpu
    (the tensor Gumbel root; 2026-07-15, was previously a hard NotImplementedError)
    and must produce valid, legal-masked π' policies + finite root values."""
    game = TicTacToe()
    A = game.action_space_size
    cfg = _tiny_config(reanalyze_use_tensor_mcts=True, use_gumbel=True,
                       gumbel_num_considered=4)
    net = _make_net(cfg, game)
    buf = ReplayBuffer(max_size=cfg.replay_buffer_size)
    _populate_buffer(buf, game, num_games=2)
    initial = [g.policies[0].copy() for g in buf.buffer]
    snaps = _run_reanalyze_and_capture(cfg, game, net, buf, device="cpu")
    for i, (policy, root_v) in enumerate(snaps):
        assert policy.shape == (A,)
        assert np.all(policy >= 0)
        assert abs(float(policy.sum()) - 1.0) < 1e-3
        assert not np.allclose(policy, initial[i], atol=1e-4), (
            "policy didn't change from initial uniform"
        )
        assert np.isfinite(root_v) and -1.0 <= root_v <= 1.0
        # π' must put ~zero mass on illegal actions of the stored position.
        legal = set(buf.buffer[i].legal_actions_list[0])
        illegal = [a for a in range(A) if a not in legal]
        if illegal:
            assert float(policy[illegal].sum()) < 1e-5, "π' leaked mass to illegal actions"


def test_reanalyze_preserves_empty_opening_policies():
    """Opening-mix random plies store an all-zero policy (no search ran; the
    zero-sum target is a deliberate loss mask). Reanalyze must NOT backfill
    them with fresh π' — that silently unmasks plies whose flat-value roots
    produce tie-chaotic targets (2026-07-22; buffer audit found reanalyze had
    erased the marker on 103/150 old-vintage games)."""
    game = TicTacToe()
    A = game.action_space_size
    cfg = _tiny_config(reanalyze_use_tensor_mcts=True, use_gumbel=True,
                       gumbel_num_considered=4)
    net = _make_net(cfg, game)
    buf = ReplayBuffer(max_size=cfg.replay_buffer_size)
    _populate_buffer(buf, game, num_games=2)
    for g in buf.buffer:
        g.policies[0] = np.zeros(A, dtype=np.float32)   # opening-mix marker
        g.root_values[0] = 123.0                        # sentinel: must survive

    _run_reanalyze_and_capture(cfg, game, net, buf, device="cpu")

    from src.training.replay_buffer import _densify_policy, _policy_is_empty
    for g in buf.buffer:
        assert _policy_is_empty(g.policies[0]), "opening mask was backfilled"
        assert float(g.root_values[0]) == 123.0, "opening root_value overwritten"
        # Later plies WERE reanalyzed.
        p1 = _densify_policy(g.policies[1], A)
        assert abs(float(p1.sum()) - 1.0) < 1e-3
        assert not np.allclose(p1, np.full(A, 1.0 / A), atol=1e-4)


def _chess_repetition_history():
    """Chess GameHistory: Nf3 Nf6 Ng1 (3 plies) + Ng8 → at pos 3 (black to
    move), Ng8 recreates the start position a 2nd time — a forced-draw
    candidate under min_repeats=2."""
    import chess as pychess
    from src.games.chess import ChessGame, _move_to_action

    g = ChessGame()
    A = g.action_space_size
    gh = GameHistory(game_name="chess")
    state = g.reset()
    moves = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for uci in moves:
        board = state.board
        mv = pychess.Move.from_uci(uci)
        a = _move_to_action(mv, board.turn)
        legals = g.legal_actions(state)
        assert a in legals
        gh.observations.append(g.to_tensor(state))
        gh.legal_actions_list.append(list(legals))
        pol = np.zeros(A, dtype=np.float32)
        pol[legals] = 1.0 / len(legals)
        gh.policies.append(pol)
        gh.actions.append(a)
        gh.root_values.append(0.0)
        gh.rewards.append(0.0)
        state, _, _ = g.step(state, a)
    gh.observations.append(g.to_tensor(state))
    gh.game_outcome = 0.0
    rep_action = _move_to_action(pychess.Move.from_uci("f6g8"),
                                 pychess.BLACK)
    return g, gh, rep_action


def test_forced_draw_sets_replay():
    """Trainer._forced_draw_sets replays the stored actions and finds the
    repetition-completing move at the right ply (and only there)."""
    from src.training.trainer import MuZeroTrainer

    _, gh, rep_action = _chess_repetition_history()
    out = MuZeroTrainer._forced_draw_sets(gh, min_repeats=2, include_stalemate=True)
    assert len(out) == 4
    assert out[3] is not None and rep_action in out[3], (
        f"pos 3 must flag Ng8 as a forced draw, got {out[3]}")
    assert out[0] is None and out[1] is None, "no forced draw exists early"
    # Non-chess games: no veto sets.
    gh_ttt = GameHistory(game_name="tictactoe")
    assert MuZeroTrainer._forced_draw_sets(gh_ttt, 2, True) is None


def test_reanalyze_passes_veto_mask(monkeypatch):
    """With root_terminal_draws on, _reanalyze must hand run_batch_gpu the
    same forced-draw mask self-play ran under (2026-07-22: it passed none —
    reanalyze restored π' mass on repetition moves self-play suppressed)."""
    from src.model.muzero_net import MuZeroNetwork as _Net
    from src.mcts.tensor_mcts import TensorMCTS as _TMCTS

    g, gh, rep_action = _chess_repetition_history()
    A = g.action_space_size

    cfg = _tiny_config(
        reanalyze_use_tensor_mcts=True, use_gumbel=True,
        gumbel_num_considered=4, root_terminal_draws=True,
        root_terminal_draws_min_repeats=2, reanalyze_batch_size=1,
        num_parallel_games=8,  # one chunk for all 4 positions
    )
    cfg.game = "chess"
    net = _Net(
        observation_channels=g.num_planes, action_space_size=A,
        hidden_planes=8, num_blocks=1, latent_h=8, latent_w=8,
        input_h=8, input_w=8, fc_hidden=8, value_support_size=10,
    ).eval()
    buf = ReplayBuffer(max_size=16)
    buf.save_game(gh)

    captured: list = []

    def _fake_run_batch_gpu(self, obs, legal_mask, add_noise=True,
                            forced_draw_mask=None, root_tb_value=None):
        captured.append(
            forced_draw_mask.clone() if forced_draw_mask is not None else None)
        pol = legal_mask.float()
        pol = pol / pol.sum(dim=1, keepdim=True)
        return {"gumbel_policy": pol,
                "root_value": torch.zeros(obs.shape[0]),
                "gumbel_action": pol.argmax(dim=1),
                "child_actions": torch.zeros(obs.shape[0], 1, dtype=torch.int32),
                "child_visits": torch.zeros(obs.shape[0], 1, dtype=torch.int32),
                "child_priors": torch.zeros(obs.shape[0], 1)}

    monkeypatch.setattr(_TMCTS, "run_batch_gpu", _fake_run_batch_gpu)
    _run_reanalyze_and_capture(cfg, g, net, buf, device="cpu")

    assert len(captured) == 1, "expected one reanalyze chunk"
    fdm = captured[0]
    assert fdm is not None, "forced_draw_mask was not passed to run_batch_gpu"
    # Items are (game, pos 0..3) in order: only pos 3 has a veto set = {Ng8}.
    assert fdm.shape == (4, A)
    assert bool(fdm[3, rep_action]), "repetition move not flagged at pos 3"
    fdm[3, rep_action] = False
    assert not bool(fdm.any()), "veto mask flagged spurious moves"


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
