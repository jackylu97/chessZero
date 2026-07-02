"""Tests for the TB endgame anchor + TB rollout fill (strategy_2026_07_02.md).

Covers:
- tb_playout: TB-optimal playouts end in mate with faithful per-ply targets.
- Anchor games: compact-dict round trip through ReplayBuffer.sample_batch with
  the production target flags (hard TB value, DTM moves-left, stored policies).
- _apply_tb_rollout_fill: a won-but-drawn (synthetically simplified) game is
  truncated at its first decisive in-TB ply, finished with mate, relabeled with
  the true outcome, and keeps every GameHistory invariant.

Needs data/syzygy (+ data/gaviota for DTM) — skipped if absent.
"""
import os

import chess
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.path.isdir("data/syzygy"), reason="syzygy tablebases not available")

TB_PARAMS = dict(
    path="data/syzygy", max_pieces=5, dtz_weight=0.05, draw_score=0.0,
    value_dtz_shape=0.5,
    gaviota_path="data/gaviota" if os.path.isdir("data/gaviota") else None,
    policy_win_thresh=0.5, policy_temp=0.3,
)

# KQvK, white to move, mate in a handful — a stable decisive test position.
KQK_FEN = "3k4/8/3K4/8/8/8/8/3Q4 w - - 0 1"


def _prober():
    from src.games.syzygy_probe import SyzygyRootProber
    return SyzygyRootProber(**TB_PARAMS)


def _anchor_history(fen: str):
    """Build one anchor GameHistory (as gen_tb_anchor_games does)."""
    from src.training.replay_buffer import GameHistory
    from src.training.tb_playout import tb_playout
    board = chess.Board(fen)
    entry_turn = board.turn
    res = tb_playout(board, _prober())
    h = GameHistory(game_name="chess")
    h.start_fen = fen
    h.actions = res["actions"]
    h.policies = res["policies"]
    h.root_values = res["root_values"]
    h.rewards = res["rewards"]
    h.tablebase_values = res["tablebase_values"]
    h.tablebase_moves_left = res["tablebase_moves_left"]
    h.tablebase_policy = res["tablebase_policy"]
    h.game_outcome = 1.0 if res["winner"] == entry_turn else -1.0
    return h


def test_tb_playout_mates_and_targets():
    from src.training.tb_playout import tb_playout
    board = chess.Board(KQK_FEN)
    res = tb_playout(board, _prober())
    assert board.is_checkmate()
    assert res["winner"] == chess.WHITE
    n = len(res["actions"])
    assert n >= 1
    for key in ("policies", "root_values", "rewards",
                "tablebase_values", "tablebase_moves_left", "tablebase_policy"):
        assert len(res[key]) == n, key
    # Mover-POV rewards: exactly one +1, on the mating (last) transition.
    assert res["rewards"][-1] == 1.0 and sum(res["rewards"]) == 1.0
    # STM-POV TB values alternate sign: winner's plies positive, loser's negative.
    for i, tv in enumerate(res["tablebase_values"]):
        assert (tv > 0) == (i % 2 == 0)
    # Policy targets are normalized sparse distributions.
    for idx, w in res["policies"]:
        assert abs(float(np.sum(w)) - 1.0) < 1e-4
        assert len(idx) == len(w) >= 1


def test_anchor_game_through_sample_batch():
    from src.games.chess import ChessGame
    from src.training.replay_buffer import GameHistory, ReplayBuffer
    game = ChessGame()
    h = _anchor_history(KQK_FEN)
    # Round-trip through the shard format (what injection does).
    h2 = GameHistory.from_compact_dict(h.to_compact_dict(), game)
    assert h2.actions == h.actions
    assert len(h2.observations) == len(h2.actions) + 1
    buf = ReplayBuffer(max_size=8)
    buf.save_game(h2)
    batch, idxs, weights = buf.sample_batch(
        batch_size=4, num_unroll_steps=5, td_steps=-1, discount=1.0,
        action_space_size=game.action_space_size, value_head_type="wdl",
        tb_value_weight=1.0, tb_value_hard=True, tb_moves_left_weight=1.0,
    )
    tv = batch["target_values"]          # (B, K+1, 3) hard WDL one-hots
    assert torch.allclose(tv.sum(dim=-1), torch.ones_like(tv.sum(dim=-1)))
    # Decisive TB positions: hard one-hot means max prob == 1 at in-game plies.
    assert float(tv[:, 0].max(dim=-1).values.min()) == 1.0
    # Moves-left target uses |DTM| (< 40 for KQvK), not some shuffle length.
    ml = batch["target_moves_left"]
    if h2.tablebase_moves_left and h2.tablebase_moves_left[0] == h2.tablebase_moves_left[0]:
        assert float(ml[:, 0].max()) < 40.0
    # Policy target: normalized at in-game plies; all-zero at the terminal
    # index (make_target's intentional zero-CE no-op for terminal samples).
    tp = batch["target_policies"]
    s = tp[:, 0].sum(dim=-1)
    assert all(abs(float(x)) < 1e-4 or abs(float(x) - 1.0) < 1e-4 for x in s)


def _simplify_from_start(seed: int, max_plies: int = 160):
    """Play a deterministic capture-greedy game from the standard start until it
    reaches a DECISIVE ≤5-man position; return (actions, n_plies) or None.

    Both sides prefer the highest-value capture (stable order), so the game
    sheds material fast; whichever side ends up ahead usually holds a decisive
    ≤5-man position within ~60-120 plies.
    """
    import random as _random
    import chess.syzygy
    from src.games.chess import _move_to_action
    rng = _random.Random(seed)
    piece_val = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                 chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    board = chess.Board()
    actions = []
    tb = chess.syzygy.open_tablebase("data/syzygy")
    try:
        for _ in range(max_plies):
            if board.is_game_over():
                return None
            if len(board.piece_map()) <= 5:
                try:
                    decisive = abs(tb.probe_wdl(board)) == 2
                except Exception:
                    return None
                if decisive:
                    # Shuffle onward a few plies (what an unconverted self-play
                    # game does) so the decisive entry is a recorded PLY, not
                    # just the terminal observation.
                    for _ in range(8):
                        if board.is_game_over():
                            break
                        quiet = [m for m in board.legal_moves
                                 if not board.is_capture(m)]
                        pool = quiet or list(board.legal_moves)
                        mv = pool[rng.randrange(len(pool))]
                        if board.gives_check(mv) or board.is_capture(mv):
                            # keep the tail undramatic; skip if we can
                            alt = [m for m in pool if not board.gives_check(m)
                                   and not board.is_capture(m)]
                            if alt:
                                mv = alt[rng.randrange(len(alt))]
                        actions.append(_move_to_action(mv, board.turn))
                        board.push(mv)
                        if board.is_game_over():
                            break
                    if board.is_game_over():
                        return None
                    return actions, len(actions)
            moves = list(board.legal_moves)
            caps = [m for m in moves if board.is_capture(m)]
            if caps:
                mv = max(caps, key=lambda m: (
                    piece_val.get(board.piece_type_at(m.to_square) or chess.PAWN, 1),
                    m.from_square, m.to_square))
            else:
                mv = moves[rng.randrange(len(moves))]
            actions.append(_move_to_action(mv, board.turn))
            board.push(mv)
        return None
    finally:
        tb.close()


def test_rollout_fill_adjudicates_unconverted_game():
    from src.games.chess import ChessGame
    from src.games.syzygy_probe import relabel_fens
    from src.training.replay_buffer import GameHistory
    from src.training.self_play import _apply_tb_rollout_fill

    sim = None
    for seed in range(40):
        sim = _simplify_from_start(seed)
        if sim is not None:
            break
    assert sim is not None, "no capture-greedy game reached a decisive TB position"
    actions, L = sim

    # Build the history the resident self-play path would have produced for a
    # game that reached this position and then (hypothetically) drew: replay for
    # obs/legals, relabel in-TB plies for tablebase_values.
    game = ChessGame()
    h = GameHistory(game_name="chess")
    state = game.reset()
    fens = []
    for a in actions:
        h.observations.append(game.to_tensor(state))
        h.legal_actions_list.append(game.legal_actions(state))
        h.actions.append(a)
        h.policies.append((np.asarray([a], dtype=np.int32),
                           np.asarray([1.0], dtype=np.float32)))
        h.root_values.append(0.0)
        h.rewards.append(0.0)
        fens.append(state.board.fen() if len(state.board.piece_map()) <= 5 else None)
        state, _, _ = game.step(state, a)
    h.observations.append(game.to_tensor(state))
    h.game_outcome = 0.0          # the draw-basin outcome the fill must correct
    h.draw_by_repetition = True
    fenmap = relabel_fens(fens, TB_PARAMS, workers=0, want_policy=False)
    h.tablebase_values = [
        (fenmap[f][0] if f is not None and f in fenmap else float("nan"))
        for f in fens
    ]

    first_dec = next(i for i, tv in enumerate(h.tablebase_values)
                     if tv == tv and abs(tv) >= 0.45)
    n = _apply_tb_rollout_fill([h], TB_PARAMS)
    assert n == 1 and h.tb_filled
    # Invariants + truncation point.
    assert len(h.observations) == len(h.actions) + 1
    assert (len(h.policies) == len(h.root_values) == len(h.rewards)
            == len(h.legal_actions_list) == len(h.actions))
    assert (len(h.tablebase_values) == len(h.tablebase_moves_left)
            == len(h.tablebase_policy) == len(h.actions))
    assert len(h.actions) > first_dec
    assert not h.draw_by_repetition
    # Replaying the (truncated + filled) actions must end in checkmate with the
    # winner matching the relabeled outcome.
    b = chess.Board()
    from src.games.chess import _action_to_move
    for a in h.actions:
        mv = _action_to_move(a, b)
        assert mv is not None and mv in b.legal_moves
        b.push(mv)
    assert b.is_checkmate()
    white_won = b.turn == chess.BLACK
    assert h.game_outcome == (1.0 if white_won else -1.0)
    # Filled tail plies carry policy targets on legal moves.
    for t in range(first_dec, len(h.actions)):
        idx, w = h.policies[t]
        assert set(int(i) for i in idx) <= set(h.legal_actions_list[t])
        assert abs(float(np.sum(w)) - 1.0) < 1e-4


def test_fill_skips_seeded_and_consistent_games():
    from src.training.replay_buffer import GameHistory
    from src.training.self_play import _apply_tb_rollout_fill
    # Seeded game (start_fen set): exempt even with decisive TB values.
    h_seed = GameHistory(game_name="chess")
    h_seed.start_fen = KQK_FEN
    h_seed.actions = [0]
    h_seed.tablebase_values = [0.9]
    # Game with no TB contact: untouched.
    h_no_tb = GameHistory(game_name="chess")
    h_no_tb.actions = [0]
    assert _apply_tb_rollout_fill([h_seed, h_no_tb], TB_PARAMS) == 0
    assert not h_seed.tb_filled and not h_no_tb.tb_filled
