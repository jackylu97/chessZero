"""Regression test for bug_hunt_2026_06_13.md §D.

A finished (done) game must be FULLY STATIC under further sentinel steps: its
board fields (pieces, side, castling, ep, halfmove, fullmove, ply) and its
repetition ring must not mutate, done must stay True, winner unchanged, and the
re-emitted reward must be 0. Before the fix only the done/winner flags were
frozen; the board kept being recomputed from the sentinel action, corrupting the
end-of-batch terminal observation (self_play.py final_obs).

This test also asserts a still-active game in the SAME batch keeps evolving,
proving only done games were frozen.
"""
import random

import chess
import torch

from src.games.chess import _move_to_action
from src.games.chess_gpu import GpuChessGame


def _fools_mate_pre_board() -> chess.Board:
    """Position after 1.f3 e5 2.g4 — black to move, Qh4# is forced mate."""
    b = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4"):
        b.push(chess.Move.from_uci(uci))
    return b


def test_done_game_board_frozen_active_game_evolves():
    game = GpuChessGame()
    device = "cpu"

    # Game 0: about to be mated. Game 1: fresh start (stays active).
    b0 = _fools_mate_pre_board()
    b1 = chess.Board()
    state = game.from_python_chess([b0, b1], device=device)

    # Deliver mate in game 0 (Qd8h4#); game 1 plays a benign opening move.
    mate_move = chess.Move.from_uci("d8h4")
    g1_move = chess.Move.from_uci("e2e4")
    actions = torch.tensor(
        [_move_to_action(mate_move, b0.turn), _move_to_action(g1_move, b1.turn)],
        dtype=torch.int64,
    )
    state, reward, done = game.step_batch(state, actions)

    assert bool(done[0]) and bool(state.done[0]), "game 0 should be mated"
    assert not bool(state.done[1]), "game 1 should still be active"
    assert float(reward[0]) == 1.0, "game 0 mover earns +1 on the mating move"
    winner0 = int(state.winner[0].item())  # black (mover) won -> -1 from white POV
    assert winner0 == -1

    # Snapshot the done game's terminal observation + raw fields.
    obs_done_ref = game.to_tensor_batch(state)[0].clone()
    pieces_ref = state.pieces[0].clone()
    ply_ref = int(state.ply[0].item())
    halfmove_ref = int(state.halfmove[0].item())
    fullmove_ref = int(state.fullmove[0].item())
    side_ref = int(state.side[0].item())
    castling_ref = state.castling[0].clone()
    ep_ref = int(state.ep[0].item())
    rep_hashes_ref = state.rep_hashes[0].clone()
    rep_count_ref = int(state.rep_count[0].item())

    # Track that the active game actually changes.
    obs_active_prev = game.to_tensor_batch(state)[1].clone()
    active_changed = False

    rng = random.Random(1234)
    for _ in range(20):
        # Game 0: arbitrary sentinel action (action 0). Game 1: a real legal move.
        legal_masks = _legal_mask(game, state)
        a1_choices = legal_masks[1].nonzero().flatten().tolist()
        a1 = rng.choice(a1_choices) if a1_choices else 0
        actions = torch.tensor([0, a1], dtype=torch.int64)
        state, reward, done = game.step_batch(state, actions)

        # ---- done game (0) must be byte-identical / frozen ----
        obs_done = game.to_tensor_batch(state)[0]
        assert torch.equal(obs_done, obs_done_ref), "done-game observation mutated"
        assert torch.equal(state.pieces[0], pieces_ref), "done-game pieces mutated"
        assert int(state.ply[0].item()) == ply_ref, "done-game ply advanced"
        assert int(state.halfmove[0].item()) == halfmove_ref
        assert int(state.fullmove[0].item()) == fullmove_ref
        assert int(state.side[0].item()) == side_ref, "done-game side flipped"
        assert torch.equal(state.castling[0], castling_ref)
        assert int(state.ep[0].item()) == ep_ref
        assert torch.equal(state.rep_hashes[0], rep_hashes_ref), "rep ring mutated"
        assert int(state.rep_count[0].item()) == rep_count_ref
        assert bool(state.done[0]) is True, "done flag must stay sticky"
        assert int(state.winner[0].item()) == winner0, "winner changed"
        assert float(reward[0]) == 0.0, "done game re-emitted a terminal reward"

        # ---- active game (1) should be evolving normally ----
        obs_active = game.to_tensor_batch(state)[1]
        if not torch.equal(obs_active, obs_active_prev):
            active_changed = True
        obs_active_prev = obs_active.clone()
        if state.done[1]:
            break

    assert active_changed, "active game in same batch never changed (over-frozen?)"


def _legal_mask(game: GpuChessGame, state) -> torch.Tensor:
    """Return per-game legal action masks using step_batch_with_legal's path."""
    from src.games.chess_gpu import _legal_mask_impl

    mask, _ = _legal_mask_impl(state)
    return mask
