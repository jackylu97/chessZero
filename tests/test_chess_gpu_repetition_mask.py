"""GpuChessGame.repetition_move_mask vs python-chess ground truth.

Validates that the per-move repetition detector (used to steer the winning side
away from shuffling into a draw) flags exactly the legal moves that python-chess
reports as completing a 2nd/3rd repetition — across a knight shuffle, a king
shuffle (exercising the castling-rights exclusion), and a threefold case.
"""
import chess
import pytest
import torch

from src.games.chess_gpu import GpuChessGame
from src.games.chess import _action_to_move, _move_to_action


def _play(game, state, board, sans):
    for san in sans:
        mv = board.parse_san(san)
        a = _move_to_action(mv, board.turn)
        state, _, _ = game.step_batch(state, torch.tensor([a], dtype=torch.int64))
        board.push(mv)
    return state, board


def _check(game, state, board, min_repeats):
    """Assert the GPU mask matches python-chess is_repetition for every legal move."""
    legal_mask = game.legal_mask(state)
    rep_mask = game.repetition_move_mask(state, legal_mask, min_repeats=min_repeats)[0]
    n_py = 0
    for a in range(4672):
        if not bool(legal_mask[0, a]):
            assert not bool(rep_mask[a]), f"illegal action {a} flagged as repetition"
            continue
        mv = _action_to_move(a, board)
        if mv is None or mv not in board.legal_moves:
            continue
        board.push(mv)
        py_rep = board.is_repetition(min_repeats)
        board.pop()
        assert py_rep == bool(rep_mask[a]), (
            f"{board.san(mv)} (action {a}): python={py_rep} gpu={bool(rep_mask[a])}"
        )
        n_py += int(py_rep)
    return n_py


def test_knight_shuffle_repetition():
    g = GpuChessGame()
    state = g.reset_batch(1, device="cpu")
    board = chess.Board()
    # Knights out and back -> start position recurs; Nf3 again repeats.
    state, board = _play(g, state, board, ["Nf3", "Nf6", "Ng1", "Ng8"])
    n = _check(g, state, board, min_repeats=2)
    assert n == 1  # exactly Nf3


def test_king_shuffle_after_castling():
    g = GpuChessGame()
    state = g.reset_batch(1, device="cpu")
    board = chess.Board()
    # Castle both sides (kings lose rights), then shuffle kings g1<->h1 / g8<->h8.
    state, board = _play(g, state, board, [
        "e4", "e5", "Nf3", "Nf6", "Bc4", "Bc5", "O-O", "O-O",
        "Kh1", "Kh8", "Kg1", "Kg8",
    ])
    # Post-castle position has now occurred twice; a king move that returns to a
    # seen position completes a 2nd repetition. Must match python-chess exactly,
    # and the king (on g1, no rights) must be eligible (not excluded by rights).
    n = _check(g, state, board, min_repeats=2)
    assert n >= 1


def test_threefold_min_repeats_3():
    g = GpuChessGame()
    state = g.reset_batch(1, device="cpu")
    board = chess.Board()
    # Two full shuffle cycles so a move can complete a *threefold*.
    state, board = _play(g, state, board, [
        "Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8",
    ])
    n3 = _check(g, state, board, min_repeats=3)
    assert n3 == 1  # Nf3 now completes the threefold
