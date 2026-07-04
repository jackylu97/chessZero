"""Equivariance tests for D4 symmetry augmentation (src/training/symmetry.py).

Non-circularity: transformed BOARDS are built by relocating pieces via SQ_MAPS,
but their legal-move sets come from python-chess's independent rule engine —
if the move-type or square tables were wrong in any way, the mapped action
sets could not coincide with rule-generated legality on the rebuilt boards.
"""
import random

import chess
import numpy as np
import pytest
import torch

from src.games.chess import ChessGame, _move_to_action
from src.training.symmetry import (
    ACT_MAPS, N_SYM, SQ_MAPS, transform_obs, transform_policy_dense,
    window_eligible,
)


def _random_pawnless_board(rng: random.Random) -> chess.Board:
    """Random legal pawnless, castle-free position, white to move."""
    while True:
        board = chess.Board.empty()
        squares = rng.sample(range(64), k=rng.randint(3, 6))
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        for sq in squares[2:]:
            pt = rng.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
            board.set_piece_at(sq, chess.Piece(pt, rng.choice([chess.WHITE, chess.BLACK])))
        board.turn = chess.WHITE
        if board.is_valid() and not board.is_game_over():
            return board


def _transform_board(board: chess.Board, g: int) -> chess.Board:
    """Relocate pieces via SQ_MAPS (white frame). Legality of the result is
    judged by python-chess independently of our tables."""
    out = chess.Board.empty()
    for sq, piece in board.piece_map().items():
        out.set_piece_at(int(SQ_MAPS[g, sq]), piece)
    out.turn = board.turn
    return out


def test_maps_are_permutations():
    for g in range(N_SYM):
        assert sorted(SQ_MAPS[g].tolist()) == list(range(64))
        assert sorted(ACT_MAPS[g].tolist()) == list(range(ACT_MAPS.shape[1]))
    assert (SQ_MAPS[0] == np.arange(64)).all()
    assert (ACT_MAPS[0] == np.arange(ACT_MAPS.shape[1])).all()


def test_legal_action_equivariance():
    """set(g(a) for legal a) == legal-action-set of the g-transformed board,
    for hundreds of random pawnless positions and all 8 elements."""
    rng = random.Random(0)
    for _ in range(200):
        board = _random_pawnless_board(rng)
        legal = {_move_to_action(m, board.turn) for m in board.legal_moves}
        for g in range(N_SYM):
            tboard = _transform_board(board, g)
            assert tboard.is_valid(), f"g={g} produced invalid board"
            tlegal = {_move_to_action(m, tboard.turn) for m in tboard.legal_moves}
            mapped = {int(ACT_MAPS[g, a]) for a in legal}
            assert mapped == tlegal, (
                f"g={g} fen={board.fen()}: mapped legal set != rule-generated "
                f"legal set (diff {mapped ^ tlegal})")


def test_step_commutation():
    """transform(step(s, a)) == step(transform(s), transform(a)) — the game-
    symmetry identity the training unroll depends on."""
    rng = random.Random(1)
    for _ in range(100):
        board = _random_pawnless_board(rng)
        mv = rng.choice(list(board.legal_moves))
        a = _move_to_action(mv, board.turn)
        after = board.copy(); after.push(mv)
        for g in range(1, N_SYM):
            tboard = _transform_board(board, g)
            ta = int(ACT_MAPS[g, a])
            tmv = None
            for cand in tboard.legal_moves:
                if _move_to_action(cand, tboard.turn) == ta:
                    tmv = cand; break
            assert tmv is not None, f"g={g}: mapped action not legal"
            tafter = tboard.copy(); tafter.push(tmv)
            expect = _transform_board(after, g)
            assert {s: p.symbol() for s, p in tafter.piece_map().items()} == \
                   {s: p.symbol() for s, p in expect.piece_map().items()}, \
                f"g={g}: step does not commute with transform"


def test_observation_equivariance():
    """to_tensor(g·board) == transform_obs(to_tensor(board)) on the spatial
    planes, for white-to-move pawnless boards (mover frame == white frame)."""
    game = ChessGame()
    rng = random.Random(2)
    from src.games.chess import GameState
    for _ in range(50):
        board = _random_pawnless_board(rng)
        obs = game.to_tensor(GameState(board=board.copy(), done=False, winner=None,
                                       current_player=1))
        for g in range(N_SYM):
            tboard = _transform_board(board, g)
            tobs = game.to_tensor(GameState(board=tboard.copy(), done=False,
                                            winner=None, current_player=1))
            got = transform_obs(obs, g)
            assert torch.equal(got[:12], tobs[:12]), f"g={g} spatial planes differ"


def test_policy_transform_preserves_mass_and_argmax():
    rng = np.random.RandomState(3)
    p = np.zeros(ACT_MAPS.shape[1], dtype=np.float32)
    idx = rng.choice(ACT_MAPS.shape[1], size=20, replace=False)
    p[idx] = rng.dirichlet(np.ones(20)).astype(np.float32)
    for g in range(N_SYM):
        tp = transform_policy_dense(p, g)
        assert abs(tp.sum() - p.sum()) < 1e-6
        assert tp[ACT_MAPS[g, int(p.argmax())]] == p.max()


def test_window_eligibility():
    game = ChessGame()
    from src.games.chess import GameState
    pawnless = chess.Board("3k4/8/3K4/8/8/8/8/3Q4 w - - 0 1")
    withpawn = chess.Board("3k4/3p4/3K4/8/8/8/8/3Q4 w - - 0 1")
    o1 = game.to_tensor(GameState(board=pawnless, done=False, winner=None, current_player=1))
    o2 = game.to_tensor(GameState(board=withpawn, done=False, winner=None, current_player=1))
    assert window_eligible([o1])
    assert not window_eligible([o2])
    assert not window_eligible([o1, o2])
