"""Unit tests for ``multipv_to_soft_policy`` in scripts/generate_stockfish_games.py.

The helper builds a soft policy target from a Stockfish MultiPV ``info_list``:
``softmax(scores / tau_label)`` over top-K moves, packed into a dense
``action_space_size`` vector. These tests fake the Stockfish output shape so
they don't require an installed engine.
"""
import math
import os
import sys
from types import SimpleNamespace

import chess
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.chess import ChessGame, _move_to_action  # noqa: E402

# Import target by exec'ing the script's module namespace, since it lives in scripts/
# and isn't a package.
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "generate_stockfish_games",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "generate_stockfish_games.py"),
)
_gsg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_gsg)
multipv_to_soft_policy = _gsg.multipv_to_soft_policy
score_to_value = _gsg.score_to_value


def _fake_info(move: chess.Move, cp: int) -> dict:
    """Mimic the dict shape of one entry in ``engine.analyse(..., multipv=K)``."""
    return {
        "pv": [move],
        "score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE),
    }


def test_returns_zero_vector_on_empty_info():
    game = ChessGame()
    out = multipv_to_soft_policy([], chess.WHITE, 0.10, game.action_space_size)
    assert out.shape == (game.action_space_size,)
    assert out.sum() == 0.0


def test_one_hot_when_single_move():
    """len(info_list) == 1 → exact one-hot at that move."""
    game = ChessGame()
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    info = [_fake_info(move, 30)]
    out = multipv_to_soft_policy(info, chess.WHITE, 0.10, game.action_space_size)
    idx = _move_to_action(move)
    assert out[idx] == pytest.approx(1.0)
    out_other = np.delete(out, idx)
    assert (out_other == 0).all()


def test_zero_temperature_falls_back_to_one_hot():
    """tau_label <= 0 → exact one-hot on the first (top) move."""
    game = ChessGame()
    e4 = chess.Move.from_uci("e2e4")
    d4 = chess.Move.from_uci("d2d4")
    info = [_fake_info(e4, 30), _fake_info(d4, 28)]
    out = multipv_to_soft_policy(info, chess.WHITE, 0.0, game.action_space_size)
    assert out[_move_to_action(e4)] == pytest.approx(1.0)
    assert out[_move_to_action(d4)] == pytest.approx(0.0)


def test_softmax_probabilities_sum_to_one():
    game = ChessGame()
    moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("d2d4"),
        chess.Move.from_uci("g1f3"),
    ]
    cps = [30, 20, 10]
    info = [_fake_info(m, c) for m, c in zip(moves, cps)]
    out = multipv_to_soft_policy(info, chess.WHITE, 0.10, game.action_space_size)
    assert out.sum() == pytest.approx(1.0)
    for m in moves:
        assert out[_move_to_action(m)] > 0.0


def test_higher_score_gets_higher_probability():
    """Verify softmax orders correctly: rank 1 mass > rank 2 mass > rank 3 mass."""
    game = ChessGame()
    moves = [
        chess.Move.from_uci("e2e4"),  # best
        chess.Move.from_uci("d2d4"),  # second
        chess.Move.from_uci("g1f3"),  # third
    ]
    cps = [50, 20, 0]
    info = [_fake_info(m, c) for m, c in zip(moves, cps)]
    out = multipv_to_soft_policy(info, chess.WHITE, 0.10, game.action_space_size)
    p1 = out[_move_to_action(moves[0])]
    p2 = out[_move_to_action(moves[1])]
    p3 = out[_move_to_action(moves[2])]
    assert p1 > p2 > p3
    # Cross-check the actual values match softmax of the score_to_value transform.
    evals = np.array([score_to_value(i["score"], chess.WHITE) for i in info])
    expected = np.exp((evals - evals.max()) / 0.10)
    expected /= expected.sum()
    np.testing.assert_allclose([p1, p2, p3], expected, atol=1e-6)


def test_tau_extremes_at_depth_8_landscape():
    """Sanity: at the depth-8 flat eval landscape (top-3 within ~50cp), small τ
    still gives a noticeable spread because gaps are tiny in raw [-1,+1] space."""
    game = ChessGame()
    moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("d2d4"),
        chess.Move.from_uci("g1f3"),
    ]
    cps = [42, 6, 13]  # actual depth-8 scores at start position
    info = [_fake_info(m, c) for m, c in zip(moves, cps)]
    out_low = multipv_to_soft_policy(info, chess.WHITE, 0.05, game.action_space_size)
    out_hi = multipv_to_soft_policy(info, chess.WHITE, 0.50, game.action_space_size)

    p1_low = out_low[_move_to_action(moves[0])]
    p1_hi = out_hi[_move_to_action(moves[0])]
    # Lower τ → sharper preference for top move; higher τ → flatter.
    assert p1_low > p1_hi
    # But because eval gaps are small, even τ=0.05 doesn't go to one-hot.
    assert p1_low < 0.95
    # All three remain present in both.
    for o in (out_low, out_hi):
        for m in moves:
            assert o[_move_to_action(m)] > 0


def test_dtype_and_shape():
    game = ChessGame()
    info = [_fake_info(chess.Move.from_uci("e2e4"), 30)]
    out = multipv_to_soft_policy(info, chess.WHITE, 0.10, game.action_space_size)
    assert out.dtype == np.float32
    assert out.shape == (game.action_space_size,)
