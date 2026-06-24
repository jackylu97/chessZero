"""Regression test for the en-passant Zobrist-hash bug (mechanistic_verdict
2026-06-19 bug #3).

The GPU Zobrist hash used to XOR the en-passant file whenever `state.ep >= 0`
(i.e. after every double pawn push), even when no pawn could legally capture en
passant. python-chess (the ground truth for repetition) only includes the ep
square in its transposition key when `has_legal_en_passant()` is true. The
mismatch made the GPU hash spuriously unique after irrelevant double-pushes →
repetition undercount / threefold detected a ply late, and corrupted the
repetition input planes (19/20) derived from the same hash.

The fix gates the ep XOR on a friendly pawn actually attacking the ep target.
This test asserts the GPU hash includes the ep file IFF python-chess reports a
legal en passant.
"""
import chess
import pytest

from src.games.chess_gpu import GpuChessGame, _zobrist_hash


def _gpu_hash(game, board):
    st = game.from_python_chess([board])
    return int(_zobrist_hash(st)[0].item())


def _clear_ep(board):
    fen = board.fen().split()
    fen[3] = "-"
    return chess.Board(" ".join(fen))


def _board(moves):
    b = chess.Board()
    for m in moves:
        b.push_san(m)
    return b


@pytest.mark.parametrize("name,moves", [
    # Legal ep: 1.e4 e6 2.e5 d5 -> white can play exd6 e.p.
    ("legal_exd6", ["e4", "e6", "e5", "d5"]),
    # Spurious ep: 1.a4 -> ep a3 set, no black pawn can capture.
    ("spurious_a4", ["a4"]),
    # Spurious ep: 1.Nf3 c5 -> ep c6 set, no white pawn adjacent.
    ("spurious_c5", ["Nf3", "c5"]),
    # Legal ep on the other side: 1.Nf3 d5 2.Ng1 d4 3.e4 -> black dxe3 e.p.
    ("legal_dxe3", ["Nf3", "d5", "Ng1", "d4", "e4"]),
])
def test_gpu_ep_hash_matches_python_chess_legality(name, moves):
    game = GpuChessGame()
    board = _board(moves)
    assert board.ep_square is not None, f"{name}: expected an ep square set"

    py_legal = board.has_legal_en_passant()
    ep_included = _gpu_hash(game, board) != _gpu_hash(game, _clear_ep(board))

    assert ep_included == py_legal, (
        f"{name}: gpu hash ep-included={ep_included} but python-chess "
        f"has_legal_en_passant={py_legal}"
    )
