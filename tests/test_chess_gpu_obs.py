"""Cross-validate GpuChessGame.to_tensor_batch against ChessGame.to_tensor.

ChessGame is the oracle. Any divergence in the observation tensor means the
network would see different inputs from the GPU path than the existing pipeline,
breaking buffer compatibility silently.
"""
import random

import chess
import torch

from src.games.base import GameState
from src.games.chess import ChessGame
from src.games.chess_gpu import GpuChessGame


def _random_positions(n: int, seed: int = 0) -> list[chess.Board]:
    """Walk short random self-play games to generate diverse positions.

    Mix of plies surfaces en-passant squares, captures, castling state, and
    near-promotion positions in their natural distribution.
    """
    rng = random.Random(seed)
    out: list[chess.Board] = []
    while len(out) < n:
        b = chess.Board()
        n_plies = rng.randint(0, 80)
        for _ in range(n_plies):
            if b.is_game_over(claim_draw=False):
                break
            move = rng.choice(list(b.legal_moves))
            b.push(move)
        out.append(b.copy())
    return out


def _oracle_tensor(board: chess.Board) -> torch.Tensor:
    cg = ChessGame()
    cp = 1 if board.turn == chess.WHITE else -1
    return cg.to_tensor(GameState(board=board, current_player=cp))



def test_to_tensor_batch_matches_single_state_random():
    boards = _random_positions(10_000, seed=42)
    oracle = torch.stack([_oracle_tensor(b) for b in boards])  # (N, 19, 8, 8)

    gg = GpuChessGame()
    state = gg.from_python_chess(boards)
    batched = gg.to_tensor_batch(state)

    assert batched.shape == oracle.shape
    if not torch.equal(batched, oracle):
        diff = (batched != oracle).any(dim=(1, 2, 3))
        first = int(diff.nonzero(as_tuple=True)[0][0].item())
        # Identify which planes diverged for diagnostic noise reduction.
        plane_diff = (batched[first] != oracle[first]).any(dim=(1, 2)).nonzero(as_tuple=True)[0].tolist()
        raise AssertionError(
            f"to_tensor_batch differs from to_tensor at position {first}; "
            f"FEN={boards[first].fen()}; planes diverged: {plane_diff}"
        )


def test_to_tensor_batch_starting_position():
    gg = GpuChessGame()
    state = gg.reset_batch(4)
    batched = gg.to_tensor_batch(state)
    expected = _oracle_tensor(chess.Board())
    for i in range(4):
        assert torch.equal(batched[i], expected)



def test_to_tensor_batch_ep_square_set():
    """ep one-hot plane after a double pawn push."""
    b = chess.Board()
    b.push_san("e4")
    assert b.ep_square is not None

    expected = _oracle_tensor(b)
    gg = GpuChessGame()
    state = gg.from_python_chess([b])
    batched = gg.to_tensor_batch(state)
    assert torch.equal(batched[0], expected)


def test_to_tensor_batch_castling_revoked_one_side():
    """Castling rights planes track per-side revocation correctly."""
    b = chess.Board()
    for m in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O", "Nf6"]:
        b.push_san(m)
    assert not b.has_kingside_castling_rights(chess.WHITE)
    assert not b.has_queenside_castling_rights(chess.WHITE)
    assert b.has_kingside_castling_rights(chess.BLACK)
    assert b.has_queenside_castling_rights(chess.BLACK)

    expected = _oracle_tensor(b)
    gg = GpuChessGame()
    state = gg.from_python_chess([b])
    batched = gg.to_tensor_batch(state)
    assert torch.equal(batched[0], expected)



def test_to_tensor_batch_mixed_batch():
    """Heterogeneous batch (different turn / castling / ep) — exercises broadcasts."""
    b1 = chess.Board()
    b2 = chess.Board(); b2.push_san("e4")
    b3 = chess.Board()
    for m in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O", "Nf6"]:
        b3.push_san(m)
    boards = [b1, b2, b3]

    oracle = torch.stack([_oracle_tensor(b) for b in boards])

    gg = GpuChessGame()
    state = gg.from_python_chess(boards)
    batched = gg.to_tensor_batch(state)
    assert torch.equal(batched, oracle)
