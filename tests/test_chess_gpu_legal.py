"""Cross-validate GpuChessGame.legal_mask against ChessGame.legal_actions.

ChessGame is the oracle. Every legal action set the GPU code produces must
exactly match python-chess's `legal_moves` (encoded via `_move_to_action`).
"""
import gc
import random

import chess
import torch

from src.games.chess import ACTION_SPACE, ChessGame, _move_to_action
from src.games.chess_gpu import GpuChessGame


def _oracle_legal_set(board: chess.Board) -> set[int]:
    return {_move_to_action(m) for m in board.legal_moves}


def _gpu_legal_set(mask_row: torch.Tensor) -> set[int]:
    return {int(i) for i in mask_row.nonzero(as_tuple=True)[0].tolist()}


def _diff_report(b: chess.Board, oracle: set[int], got: set[int]) -> str:
    extra = got - oracle           # GPU says legal, oracle says not
    missing = oracle - got          # oracle says legal, GPU says not
    parts = []
    if extra:
        parts.append("EXTRA (GPU says legal, oracle no): " + ", ".join(
            f"{i}({chess.square_name(i // 73)}→mt{i % 73})" for i in sorted(extra)
        ))
    if missing:
        parts.append("MISSING (oracle says legal, GPU no): " + ", ".join(
            f"{i}({chess.square_name(i // 73)}→mt{i % 73})" for i in sorted(missing)
        ))
    return f"\nFEN={b.fen()}\n" + "\n".join(parts)


def _random_positions(n: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    out: list[chess.Board] = []
    while len(out) < n:
        b = chess.Board()
        n_plies = rng.randint(0, 80)
        for _ in range(n_plies):
            if b.is_game_over(claim_draw=False):
                break
            b.push(rng.choice(list(b.legal_moves)))
        out.append(b.copy())
    return out


def test_legal_mask_starting_position():
    b = chess.Board()
    oracle = _oracle_legal_set(b)
    assert len(oracle) == 20  # standard fact: 20 first moves

    gg = GpuChessGame()
    state = gg.from_python_chess([b])
    mask = gg.legal_mask(state)
    got = _gpu_legal_set(mask[0])
    assert got == oracle, _diff_report(b, oracle, got)


def test_legal_mask_ep_pin_horizontal():
    """EP capture is illegal when removing both pawns exposes the king on the rank.

    Setup: white K on e5, P on f5, black just played g7-g5 (EP target g6).
    Black rook on h5. Capturing EP (f5xg6) would clear f5 and g5, leaving
    the rook with a clear line to the king on e5. Oracle rejects this move;
    standard pin filter doesn't catch it because the pin is created BY the
    EP move, not pre-existing.
    """
    b = chess.Board("4k3/8/8/4KPpr/8/8/8/8 w - g6 0 1")
    oracle = _oracle_legal_set(b)
    # Sanity: make sure oracle has the right shape (king moves + maybe a push,
    # but NOT the f5xg6 EP capture).
    f5xg6_action = _move_to_action(chess.Move(chess.F5, chess.G6))
    assert f5xg6_action not in oracle, "oracle should reject f5xg6 EP"

    gg = GpuChessGame()
    state = gg.from_python_chess([b])
    mask = gg.legal_mask(state)
    got = _gpu_legal_set(mask[0])
    assert got == oracle, _diff_report(b, oracle, got)


def test_legal_mask_after_e4():
    b = chess.Board()
    b.push_san("e4")
    oracle = _oracle_legal_set(b)

    gg = GpuChessGame()
    state = gg.from_python_chess([b])
    mask = gg.legal_mask(state)
    got = _gpu_legal_set(mask[0])
    assert got == oracle, _diff_report(b, oracle, got)


def test_legal_mask_random_small_corpus():
    """50k random positions: full cross-validation against ChessGame oracle.

    Surfaces EP-pins, castling-through-check, double-check, and other edge
    cases that the smaller obs/attacks corpora don't reach.

    Chunked at 5k positions/chunk: the legal_mask path allocates ~25 MB per
    intermediate `(N, 64)` int64 tensor over ~2.5k Python-loop allocations
    (Kogge-Stone slider attacks × 8 directions × 64 from-squares). At
    N=50k that pegs WSL's allocator into 1-2 GB of cached churn, which
    on this box has historically tripped Hyper-V instability. 5k×10
    keeps peak working set bounded and forces allocator reuse via gc.
    """
    chunk_size = 5_000
    n_total = 50_000
    seed = 99
    gg = GpuChessGame()

    # Generate positions chunk-by-chunk to bound python-chess board memory too.
    rng = random.Random(seed)
    fails: list[tuple[int, str]] = []
    n_seen = 0
    while n_seen < n_total and len(fails) < 3:
        chunk_n = min(chunk_size, n_total - n_seen)
        boards: list[chess.Board] = []
        while len(boards) < chunk_n:
            b = chess.Board()
            n_plies = rng.randint(0, 80)
            for _ in range(n_plies):
                if b.is_game_over(claim_draw=False):
                    break
                b.push(rng.choice(list(b.legal_moves)))
            boards.append(b.copy())

        state = gg.from_python_chess(boards)
        mask = gg.legal_mask(state)
        for i, b in enumerate(boards):
            oracle = _oracle_legal_set(b)
            got = _gpu_legal_set(mask[i])
            if got != oracle:
                fails.append((n_seen + i, _diff_report(b, oracle, got)))
                if len(fails) >= 3:
                    break
        n_seen += chunk_n

        # Drop refs + force gc so allocator reuses memory across chunks
        # rather than letting cached intermediates accumulate.
        del state, mask, boards
        gc.collect()

    if fails:
        msg = "\n\n".join(f"--- position {i} ---{rep}" for i, rep in fails)
        raise AssertionError(f"{len(fails)} divergences in {n_seen} positions:\n{msg}")
