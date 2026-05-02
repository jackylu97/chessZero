"""Cross-validate attacks_by_color (batched) against python-chess.

The attack bitboard is the foundation for legality (king-can't-move-into-check,
castling-path-not-attacked, and check detection). Any divergence from
python-chess here propagates into legal_mask in Phase 2.
"""
import random

import chess
import numpy as np
import torch

from src.games.chess_gpu import (
    GpuChessGame,
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    PAWN_ATTACKS_B,
    PAWN_ATTACKS_W,
    RAYS,
    attacks_by_color,
)


def _attacks_oracle(board: chess.Board, color: bool) -> int:
    """Reference attack bitboard: union of `is_attacked_by(color, sq)` over all squares."""
    bb = 0
    for sq in range(64):
        if board.is_attacked_by(color, sq):
            bb |= 1 << sq
    return bb


def _bb_to_int(t: torch.Tensor) -> int:
    """Reinterpret a single int64 tensor as the equivalent uint64 bit pattern."""
    arr = t.to(torch.int64).cpu().numpy().astype(np.int64)
    return int(arr.view(np.uint64).item())


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


def test_lookup_table_shapes():
    assert KNIGHT_ATTACKS.shape == (64,)
    assert KING_ATTACKS.shape == (64,)
    assert PAWN_ATTACKS_W.shape == (64,)
    assert PAWN_ATTACKS_B.shape == (64,)
    assert RAYS.shape == (8, 64)


def test_knight_attacks_d4():
    """Sanity: knight on d4 attacks 8 squares (b3, b5, c2, c6, e2, e6, f3, f5)."""
    d4 = chess.D4
    expected = {
        chess.B3, chess.B5, chess.C2, chess.C6,
        chess.E2, chess.E6, chess.F3, chess.F5,
    }
    bb = _bb_to_int(KNIGHT_ATTACKS[d4])
    bits = {s for s in range(64) if (bb >> s) & 1}
    assert bits == expected


def test_pawn_attacks_white_e4():
    """White pawn on e4 attacks d5 and f5."""
    e4 = chess.E4
    expected = {chess.D5, chess.F5}
    bb = _bb_to_int(PAWN_ATTACKS_W[e4])
    bits = {s for s in range(64) if (bb >> s) & 1}
    assert bits == expected


def test_attacks_by_color_starting_position():
    """Starting position: white attacks rank 3 (pawn-defended) + b1+g1 + e2/d2 squares
    via knights/bishops/king. Compare against python-chess oracle."""
    gg = GpuChessGame()
    state = gg.from_python_chess([chess.Board()])
    for color in (0, 1):
        got = _bb_to_int(attacks_by_color(state, color)[0])
        oracle = _attacks_oracle(chess.Board(), color == 0)
        assert got == oracle, (
            f"color={color}: starting attack bb diverges; "
            f"got 0x{got:016x} oracle 0x{oracle:016x}"
        )


def test_attacks_by_color_random_positions():
    """10k random positions, both colors. Bitboard equality with python-chess."""
    boards = _random_positions(10_000, seed=11)
    gg = GpuChessGame()
    state = gg.from_python_chess(boards)

    for color in (0, 1):
        got = attacks_by_color(state, color)
        for i, b in enumerate(boards):
            got_i = _bb_to_int(got[i])
            oracle = _attacks_oracle(b, color == 0)
            if got_i != oracle:
                # Localize the divergence for easier debugging.
                diff = got_i ^ oracle
                squares = [chess.square_name(s) for s in range(64) if (diff >> s) & 1]
                raise AssertionError(
                    f"position {i} (FEN={b.fen()}, color={color}): "
                    f"attack bbs differ at {squares}; "
                    f"got 0x{got_i:016x} oracle 0x{oracle:016x}"
                )


def test_attacks_by_color_with_kingless_occupancy():
    """King-x-ray: opponent attacks computed with our king masked out from occupancy
    must include the squares behind our king (which the king blocks otherwise)."""
    # Construct: white king on e1, black rook on e8, no other pieces between.
    # Without our king blocking, the rook attacks e1 and through it to nothing.
    # With our king masked, the rook should "see through" — but in this case,
    # nothing beyond e1, so behavior is identical. Use a different setup:
    # white king on e4, black rook on e8, white pawn on e2.
    # Normal occupancy: rook attacks e7, e6, e5 (blocked at king on e4).
    # Kingless occupancy: rook attacks e7, e6, e5, e4, e3, e2 (blocked at pawn on e2).
    b = chess.Board.empty()
    b.set_piece_at(chess.E4, chess.Piece(chess.KING, chess.WHITE))
    b.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))
    b.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.BLACK))  # required for board legality, not relevant
    b.set_piece_at(chess.A8, chess.Piece(chess.ROOK, chess.BLACK))  # avoid king-on-king-adjacent invalid issue
    b.set_piece_at(chess.E8, chess.Piece(chess.ROOK, chess.BLACK))

    gg = GpuChessGame()
    state = gg.from_python_chess([b])

    # Build kingless occupancy: full occupancy minus our king.
    pieces = state.pieces[0]
    occ = torch.zeros((), dtype=torch.int64)
    for p in range(12):
        occ = occ | pieces[p]
    occ_no_king = occ & ~pieces[5]  # remove white king (P_KING = 5, white base 0)

    attacks = attacks_by_color(state, color=1, occupancy=occ_no_king.unsqueeze(0))
    got = _bb_to_int(attacks[0])

    # The black rook on e8 should now attack e7, e6, e5, e4, e3, e2 (blocked at e2 pawn).
    expected_via_e_file = {chess.E7, chess.E6, chess.E5, chess.E4, chess.E3, chess.E2}
    for sq in expected_via_e_file:
        assert (got >> sq) & 1, f"expected attack on {chess.square_name(sq)} via x-ray"
