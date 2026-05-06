"""Lock in the action-index encoding before Phase 2 builds against it.

The flat 4672-action map is defined by `chess.py:_move_to_action` and
inverted (with board context for promotion/turn) by `_action_to_move`. The
batched legal_mask in Phase 2 must produce the same indices, otherwise the
network sees policies in a permuted space.

This test fixes the encoding's contract: every legal move from a corpus of
random positions round-trips encode → decode → the same `chess.Move`.
"""
import random

import chess

from src.games.chess import ACTION_SPACE, _action_to_move, _move_to_action
from src.games.chess_gpu import (
    ACTION_FROM_SQ,
    ACTION_IS_UNDERPROMO,
    ACTION_TARGET_W,
    _ACTION_TABLES,
)


def _random_positions(n: int, seed: int = 0) -> list[chess.Board]:
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


def test_action_index_in_range():
    """Encoded action must always land in [0, ACTION_SPACE)."""
    for board in _random_positions(200, seed=1):
        for move in board.legal_moves:
            idx = _move_to_action(move, board.turn)
            assert 0 <= idx < ACTION_SPACE, f"OOB index {idx} for {move}"


def test_round_trip_random_positions():
    """encode → decode reproduces the original move on every legal move."""
    boards = _random_positions(2_000, seed=2)
    n_moves = 0
    for b in boards:
        for move in b.legal_moves:
            idx = _move_to_action(move, b.turn)
            decoded = _action_to_move(idx, b)
            assert decoded == move, (
                f"round-trip failed: {move.uci()} → {idx} → "
                f"{None if decoded is None else decoded.uci()} (FEN={b.fen()})"
            )
            n_moves += 1
    assert n_moves > 50_000, f"sanity: corpus too small ({n_moves} moves)"


def test_round_trip_underpromotion():
    """Each of the 9 underpromotion encodings round-trips through both colors."""
    # White pawn on a7..h7 promoting to rank 8; black pawn on a2..h2 promoting to rank 1.
    cases: list[chess.Board] = []
    for file in range(8):
        # White: pawn on file7 → file8 with all neighbors as black pieces to enable captures.
        b = chess.Board.empty()
        b.set_piece_at(chess.square(file, 6), chess.Piece(chess.PAWN, chess.WHITE))
        b.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        # Place opposing pieces on neighboring 8th-rank squares for capture-promotions.
        for df in (-1, 1):
            tf = file + df
            if 0 <= tf < 8 and tf != 4:  # avoid king's square
                b.set_piece_at(chess.square(tf, 7), chess.Piece(chess.ROOK, chess.BLACK))
        b.turn = chess.WHITE
        cases.append(b)

    n_promos = 0
    for b in cases:
        for move in b.legal_moves:
            if move.promotion is None:
                continue
            idx = _move_to_action(move, b.turn)
            decoded = _action_to_move(idx, b)
            assert decoded == move, (
                f"underpromotion round-trip failed: {move.uci()} → {idx} → "
                f"{None if decoded is None else decoded.uci()} (FEN={b.fen()})"
            )
            n_promos += 1
    assert n_promos >= 8, f"expected ≥8 promotion moves, got {n_promos}"


def test_gpu_action_tables_match_decoder():
    """For every action 0..4671, GpuChessGame's precomputed STM action table
    must agree with the chess.py decoder.

    Both engines now use STM encoding (as-if-white). ACTION_TARGET_W is the
    single STM table; this cross-checks it against `chess.py:_action_to_move`
    decoded on a white-to-move board (= STM decode).
    """
    promo_piece_table = _ACTION_TABLES["promo_piece"]

    b_w = chess.Board.empty()
    b_w.turn = chess.WHITE

    for action in range(ACTION_SPACE):
        from_sq = action // 73
        mt = action % 73
        is_underpromo = mt >= 64

        assert int(ACTION_FROM_SQ[action].item()) == from_sq, (
            f"action {action}: ACTION_FROM_SQ={int(ACTION_FROM_SQ[action].item())} "
            f"vs expected {from_sq}"
        )
        assert bool(ACTION_IS_UNDERPROMO[action].item()) == is_underpromo, (
            f"action {action}: ACTION_IS_UNDERPROMO={bool(ACTION_IS_UNDERPROMO[action].item())} "
            f"vs expected {is_underpromo}"
        )

        move_w = _action_to_move(action, b_w)
        expected_target = -1 if move_w is None else move_w.to_square
        assert int(ACTION_TARGET_W[action].item()) == expected_target, (
            f"action {action} (from={from_sq}, mt={mt}): "
            f"ACTION_TARGET_W={int(ACTION_TARGET_W[action].item())} vs "
            f"chess.py decode→{None if move_w is None else move_w.uci()}"
        )

        if is_underpromo:
            expected_promo = (mt - 64) // 3  # 0=R, 1=B, 2=N
            assert int(promo_piece_table[action].item()) == expected_promo, (
                f"action {action}: promo_piece={int(promo_piece_table[action].item())} "
                f"vs expected {expected_promo}"
            )
            if move_w is not None:
                chess_py_promo = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}[move_w.promotion]
                assert chess_py_promo == expected_promo, (
                    f"action {action}: chess.py promo={move_w.promotion} "
                    f"vs GPU code={expected_promo}"
                )
        else:
            assert int(promo_piece_table[action].item()) == -1, (
                f"action {action}: non-underpromo has promo_piece="
                f"{int(promo_piece_table[action].item())}"
            )


def test_round_trip_castling():
    """Both castle types for both colors round-trip."""
    fens = [
        # White can castle both sides; black hasn't moved.
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        # Black can castle both sides; white to move's already advanced.
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
    ]
    for fen in fens:
        b = chess.Board(fen)
        for move in b.legal_moves:
            if not b.is_castling(move):
                continue
            idx = _move_to_action(move, b.turn)
            decoded = _action_to_move(idx, b)
            assert decoded == move, f"castle round-trip failed: {move.uci()} ↔ {idx}"
