"""Cross-validate the GPU engine's insufficient-material draw against python-chess.

Regression test for bug_hunt_2026_06_13.md §C: the GPU engine's terminal set
(`_step_batch_impl`) was missing an insufficient-material term, so K-vs-K /
K+B-vs-K / K+N-vs-K endgames ran ~150-230 extra plies on GPU (winner=0 the whole
way), flooding the buffer with artificial draw labels and desyncing self-play
from the python-chess warmstart/eval labels.

python-chess `chess.Board.is_insufficient_material()` is the oracle. The GPU
`_insufficient_material` flag must be TRUE *exactly* when the oracle is TRUE.

SAFETY: 0 false positives is REQUIRED (never terminate a winnable game early).
The implementation is an exact match, so we also assert 0 false negatives.
"""
import random

import chess
import torch

from src.games.chess_gpu import GpuChessGame, _insufficient_material, _step_batch_impl


def _gpu_flag(boards: list[chess.Board]) -> torch.Tensor:
    gg = GpuChessGame()
    state = gg.from_python_chess(boards, device="cpu")
    return _insufficient_material(state)


def _oracle_flags(boards: list[chess.Board]) -> torch.Tensor:
    return torch.tensor([b.is_insufficient_material() for b in boards], dtype=torch.bool)


def _assert_match(boards: list[chess.Board], allow_false_negatives: bool = False):
    gpu = _gpu_flag(boards)
    oracle = _oracle_flags(boards)
    # False positive: GPU True, oracle False — NEVER allowed (early termination
    # of a possibly-winnable game).
    false_pos = gpu & ~oracle
    fp_idx = false_pos.nonzero(as_tuple=True)[0].tolist()
    assert not fp_idx, (
        "FALSE POSITIVE (GPU insufficient, python-chess says winnable):\n"
        + "\n".join(boards[i].fen() for i in fp_idx)
    )
    if not allow_false_negatives:
        false_neg = ~gpu & oracle
        fn_idx = false_neg.nonzero(as_tuple=True)[0].tolist()
        assert not fn_idx, (
            "FALSE NEGATIVE (python-chess insufficient, GPU missed):\n"
            + "\n".join(boards[i].fen() for i in fn_idx)
        )


# ---- constructed battery (each tagged with expected oracle outcome) --------

# (fen, expected_is_insufficient)
_CASES = [
    # bare-king endgames — insufficient.
    ("8/8/8/4k3/8/3K4/8/8 w - - 0 1", True),                 # KvK
    ("8/8/8/4k3/8/3K4/8/4N3 w - - 0 1", True),               # K+N vs K (white knight)
    ("8/8/8/4k3/8/3K4/8/4n3 w - - 0 1", True),               # K vs K+N (black knight)
    ("8/8/8/4k3/8/3K4/8/4B3 w - - 0 1", True),               # K+B vs K
    ("8/8/8/4k3/8/3K4/8/4b3 w - - 0 1", True),               # K vs K+B
    # KB vs KB, both bishops on the SAME square color — insufficient.
    ("8/8/8/2b1k3/8/2B1K3/8/8 w - - 0 1", True),
    # multiple same-color bishops one side, bare king other — insufficient.
    ("8/8/8/4k3/8/3K4/8/2B1B3 w - - 0 1", True),             # both light bishops? check below

    # --- NOT insufficient (must be False) ---
    # KB vs KB, bishops on OPPOSITE square colors — can (in principle) selfmate.
    ("8/8/8/3bk3/8/2B1K3/8/8 w - - 0 1", False),
    # K + 2 knights vs K — python-chess treats as sufficient (popcount > 2).
    ("8/8/8/4k3/8/3K4/8/3NN3 w - - 0 1", False),
    # K+N vs K+N — each side has a knight; opponent has non-king/non-queen
    # material (a knight) → selfmate material → sufficient.
    ("8/8/8/3nk3/8/3NK3/8/8 w - - 0 1", False),
    # any pawn — sufficient.
    ("8/4p3/8/4k3/8/3K4/8/8 w - - 0 1", False),
    # any rook — sufficient.
    ("8/8/8/4k3/8/3K4/8/4R3 w - - 0 1", False),
    # any queen — sufficient.
    ("8/8/8/4k3/8/3K4/8/4Q3 w - - 0 1", False),
    # starting position — sufficient.
    (chess.STARTING_FEN, False),
    # K+B vs K+N — bishop side, but a knight exists on the board → not the
    # bishops-only case → sufficient (knight branch on the N side too).
    ("8/8/8/3nk3/8/2B1K3/8/8 w - - 0 1", False),
]


def test_constructed_battery():
    boards = []
    expected = []
    for fen, exp in _CASES:
        b = chess.Board(fen)
        # sanity: our hand-labeled expectation matches python-chess itself.
        assert b.is_insufficient_material() == exp, (
            f"hand label wrong for {fen}: python-chess says "
            f"{b.is_insufficient_material()}, test expected {exp}"
        )
        boards.append(b)
        expected.append(exp)
    _assert_match(boards)


def test_kvk_terminates_via_step():
    """End-to-end: a KvK position must produce done=True (a draw) on the GPU
    step, not run on like before the fix."""
    gg = GpuChessGame()
    # K on e3 (white, to move) and k on e5; a quiet king move keeps it KvK.
    b = chess.Board("8/8/8/4k3/8/4K3/8/8 w - - 0 1")
    state = gg.from_python_chess([b], device="cpu")
    # Kd3 (e3 -> d3): action via the CPU oracle encoding.
    move = chess.Move(chess.E3, chess.D3)
    assert move in b.legal_moves
    from src.games.chess import _move_to_action
    action = _move_to_action(move, b.turn)
    new_state, _reward, done, _legal = _step_batch_impl(
        state, torch.tensor([action], dtype=torch.int64), gg.max_plies
    )
    assert bool(done[0]) is True
    assert int(new_state.winner[0]) == 0  # draw, not a win for either side


def _random_simplified_positions(n: int, seed: int) -> list[chess.Board]:
    """Random positions biased toward simplified endgames.

    Two generators: (a) random play from start (occasionally simplifies), and
    (b) explicitly sparse boards built by sprinkling a handful of pieces, which
    hits the insufficient-material region far more often than random play does.
    """
    rng = random.Random(seed)
    out: list[chess.Board] = []

    # (a) random play to a stopping point.
    while len(out) < n // 2:
        b = chess.Board()
        for _ in range(rng.randint(0, 120)):
            if b.is_game_over(claim_draw=False):
                break
            b.push(rng.choice(list(b.legal_moves)))
        out.append(b.copy())

    # (b) sparse constructed boards (kings + a few minor pieces / occasional
    # pawn) — exercises both the insufficient region and its boundary.
    piece_pool = [
        chess.Piece(chess.KNIGHT, chess.WHITE),
        chess.Piece(chess.KNIGHT, chess.BLACK),
        chess.Piece(chess.BISHOP, chess.WHITE),
        chess.Piece(chess.BISHOP, chess.BLACK),
        chess.Piece(chess.PAWN, chess.WHITE),
        chess.Piece(chess.ROOK, chess.BLACK),
    ]
    while len(out) < n:
        board = chess.Board.empty()
        squares = rng.sample(range(64), rng.randint(2, 6))
        # place the two kings on the first two squares, non-adjacent.
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        # find a non-adjacent square for the black king.
        bk = None
        for s in squares[1:]:
            if chess.square_distance(squares[0], s) >= 2:
                bk = s
                break
        if bk is None:
            continue
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        used = {squares[0], bk}
        for s in squares[1:]:
            if s in used:
                continue
            # don't put pawns on rank 1/8.
            piece = rng.choice(piece_pool)
            if piece.piece_type == chess.PAWN and chess.square_rank(s) in (0, 7):
                continue
            board.set_piece_at(s, piece)
            used.add(s)
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        if not board.is_valid():
            continue
        out.append(board)
    return out


def test_random_simplified_positions():
    boards = _random_simplified_positions(1200, seed=20260613)
    # sanity: the battery actually contains insufficient-material positions,
    # otherwise the test would pass trivially.
    n_insuff = sum(b.is_insufficient_material() for b in boards)
    assert n_insuff >= 20, f"too few insufficient positions sampled: {n_insuff}"
    _assert_match(boards)
