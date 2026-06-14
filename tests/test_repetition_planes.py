"""Repetition-count + no-progress observation planes (AlphaZero, Silver 2018).

Three constant planes appended after the move-count plane:
  19: current position occurred >= 2 times total (>= 1 time before)
  20: current position occurred >= 3 times total (>= 2 times before)
  21: halfmove_clock / 100 (no-progress / 50-move clock, UNCLAMPED)

They are STM-invariant scalars broadcast over the board, NOT rank-flipped for
black-to-move. These tests step a knight-shuffle repetition line through BOTH
the CPU (`ChessGame`) and GPU (`GpuChessGame`) engines move-for-move and assert:
  - plane 19/20 match python-chess `board.is_repetition(2)/(3)` exactly,
  - the GPU planes are byte-identical to the CPU planes,
  - repetition planes RESET after an irreversible move (pawn move / capture),
  - the no-progress plane equals halfmove_clock/100 on both engines,
  - the three new planes are STM-invariant (not flipped for black).
"""
import chess
import torch

from src.games.base import GameState
from src.games.chess import ChessGame, _move_to_action
from src.games.chess_gpu import GpuChessGame

REP2, REP3, NOPROG = 19, 20, 21


def test_num_planes_is_22():
    assert ChessGame.num_planes == 22
    assert GpuChessGame.num_planes == 22


class _DualEngine:
    """Drive CPU + GPU engines and a python-chess oracle board in lockstep."""

    def __init__(self):
        self.cpu = ChessGame()
        self.gpu = GpuChessGame()
        self.cpu_state = self.cpu.reset()
        self.gpu_state = self.gpu.reset_batch(1, device="cpu")
        self.oracle = chess.Board()

    def push_uci(self, uci: str):
        move = chess.Move.from_uci(uci)
        # Action index is STM-relative; encode from the oracle's side to move.
        action = _move_to_action(move, self.oracle.turn)
        self.cpu_state, _, _ = self.cpu.step(self.cpu_state, action)
        actions = torch.tensor([action], dtype=torch.int64)
        self.gpu_state, _, _ = self.gpu.step_batch(self.gpu_state, actions)
        self.oracle.push(move)

    def obs(self):
        cpu_obs = self.cpu.to_tensor(self.cpu_state)              # (22, 8, 8)
        gpu_obs = self.gpu.to_tensor_batch(self.gpu_state)[0]     # (22, 8, 8)
        return cpu_obs, gpu_obs

    def assert_parity_and_oracle(self, ply_tag: str):
        cpu_obs, gpu_obs = self.obs()
        # CPU/GPU byte-identical across ALL planes.
        assert torch.equal(cpu_obs, gpu_obs), (
            f"[{ply_tag}] CPU/GPU obs diverge; planes "
            f"{(cpu_obs != gpu_obs).any(dim=(1, 2)).nonzero(as_tuple=True)[0].tolist()}; "
            f"FEN={self.oracle.fen()}"
        )
        # Repetition planes match python-chess exactly.
        exp2 = 1.0 if self.oracle.is_repetition(2) else 0.0
        exp3 = 1.0 if self.oracle.is_repetition(3) else 0.0
        assert cpu_obs[REP2].eq(exp2).all(), f"[{ply_tag}] plane19 != is_repetition(2)={exp2}"
        assert cpu_obs[REP3].eq(exp3).all(), f"[{ply_tag}] plane20 != is_repetition(3)={exp3}"
        # No-progress plane = halfmove_clock / 100.
        exp_np = self.oracle.halfmove_clock / 100.0
        assert torch.allclose(cpu_obs[NOPROG], torch.full((8, 8), exp_np)), (
            f"[{ply_tag}] plane21 != halfmove/100={exp_np}"
        )
        return cpu_obs, gpu_obs


# Knight-shuffle line: the post-move position after each full 4-ply cycle
# (Nf3 Nf6 Ng1 Ng8) returns to the start position with the same side to move.
# This reaches the start position the 1st, 2nd, 3rd ... time so both rep planes
# get exercised.
_SHUFFLE = ["g1f3", "g8f6", "f3g1", "f6g8"]


def test_repetition_planes_knight_shuffle_cpu_gpu_parity():
    eng = _DualEngine()

    # Cycle 1: after 4 plies we're back at the start position (2nd occurrence).
    for uci in _SHUFFLE:
        eng.push_uci(uci)
    cpu_obs, _ = eng.assert_parity_and_oracle("cycle1-end")
    assert eng.oracle.is_repetition(2)
    assert not eng.oracle.is_repetition(3)
    assert cpu_obs[REP2].eq(1.0).all()   # occurred 2x total -> plane19 set
    assert cpu_obs[REP3].eq(0.0).all()

    # Cycle 2: start position reached the 3rd time -> plane20 (is_repetition(3)).
    for uci in _SHUFFLE:
        eng.push_uci(uci)
    cpu_obs, _ = eng.assert_parity_and_oracle("cycle2-end")
    assert eng.oracle.is_repetition(3)
    assert cpu_obs[REP2].eq(1.0).all()
    assert cpu_obs[REP3].eq(1.0).all()   # occurred 3x total -> plane20 set


def test_repetition_planes_reset_after_irreversible_move():
    """A pawn move clears the repetition window on both engines.

    Uses a single shuffle cycle (2-fold, plane19 set, game NOT terminal) so the
    irreversible move is applied to a live game — a 3-fold would auto-terminate
    and freeze the board, making any subsequent move a no-op.
    """
    eng = _DualEngine()
    # One shuffle cycle -> start position reached the 2nd time (2-fold).
    for uci in _SHUFFLE:
        eng.push_uci(uci)
    cpu_obs, _ = eng.assert_parity_and_oracle("pre-reset")
    assert cpu_obs[REP2].eq(1.0).all()
    assert cpu_obs[REP3].eq(0.0).all()
    assert cpu_obs[NOPROG].eq(4 / 100.0).all()   # 4 reversible plies
    assert not eng.oracle.is_game_over(claim_draw=False)

    # Irreversible pawn move: repetition window resets, halfmove clock resets.
    eng.push_uci("e2e4")
    cpu_obs, _ = eng.assert_parity_and_oracle("post-reset")
    assert cpu_obs[REP2].eq(0.0).all()
    assert cpu_obs[REP3].eq(0.0).all()
    assert cpu_obs[NOPROG].eq(0.0).all()         # pawn move reset clock
    assert not eng.oracle.is_repetition(2)


def test_repetition_planes_reset_after_capture():
    """A capture (irreversible) also clears the repetition window.

    Single bishop-shuffle cycle (2-fold, not terminal) before the capture.
    """
    eng = _DualEngine()
    # Open lines so a knight capture on e5 exists.
    setup = ["e2e4", "e7e5", "g1f3", "g8f6"]
    for uci in setup:
        eng.push_uci(uci)
    # One bishop-shuffle cycle -> repeats the post-setup position once (2-fold).
    shuffle = ["f1c4", "f8c5", "c4f1", "c5f8"]
    for uci in shuffle:
        eng.push_uci(uci)
    cpu_obs, _ = eng.assert_parity_and_oracle("pre-capture")
    assert cpu_obs[REP2].eq(1.0).all()
    assert cpu_obs[REP3].eq(0.0).all()
    assert not eng.oracle.is_game_over(claim_draw=False)

    # Knight capture Nxe5 — irreversible, resets repetition + halfmove clock.
    eng.push_uci("f3e5")
    cpu_obs, _ = eng.assert_parity_and_oracle("post-capture")
    assert cpu_obs[REP2].eq(0.0).all()
    assert cpu_obs[REP3].eq(0.0).all()
    assert cpu_obs[NOPROG].eq(0.0).all()


def test_no_progress_plane_increments_and_resets():
    """plane21 == halfmove_clock/100 across reversible and irreversible moves."""
    eng = _DualEngine()
    # Reversible knight moves increment the clock by 1 each ply.
    seq = ["g1f3", "g8f6", "b1c3", "b8c6"]   # 4 reversible plies
    for i, uci in enumerate(seq, start=1):
        eng.push_uci(uci)
        cpu_obs, gpu_obs = eng.assert_parity_and_oracle(f"reversible-{i}")
        assert cpu_obs[NOPROG].eq(i / 100.0).all()
        assert gpu_obs[NOPROG].eq(i / 100.0).all()

    # Irreversible pawn push resets the clock to 0.
    eng.push_uci("e2e4")
    cpu_obs, gpu_obs = eng.assert_parity_and_oracle("irreversible")
    assert cpu_obs[NOPROG].eq(0.0).all()
    assert gpu_obs[NOPROG].eq(0.0).all()


def test_new_planes_are_stm_invariant_not_flipped():
    """The three new planes (19-21) must NOT be rank-flipped for black-to-move.

    A white-to-move position and the colour/rank-mirrored black-to-move analogue
    with the same repetition/clock state produce identical planes 19-21.
    """
    cpu = ChessGame()
    gpu = GpuChessGame()

    # White-to-move position with a non-trivial halfmove clock (no repetition).
    wb = chess.Board()
    for uci in ["g1f3", "g8f6", "b1c3", "b8c6"]:   # halfmove_clock = 4, white to move
        wb.push(chess.Move.from_uci(uci))
    assert wb.turn == chess.WHITE and wb.halfmove_clock == 4

    # Black-to-move analogue with a different (still non-trivial) clock.
    bb = chess.Board()
    for uci in ["g1f3", "g8f6", "b1c3", "b8c6", "f3g1"]:  # halfmove 5, BLACK to move
        bb.push(chess.Move.from_uci(uci))
    assert bb.turn == chess.BLACK and bb.halfmove_clock == 5

    def planes(board):
        cp = 1 if board.turn == chess.WHITE else -1
        c = cpu.to_tensor(GameState(board=board, current_player=cp))
        state = gpu.from_python_chess([board], device="cpu")
        g = gpu.to_tensor_batch(state)[0]
        return c, g

    cw, gw = planes(wb)
    cbk, gbk = planes(bb)

    # CPU/GPU agree on the no-progress plane for both (history-independent).
    assert torch.equal(cw[NOPROG], gw[NOPROG])
    assert torch.equal(cbk[NOPROG], gbk[NOPROG])

    # The no-progress plane reflects halfmove/100 regardless of side to move and
    # is a flat broadcast (every cell equal) -> trivially not rank-flipped.
    assert cw[NOPROG].eq(4 / 100.0).all()
    assert cbk[NOPROG].eq(5 / 100.0).all()

    # Repetition planes are 0 here; flat broadcasts, identical under rank flip.
    for p in (REP2, REP3):
        assert cw[p].eq(0.0).all() and cbk[p].eq(0.0).all()
        assert gw[p].eq(0.0).all() and gbk[p].eq(0.0).all()

    # Direct STM-invariance check: planes 19-21 are constant across the board,
    # so a rank flip (row reversal) leaves them unchanged on both engines.
    for p in (REP2, REP3, NOPROG):
        for obs in (cw, cbk, gw, gbk):
            assert torch.equal(obs[p], obs[p].flip(0)), f"plane {p} is not flat/STM-invariant"
