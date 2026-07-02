"""Tablebase-optimal playouts → GameHistory-compatible trajectories.

Shared by two consumers (strategy_2026_07_02.md):
- ``scripts/gen_tb_anchor_games.py`` — the supervised endgame ANCHOR:
  demonstration games from random decisive ≤5-man seeds, injected into the
  rolling buffer on an interval (never annealed away).
- the self-play TB ROLLOUT FILL (``tb_rollout_fill``) — a won-but-unconverted
  self-play game is truncated at its first decisive in-TB ply and finished with
  TB-optimal play by BOTH sides, so (a) the stored ``game_outcome`` becomes the
  TRUE result for the ENTIRE trajectory (the win-adjudication z the value
  relabel can't propagate to pre-TB plies) and (b) the appended tail is an
  on-distribution conversion demonstration for the policy head.

Move policy: the winning side mates immediately when possible, else plays the
win-preserving move minimizing the opponent's DTZ (preferring zeroing moves
when the halfmove clock runs high — respects the 50-move rule); the losing side
plays maximum-DTZ resistance. Per-ply targets mirror the deferred-relabel
semantics exactly (same ``SyzygyRootProber`` methods):

  policies             <- soft win-preserving DTZ-softmax (_classify_to_policy);
                          one-hot on the played move where no winning move
                          exists (the losing side's plies)
  root_values          <- STM-POV DTZ-shaped position value (_position_value)
  tablebase_values     <- same (feeds the tb_value_weight blend / tb_value_hard)
  tablebase_moves_left <- |DTM| plies-to-mate, Gaviota (tb_moves_left_weight)
  tablebase_policy     <- the soft policy tuple, None where no winning move
  rewards              <- +1 mover-POV on the mating transition, else 0
                          (chess_gpu convention: mate sets reward=+1)
"""
from __future__ import annotations

import chess
import chess.syzygy
import numpy as np

from src.games.chess import _move_to_action


class TBPlayoutError(Exception):
    """Playout could not be completed faithfully (missing table, 50-move risk,
    non-decisive entry). Callers skip the game rather than store bad targets."""


def _probe_wdl(tb, board: chess.Board) -> int:
    try:
        return int(tb.probe_wdl(board))
    except (KeyError, ValueError, chess.syzygy.MissingTableError) as e:
        raise TBPlayoutError(f"WDL probe failed: {e}") from e


def _probe_dtz(tb, board: chess.Board) -> int:
    try:
        return abs(int(tb.probe_dtz(board)))
    except (KeyError, ValueError, chess.syzygy.MissingTableError) as e:
        raise TBPlayoutError(f"DTZ probe failed: {e}") from e


def _pick_move(tb, board: chess.Board) -> chess.Move:
    """TB-optimal move for the side to move. Winner: mate-now > min-DTZ
    win-preserving (zeroing preferred at a high halfmove clock). Loser:
    max-DTZ resistance. Raises TBPlayoutError on probe failure."""
    mover_wdl = _probe_wdl(tb, board)
    winning = mover_wdl >= 2
    best_mv, best_key = None, None
    for mv in board.legal_moves:
        zeroing = board.is_zeroing(mv)
        board.push(mv)
        try:
            if board.is_checkmate():
                return mv  # mate now beats everything (finally pops first)
            if board.is_stalemate() or board.is_insufficient_material():
                # A drawing move: never optimal for the winner; impossible to
                # reach here for the loser (a draw in hand => not lost).
                continue
            child_wdl = _probe_wdl(tb, board)  # opponent POV after the move
            if winning:
                if child_wdl > -2:
                    continue  # throws the win
                dtz = _probe_dtz(tb, board)
                # ALWAYS prefer win-keeping ZEROING moves (pawn push / capture):
                # DTZ counts to the next zeroing event, so plain min-child-DTZ
                # shuffles forever at cur_dtz==1 (the child's DTZ stays small
                # because the zeroing move "remains available") — the classic
                # min-DTZ repetition trap. A zeroing move completes the phase
                # NOW, strictly consumes material/pawn-advance budget (finite),
                # and resets the 50-move clock; min child DTZ breaks ties.
                key = (0 if zeroing else 1, dtz)
            else:
                # Everything loses (position is lost); resist longest.
                dtz = _probe_dtz(tb, board)
                key = (-dtz,)
            if best_key is None or key < best_key:
                best_mv, best_key = mv, key
        finally:
            board.pop()
    if best_mv is None:
        raise TBPlayoutError("no playable TB move found")
    return best_mv


def tb_playout(board: chess.Board, prober, max_plies: int = 180) -> dict:
    """Play TB-optimal chess from ``board`` (which must be a DECISIVE in-TB
    position) until checkmate. Returns a dict of per-ply target lists (see
    module docstring) plus ``winner`` (chess.WHITE / chess.BLACK):

        actions, policies, root_values, rewards,
        tablebase_values, tablebase_moves_left, tablebase_policy, winner

    ``board`` is consumed (mutated to the terminal position). Raises
    TBPlayoutError when the playout can't be completed faithfully.
    """
    tb = prober.tb
    entry_wdl = _probe_wdl(tb, board)
    if abs(entry_wdl) < 2:
        raise TBPlayoutError(f"entry position not decisive (wdl={entry_wdl})")
    # 50-move-rule guard: syzygy WDL ignores the live halfmove clock, so a
    # "won" position with a nearly-spent clock may be a real draw. DTZ-optimal
    # play zeroes within DTZ plies, so hmc + DTZ < 100 is safe.
    if board.halfmove_clock + _probe_dtz(tb, board) > 99:
        raise TBPlayoutError("50-move-rule risk at entry (clock + DTZ > 99)")
    winner = board.turn if entry_wdl >= 2 else (not board.turn)

    actions: list[int] = []
    policies: list = []
    root_values: list[float] = []
    rewards: list[float] = []
    tb_values: list[float] = []
    tb_ml: list[float] = []
    tb_policy: list = []

    for _ in range(max_plies):
        if board.is_game_over():
            break
        # Per-ply targets — the exact deferred-relabel semantics.
        pv, ml, pol = prober.relabel_position(board, want_policy=True)
        mv = _pick_move(tb, board)
        action = _move_to_action(mv, board.turn)
        if pol is None:
            # No winning move (the losing side's ply): one-hot on the played
            # max-resistance move — the best defensive target available.
            pol_target = (np.asarray([action], dtype=np.int32),
                          np.asarray([1.0], dtype=np.float32))
        else:
            pol_target = (np.asarray(pol[0], dtype=np.int32),
                          np.asarray(pol[1], dtype=np.float32))
        board.push(mv)
        actions.append(int(action))
        policies.append(pol_target)
        root_values.append(float(pv) if pv == pv else 0.0)
        rewards.append(1.0 if board.is_checkmate() else 0.0)
        tb_values.append(float(pv))
        tb_ml.append(float(ml))
        tb_policy.append(pol)

    if not board.is_checkmate():
        raise TBPlayoutError(
            f"playout did not end in mate after {len(actions)} plies "
            f"(result={board.result(claim_draw=True)})")
    if (not board.turn) != winner:
        raise TBPlayoutError("mated side is not the predicted loser")

    return {
        "actions": actions,
        "policies": policies,
        "root_values": root_values,
        "rewards": rewards,
        "tablebase_values": tb_values,
        "tablebase_moves_left": tb_ml,
        "tablebase_policy": tb_policy,
        "winner": winner,
    }
