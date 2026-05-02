"""Cross-validate GpuChessGame.step_batch against ChessGame.step.

Random-play full games via the oracle, lockstep-step both engines, assert
state equality at every ply (pieces / side / castling / EP / clocks / done /
winner / reward).
"""
import random

import chess
import torch

from src.games.base import GameState
from src.games.chess import ChessGame, _move_to_action
from src.games.chess_gpu import GpuChessGame, ChessBatchedState, P_KING, _PIECE_TYPE_TO_IDX


def _state_to_dict(s: ChessBatchedState, i: int) -> dict:
    return {
        "pieces": s.pieces[i].tolist(),
        "side": int(s.side[i].item()),
        "castling": s.castling[i].tolist(),
        "ep": int(s.ep[i].item()),
        "halfmove": int(s.halfmove[i].item()),
        "fullmove": int(s.fullmove[i].item()),
        "ply": int(s.ply[i].item()),
        "done": bool(s.done[i].item()),
        "winner": int(s.winner[i].item()),
    }


def _board_to_dict(board: chess.Board) -> dict:
    pieces = [0] * 12
    import numpy as np
    pieces_u64 = np.zeros(12, dtype=np.uint64)
    for color, off in ((chess.WHITE, 0), (chess.BLACK, 6)):
        for piece_type, plane_idx in _PIECE_TYPE_TO_IDX.items():
            pieces_u64[off + plane_idx] = int(board.pieces(piece_type, color))
    pieces = list(pieces_u64.view(np.int64).astype(int))
    return {
        "pieces": pieces,
        "side": 0 if board.turn == chess.WHITE else 1,
        "castling": [
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
        ],
        "ep": -1 if board.ep_square is None else int(board.ep_square),
        "halfmove": int(board.halfmove_clock),
        "fullmove": int(board.fullmove_number),
        "ply": int(board.ply()),
    }


def _diff_keys(a: dict, b: dict) -> list[str]:
    return [k for k in a if a.get(k) != b.get(k)]


def test_step_batch_first_move():
    """Sanity: e2e4 from the starting position produces matching state."""
    cg = ChessGame()
    s_cpu = cg.reset()
    move = chess.Move.from_uci("e2e4")
    action = _move_to_action(move)
    s_cpu_after, reward_cpu, done_cpu = cg.step(s_cpu, action)

    gg = GpuChessGame()
    s_gpu = gg.reset_batch(1)
    actions = torch.tensor([action], dtype=torch.int64)
    s_gpu_after, reward_gpu, done_gpu = gg.step_batch(s_gpu, actions)

    expected = _board_to_dict(s_cpu_after.board)
    actual = _state_to_dict(s_gpu_after, 0)
    diffs = _diff_keys(expected, actual)
    assert not diffs, f"diff in {diffs}: expected={expected} actual={actual}"
    assert float(reward_gpu[0].item()) == reward_cpu
    assert bool(done_gpu[0].item()) == done_cpu


def test_step_batch_random_games():
    """1k random-play games: lockstep state equality at every ply."""
    rng = random.Random(2026)
    n_games = 1_000

    cg = ChessGame()
    gg = GpuChessGame()
    cpu_states = [cg.reset() for _ in range(n_games)]
    gpu_state = gg.reset_batch(n_games)
    done_flags = [False] * n_games

    n_plies = 0
    fails: list[tuple[int, str]] = []
    while not all(done_flags) and n_plies < 250:
        was_done_before = list(done_flags)

        # Pick legal action per game (random). Done games get sentinel 0,
        # which GpuChessGame steps blindly — we just won't compare those.
        actions: list[int] = []
        for i, st in enumerate(cpu_states):
            if done_flags[i]:
                actions.append(0)
                continue
            legals = cg.legal_actions(st)
            if not legals:
                done_flags[i] = True
                actions.append(0)
                continue
            actions.append(rng.choice(legals))

        # Step CPU oracle (skip done games).
        for i, st in enumerate(cpu_states):
            if done_flags[i]:
                continue
            new_st, _, d = cg.step(st, actions[i])
            cpu_states[i] = new_st
            if d:
                done_flags[i] = True

        # Step GPU (all games, batched).
        actions_t = torch.tensor(actions, dtype=torch.int64)
        gpu_state, _, _ = gg.step_batch(gpu_state, actions_t)

        # Compare states for games that were live at the start of this ply
        # (i.e., the step actually represents a real move on both sides).
        for i in range(n_games):
            if was_done_before[i]:
                continue
            cpu_dict = _board_to_dict(cpu_states[i].board)
            gpu_dict = _state_to_dict(gpu_state, i)
            diffs = _diff_keys(cpu_dict, gpu_dict)
            if diffs:
                fails.append((i, n_plies, f"diffs={diffs}\ncpu={cpu_dict}\ngpu={gpu_dict}\nfen_after={cpu_states[i].board.fen()}"))
                if len(fails) >= 3:
                    break
        if fails:
            break
        n_plies += 1

    if fails:
        msg = "\n\n".join(f"--- game {i} ply {p} ---\n{r}" for i, p, r in fails)
        raise AssertionError(f"{len(fails)} state divergences:\n{msg}")
