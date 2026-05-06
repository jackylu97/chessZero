"""Cross-validate GpuChessGame.step_batch against ChessGame.step.

Random-play full games via the oracle, lockstep-step both engines, assert
state + reward + done + winner equality at every ply (pieces / side /
castling / EP / clocks / done / winner / step-reward / step-done).
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
    action_stm = _move_to_action(move, s_cpu.board.turn)
    s_cpu_after, reward_cpu, done_cpu = cg.step(s_cpu, action_stm)

    gg = GpuChessGame()
    s_gpu = gg.reset_batch(1)
    actions = torch.tensor([action_stm], dtype=torch.int64)
    s_gpu_after, reward_gpu, done_gpu = gg.step_batch(s_gpu, actions)

    expected = _board_to_dict(s_cpu_after.board)
    actual = _state_to_dict(s_gpu_after, 0)
    diffs = _diff_keys(expected, actual)
    assert not diffs, f"diff in {diffs}: expected={expected} actual={actual}"
    assert float(reward_gpu[0].item()) == reward_cpu
    assert bool(done_gpu[0].item()) == done_cpu


def test_step_batch_random_games():
    """1k random-play games: lockstep state + reward/done/winner equality at every ply.

    GpuChessGame mirrors mate/stalemate/threefold/75-move/ply-cap. Other
    python-chess auto-terminations (insufficient material, fivefold) aren't
    modeled on GPU because tracking them isn't worth the cost — random play
    hits them rarely and at worst the GPU keeps shuffling a drawn position
    until another terminal fires. We drop those CPU-only terminations from
    the lockstep so any real divergence on a GPU-tracked terminal is still
    caught.
    """
    import chess as _chess  # avoid shadowing module-level import in pyflakes

    rng = random.Random(2026)
    n_games = 1_000

    cg = ChessGame()
    gg = GpuChessGame()
    cpu_states = [cg.reset() for _ in range(n_games)]
    gpu_state = gg.reset_batch(n_games)
    done_flags = [False] * n_games
    drop_from_lockstep = [False] * n_games  # CPU-only terminations the GPU doesn't track

    n_plies = 0
    fails: list[tuple[int, int, str]] = []
    while not all(done_flags) and n_plies < 250:
        was_done_before = list(done_flags)

        # Pick legal action per game (random). Done games get sentinel 0,
        # which GpuChessGame steps blindly — we just won't compare those.
        # Encode moves in both STM (for CPU) and absolute (for GPU) forms.
        actions_stm: list[int] = []
        actions_abs: list[int] = []
        for i, st in enumerate(cpu_states):
            if done_flags[i]:
                actions_stm.append(0)
                actions_abs.append(0)
                continue
            legal_moves = list(st.board.legal_moves)
            if not legal_moves:
                done_flags[i] = True
                actions_stm.append(0)
                actions_abs.append(0)
                continue
            move = rng.choice(legal_moves)
            action = _move_to_action(move, st.board.turn)
            actions_stm.append(action)
            actions_abs.append(action)

        # Step CPU oracle (skip done games), track per-game (reward, done).
        cpu_reward = [0.0] * n_games
        cpu_done_now = [False] * n_games
        for i, st in enumerate(cpu_states):
            if done_flags[i]:
                continue
            new_st, r, d = cg.step(st, actions_stm[i])
            cpu_states[i] = new_st
            cpu_reward[i] = r
            cpu_done_now[i] = d
            if d:
                done_flags[i] = True
                # Drop from lockstep if the CPU terminated for a reason GPU
                # doesn't model. Tracked: checkmate / stalemate / threefold /
                # 75-move (auto) / ply-cap. Untracked: insufficient material,
                # fivefold repetition.
                outcome = new_st.board.outcome(claim_draw=False)
                gpu_tracked = (
                    outcome is not None
                    and outcome.termination in (
                        _chess.Termination.CHECKMATE,
                        _chess.Termination.STALEMATE,
                        _chess.Termination.SEVENTYFIVE_MOVES,
                        _chess.Termination.THREEFOLD_REPETITION,
                    )
                ) or new_st.board.is_repetition(3) or new_st.board.ply() >= cg.max_plies
                if not gpu_tracked:
                    drop_from_lockstep[i] = True

        # Step GPU (all games, batched).
        actions_t = torch.tensor(actions_abs, dtype=torch.int64)
        new_gpu_state, gpu_reward_t, gpu_done_t = gg.step_batch(gpu_state, actions_t)
        gpu_state = new_gpu_state

        # Compare for games that were live at the start of this ply.
        for i in range(n_games):
            if was_done_before[i] or drop_from_lockstep[i]:
                continue
            cpu_dict = _board_to_dict(cpu_states[i].board)
            gpu_dict = _state_to_dict(gpu_state, i)
            diffs = _diff_keys(cpu_dict, gpu_dict)
            # Compare done sticky-flag (after this step both should agree).
            cpu_done_state = cpu_states[i].done
            gpu_done_state = bool(gpu_state.done[i].item())
            cpu_winner_state = cpu_states[i].winner
            gpu_winner_state = int(gpu_state.winner[i].item())
            # Compare reward/done returned by the step call itself.
            r_gpu = float(gpu_reward_t[i].item())
            d_gpu = bool(gpu_done_t[i].item())
            extra: list[str] = []
            if r_gpu != cpu_reward[i]:
                extra.append(f"reward CPU={cpu_reward[i]} vs GPU={r_gpu}")
            if d_gpu != cpu_done_now[i]:
                extra.append(f"done(step) CPU={cpu_done_now[i]} vs GPU={d_gpu}")
            if cpu_done_state != gpu_done_state:
                extra.append(f"done(state) CPU={cpu_done_state} vs GPU={gpu_done_state}")
            if cpu_winner_state != gpu_winner_state:
                extra.append(f"winner CPU={cpu_winner_state} vs GPU={gpu_winner_state}")
            if diffs or extra:
                fails.append((i, n_plies,
                    f"state_diffs={diffs}\nextras={extra}\n"
                    f"cpu={cpu_dict}\ngpu={gpu_dict}\nfen_after={cpu_states[i].board.fen()}"
                ))
                if len(fails) >= 3:
                    break
        if fails:
            break
        n_plies += 1

    if fails:
        msg = "\n\n".join(f"--- game {i} ply {p} ---\n{r}" for i, p, r in fails)
        raise AssertionError(f"{len(fails)} divergences:\n{msg}")
