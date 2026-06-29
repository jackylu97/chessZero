"""Decisive draw-collapse trace.

Loads checkpoint_6000.pt (chess_small, CPU), builds KNOWN-WON positions with
proper 8-frame history, and runs the numpy MCTS (src/mcts/mcts.py). Dumps:
  - raw WDL probs + scalar root value (value head verdict)
  - per-child Q (= reward + discount*child_value) and visit counts
  - argmax-visit move + its Q
  - MinMaxStats min/max
  - whether a repetition-leading move gets a phantom-high backed-up value

Question: does the value head SAY win on KQvK? If yes but MCTS prefers a
repeating/non-progress move -> SMOKING GUN. Else fundamental target issue.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import chess

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _move_to_action, _action_to_move
from src.model.utils import wdl_to_scalar
from src.training.replay_buffer import stack_with_history
from scripts.eval_checkpoint_health import build_network

torch.serialization.add_safe_globals([MuZeroConfig])

CKPT = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"


def build_obs_with_history(game, board, history_frames):
    """Build an 8-frame stacked obs by replaying the board's move stack so the
    repetition / no-progress planes are populated correctly. Returns (obs, legal)."""
    # Reconstruct per-ply single-frame observations from the move stack.
    moves = list(board.move_stack)
    root = chess.Board()
    # If board has a custom start (e.g. from FEN), python-chess move_stack starts
    # from board's root position. Use board.root() if available.
    try:
        root = board.root()
    except Exception:
        root = chess.Board()
    frames = []  # chronological single-frame obs, oldest first
    st = game.reset()
    st.board = root.copy()
    cur_board = root.copy()
    frames.append(game.to_tensor(_state_for(game, cur_board)))
    for mv in moves:
        cur_board.push(mv)
        frames.append(game.to_tensor(_state_for(game, cur_board)))
    current = frames[-1]
    prior = frames[:-1]
    obs = stack_with_history(current, prior, history_frames)
    state = _state_for(game, board)
    legal = game.legal_actions(state)
    return obs, legal, state


def _state_for(game, board):
    st = game.reset()
    st.board = board.copy()
    st.current_player = 1 if board.turn == chess.WHITE else -1
    return st


@torch.no_grad()
def value_head_verdict(net, obs, cfg):
    h = net.representation(obs.unsqueeze(0))
    pl, vl = net.prediction(h)
    wdl = F.softmax(vl.float(), -1)[0].numpy()
    v = wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).item()
    v_nodraw = wdl_to_scalar(vl.float(), draw_score=0.0).item()
    return wdl, v, v_nodraw, pl[0].numpy()


def trace_position(net, game, cfg, name, board, note=""):
    print(f"\n{'='*78}\n{name}\n  FEN: {board.fen()}\n  {note}\n{'='*78}")
    obs, legal, state = build_obs_with_history(game, board, cfg.history_frames)
    wdl, v, v_nodraw, pl = value_head_verdict(net, obs, cfg)
    print(f"  VALUE HEAD (root, side-to-move POV): WDL=(W={wdl[0]:.3f}, D={wdl[1]:.3f}, L={wdl[2]:.3f})")
    print(f"    scalar V (draw_score={cfg.draw_score}) = {v:+.4f}   |  V (draw_score=0) = {v_nodraw:+.4f}")

    # Top policy-prior legal moves
    order = np.argsort(-pl)
    top_pol = []
    for ai in order:
        mv = _action_to_move(int(ai), board)
        if mv is not None and mv in board.legal_moves:
            top_pol.append((board.san(mv), int(ai)))
        if len(top_pol) >= 6:
            break
    print(f"  TOP POLICY PRIORS (legal): {[(s, round(float(F.softmax(torch.tensor(pl),0)[a].item()),3)) for s,a in top_pol]}")

    # Run numpy MCTS
    from src.mcts.mcts import MCTS, MinMaxStats
    mcts = MCTS(net, game, cfg, "cpu")
    root = mcts.run(obs, legal, add_noise=False)

    print(f"  MCTS root.value (backed up, STM POV) = {root.value:+.4f}   visits={root.visit_count}")

    visits = root.child_visits
    rewards = root.child_rewards
    vsums = root.child_value_sums
    actions = root.child_actions
    discount = cfg.discount

    with np.errstate(invalid="ignore", divide="ignore"):
        child_vals = np.where(visits > 0, vsums / np.maximum(visits, 1.0), 0.0)
    raw_q = rewards - discount * child_vals  # mover-POV Q

    order = np.argsort(-visits)
    print(f"  TOP CHILDREN BY VISITS (Q = reward - discount*child_value, mover POV):")
    print(f"    {'move':9s} {'visits':>7s} {'Q':>9s} {'reward':>8s} {'child_v':>9s} {'prior':>7s} {'rep?':>5s}")
    picked_san = None
    for rank, j in enumerate(order[:8]):
        a = int(actions[j])
        mv = _action_to_move(a, board)
        legal_mv = mv is not None and mv in board.legal_moves
        san = board.san(mv) if legal_mv else f"a{a}(ill)"
        rep_flag = ""
        if legal_mv:
            nb = board.copy(); nb.push(mv)
            if nb.is_repetition(3):
                rep_flag = "3FOLD"
            elif nb.is_repetition(2):
                rep_flag = "2rep"
        if rank == 0:
            picked_san = san
        print(f"    {san:9s} {visits[j]:7.0f} {raw_q[j]:+9.4f} {rewards[j]:+8.4f} {child_vals[j]:+9.4f} {root.child_priors[j]:7.3f} {rep_flag:>5s}")

    # Is the argmax-visit move legal? Does it make progress?
    best_j = int(order[0])
    best_a = int(actions[best_j])
    best_mv = _action_to_move(best_a, board)
    best_legal = best_mv is not None and best_mv in board.legal_moves
    print(f"  PICKED (argmax visits): {picked_san}  legal={best_legal}  Q={raw_q[best_j]:+.4f}")

    # Detect repetition: does the picked move create a repetition / does any
    # high-visit move lead to a repeated position?
    if best_legal:
        nb = board.copy()
        nb.push(best_mv)
        creates_rep = nb.is_repetition(2) or nb.is_repetition(3)
        is_capture = board.is_capture(best_mv)
        gives_check = nb.is_check()
        print(f"  PICKED move: creates_repetition={creates_rep}  is_capture={is_capture}  gives_check={gives_check}")

    return root, wdl, v, raw_q, visits, actions


def main():
    game = ChessGame()
    cfg = get_config("chess_small")
    cfg.device = "cpu"
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    print(f"Loaded {CKPT}  step={ckpt.get('step')}")
    print(f"cfg: value_head_type={cfg.value_head_type} draw_score={cfg.draw_score} "
          f"num_sims={cfg.num_simulations} sample_k={cfg.sample_k} "
          f"moves_left_mcts={cfg.moves_left_mcts}")
    net = build_network(ckpt, game, cfg, "cpu")

    # 1. K+Q vs K — trivially won for white
    b1 = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    trace_position(net, game, cfg, "[1] K+Q vs K (white to move, trivially WON)", b1)

    # 2. K+R vs K — won for white
    b2 = chess.Board("4k3/8/8/8/8/8/8/3RK3 w - - 0 1")
    trace_position(net, game, cfg, "[2] K+R vs K (white to move, WON)", b2)

    # 3. KQ vs K, mate in 1 available
    b3 = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    trace_position(net, game, cfg, "[3] KQ vs K, MATE IN 1 (Qf7-g7#)", b3)

    # 4/5. Position deep in a shuffle: KQ vs K where white (WINNING) has been
    # triangulating the queen and the position has repeated. White to move; some
    # legal moves return to a position already seen (one ply from / into 3fold),
    # while OTHER legal moves make progress. Does MCTS prefer the repeating shuffle?
    b5 = chess.Board("7k/8/6K1/8/8/8/8/3Q4 w - - 0 1")
    # Triangulate so the position repeats. Qd1<->Qd4 (non-check) with black king
    # shuffling g8<->h8.
    seq = ["Qd4", "Kg8", "Qd1", "Kh8", "Qd4", "Kg8", "Qd1", "Kh8"]
    for s in seq:
        b5.push_san(s)
    # White to move. Qd4 returns to a position seen twice before -> a 3fold DRAW.
    # CRUCIALLY, Qd8# (mate in 1) is also available — a winning, progress move.
    nb_test = b5.copy(); nb_test.push_san("Qd4")
    a_rep = _move_to_action(b5.parse_san("Qd4"), b5.turn)
    a_mate = _move_to_action(b5.parse_san("Qd8#"), b5.turn)
    print(f"\n  [pre-5] after Qd4: is_repetition(3)={nb_test.is_repetition(3)} "
          f"is_repetition(2)={nb_test.is_repetition(2)} | Qd4 action={a_rep} Qd8# action={a_mate}")
    trace_position(net, game, cfg,
                   "[5] KQ vs K deep in shuffle: Qd4 -> 3fold DRAW, but Qd8# MATES",
                   b5,
                   note=f"Qd4 (action {a_rep}) creates 3fold (DRAW); Qd8# (action {a_mate}) MATES. "
                        f"Does the repeating Qd4 get high backed-up Q (phantom win)?")
    return  # positions [4] folded into [5]


if __name__ == "__main__":
    main()
