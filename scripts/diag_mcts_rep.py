"""Trace MCTS on real positions taken from just before a 3-fold draw in the
buffer. Question: the value head learned rep->loss at the ROOT (diag_value_probe).
But MCTS dynamics is latent — does the search DEVALUE the repeating move, or does
it still bank a high Q on it? This is the 'value right but search still draws' test.

For the chosen position we:
  - reconstruct the board + 8-frame history
  - run MCTS (numpy reference engine)
  - for each root child, print prior, visit count, Q
  - identify which child is the move that LEADS BACK to a repeated position
    (i.e. produces a board whose hash already appeared earlier) and report its
    Q / visits vs the best non-repeating alternative.
"""
import sys
import numpy as np
import torch
import chess

sys.path.insert(0, "/workspace/chessZero")
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move, _move_to_action
from src.mcts.mcts import MCTS
from src.training.replay_buffer import ReplayBuffer, stack_with_history
from scripts.eval_checkpoint_health import build_network

CKPT = "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"
BUF = "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf"


def main():
    torch.serialization.add_safe_globals([MuZeroConfig])
    cfg = get_config("chess_small"); cfg.device = "cpu"
    cfg.num_simulations = 64
    game = ChessGame()
    HF = cfg.history_frames
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    net = build_network(ck, game, cfg, "cpu")
    mcts = MCTS(net, game, cfg, "cpu")

    rb = ReplayBuffer(max_size=100000)
    rb.load(BUF, game=game)

    # Find a game that drew by repetition and replay to near its end.
    rep_games = [g for g in rb.buffer if getattr(g, "draw_by_repetition", False) and len(g) > 40]
    print(f"rep-draw games >40 plies: {len(rep_games)}")
    g = rep_games[3]
    print(f"chosen game len={len(g)} outcome={g.game_outcome}")

    # Find the LAST ply whose stored obs has rep2==1 (a 2-fold already present);
    # at that ply at least one move likely completes the 3-fold. Trace from there.
    target_ply = None
    for p in range(len(g) - 2, 0, -1):
        if float(g.observations[p][19, 0, 0]) >= 0.5:  # rep2 plane set
            target_ply = p
            break
    if target_ply is None:
        target_ply = len(g) - 6
    print(f"using target_ply={target_ply} (rep2 set there)")
    board = chess.Board()
    hashes = {}  # board hash -> count, replicating what's "repeated"
    states = []
    for i in range(target_ply):
        states.append(board.fen())
        mv = _action_to_move(g.actions[i], board)
        board.push(mv)
    cur_board = board.copy()

    # Build history frames from stored observations (already correct rep planes).
    frames = [g.observations[i] for i in range(target_ply)]
    cur_obs = g.observations[target_ply]  # single frame at target_ply
    obs_stack = stack_with_history(cur_obs, frames, HF)  # mcts.run adds the batch dim

    legal = g.legal_actions_list[target_ply] if target_ply < len(g.legal_actions_list) else \
        [_move_to_action(m, cur_board.turn) for m in cur_board.legal_moves]

    print(f"\nposition at ply {target_ply}: {cur_board.fen()}")
    print(f"side to move: {'white' if cur_board.turn else 'black'}  legal moves: {len(legal)}")
    # rep planes of current obs
    print(f"cur obs rep2={float(cur_obs[19,0,0])} rep3={float(cur_obs[20,0,0])} np={float(cur_obs[21,0,0]):.3f}")

    # Identify which legal moves lead to a previously-seen position (a repetition).
    seen = set()
    bb = chess.Board()
    for i in range(target_ply):
        seen.add(bb._transposition_key())
        mv = _action_to_move(g.actions[i], bb)
        bb.push(mv)
    seen.add(cur_board._transposition_key())  # current position too

    rep_moves = set()
    for a in legal:
        mv = _action_to_move(a, cur_board)
        tmp = cur_board.copy(); tmp.push(mv)
        if tmp._transposition_key() in seen:
            rep_moves.add(a)
    print(f"moves that repeat a prior position: {len(rep_moves)} of {len(legal)}")

    # Run MCTS.
    root = mcts.run(obs_stack, legal, add_noise=False)
    print(f"\nroot.value (search estimate) = {root.value:.3f}")

    # Read per-child stats from parallel arrays. Q = reward + value_sum/visits (child POV),
    # but we want parent POV: parent_Q = reward + gamma*(-child_value). The arrays store
    # child_value_sums in CHILD POV; compute parent-POV mean Q like _select_child does.
    acts = root.child_actions
    priors = root.child_priors
    visits = root.child_visits
    rewards = root.child_rewards
    vsum = root.child_value_sums
    children = []
    for i in range(len(acts)):
        a = int(acts[i])
        vc = int(visits[i])
        if vc > 0:
            child_val = vsum[i] / vc  # child POV
            parent_q = rewards[i] + cfg.discount * (-child_val)
        else:
            parent_q = float("nan")
        children.append((a, float(priors[i]), vc, parent_q, a in rep_moves))
    children.sort(key=lambda x: -x[2])  # by visit count

    print(f"\n{'rank':>4} {'move':>7} {'prior':>7} {'visits':>7} {'Q(par)':>8} {'repeats?':>9}")
    for i, (a, prior, vc, q, isrep) in enumerate(children[:14]):
        mv = _action_to_move(a, cur_board)
        print(f"{i+1:>4} {mv.uci():>7} {prior:>7.3f} {vc:>7d} {q:>8.3f} {'  REPEAT' if isrep else '':>9}")

    # Summary: top move repeats?
    top = children[0]
    print(f"\nTOP move ({_action_to_move(top[0], cur_board).uci()}) repeats? {top[0] in rep_moves}")
    rep_children = [c for c in children if c[4]]
    nonrep_children = [c for c in children if not c[4]]
    if rep_children:
        best_rep = max(rep_children, key=lambda x: x[2])
        print(f"best repeating move: {_action_to_move(best_rep[0],cur_board).uci()} "
              f"visits={best_rep[2]} Q={best_rep[3]:.3f}")
    if nonrep_children:
        best_nonrep = max(nonrep_children, key=lambda x: x[2])
        print(f"best non-rep move:   {_action_to_move(best_nonrep[0],cur_board).uci()} "
              f"visits={best_nonrep[2]} Q={best_nonrep[3]:.3f}")


if __name__ == "__main__":
    main()
