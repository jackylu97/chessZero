"""Trace self-play games from a checkpoint and dump the MCTS root tree per move
for games that end in threefold repetition.

Replicates play_games_parallel (BatchedMCTS, add_noise=True, production temp
schedule, 200 sims) but captures, per move: FEN, halfmove clock, repetition
state, the chosen move, root value, and the top-K children by visits with their
visit count, mover-POV Q (= reward - discount*child_value), and prior.

Threefold games are written one-per-file to <out>/game_XXXX.json for subagent
tracing. Usage:
    .venv/bin/python scripts/trace_threefold.py --checkpoint <ckpt> --games 24 --out /tmp/tf
"""
import argparse, os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.mcts.mcts import BatchedMCTS, select_action
from src.training.replay_buffer import stack_with_history
import scripts.play_web as play_web


def root_topk(root, board, k=8):
    acts = np.asarray(root.child_actions)
    vis = np.asarray(root.child_visits, dtype=float)
    vs = np.asarray(root.child_value_sums, dtype=float)
    rw = np.asarray(root.child_rewards, dtype=float)
    pri = np.asarray(root.child_priors, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        child_v = np.where(vis > 0, vs / np.maximum(vis, 1.0), 0.0)
    q_mover = rw - 1.0 * child_v  # discount=1
    order = np.argsort(-vis)[:k]
    tot = vis.sum() or 1.0
    out = []
    for i in order:
        mv = _action_to_move(int(acts[i]), board)
        out.append({
            "uci": mv.uci() if mv is not None else None,
            "visits": int(vis[i]),
            "frac": round(float(vis[i] / tot), 3),
            "q_mover": round(float(q_mover[i]), 3),
            "prior": round(float(pri[i]), 3),
        })
    return out


def _rep_override(root, board, chosen_action, draw_score):
    """Root-terminal-draws proxy. If `chosen_action` completes a draw-by-repetition
    or 50-move, return the best NON-drawing child (most visits) whose mover-POV Q
    exceeds draw_score; else None (keep the draw — correct when no better move exists,
    e.g. a losing side). Uses only the real board the agent already has at the root."""
    mv = _action_to_move(int(chosen_action), board)
    if mv is None:
        return None
    board.push(mv)
    chosen_draws = board.is_repetition(3) or board.halfmove_clock >= 100
    board.pop()
    if not chosen_draws:
        return None
    acts = np.asarray(root.child_actions)
    vis = np.asarray(root.child_visits, dtype=float)
    vs = np.asarray(root.child_value_sums, dtype=float)
    rw = np.asarray(root.child_rewards, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        child_v = np.where(vis > 0, vs / np.maximum(vis, 1.0), 0.0)
    q = rw - child_v  # mover-POV, discount=1
    best, best_vis = None, -1.0
    for i, a in enumerate(acts):
        m = _action_to_move(int(a), board)
        if m is None:
            continue
        board.push(m)
        d = board.is_repetition(3) or board.halfmove_clock >= 100
        board.pop()
        if d:
            continue
        if q[i] > draw_score and vis[i] > best_vis:
            best, best_vis = int(a), vis[i]
    return best


def _forced_draw_actions(board, legal_actions):
    """Set of root action indices that lead to a draw-by-repetition / 50-move,
    via the real board (the agent's own move history). Passed to run_batch so the
    in-search root-terminal-draws override can value them at draw_score."""
    fd = set()
    for a in legal_actions:
        mv = _action_to_move(int(a), board)
        if mv is None:
            continue
        board.push(mv)
        if board.is_repetition(3) or board.halfmove_clock >= 100:
            fd.add(int(a))
        board.pop()
    return fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--max-moves", type=int, default=220)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/tmp/threefold_trace")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--temp-final", type=float, default=None,
                    help="Override config.temperature_final (late-game temperature after move 30).")
    ap.add_argument("--dirichlet-alpha", type=float, default=None,
                    help="Override config.dirichlet_alpha (root exploration noise concentration).")
    ap.add_argument("--root-terminal-draws", action="store_true",
                    help="In-search root-terminal-draws override: forced-draw root children valued at "
                         "draw_score (winning side plays on; losing side still draws).")
    ap.add_argument("--moves-left-mcts", action="store_true",
                    help="Enable the MLH search utility (prefer faster wins). Needs a trained moves-left head.")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    cfg = get_config("chess"); cfg.device = args.device; cfg.num_simulations = args.sims
    if args.temp_final is not None:
        cfg.temperature_final = args.temp_final
    if args.dirichlet_alpha is not None:
        cfg.dirichlet_alpha = args.dirichlet_alpha
    if args.moves_left_mcts:
        cfg.moves_left_mcts = True
    game = ChessGame()
    net = play_web.load_network(args.checkpoint, game, cfg, args.device)
    mcts = BatchedMCTS(net, game, cfg, args.device)
    A = game.action_space_size
    nf = cfg.history_frames; n_random = cfg.random_opening_plies
    temp_init = cfg.temperature_init

    states = [game.reset() for _ in range(args.games)]
    obs_hist = [[] for _ in range(args.games)]            # per-game single-frame history
    traces = [[] for _ in range(args.games)]
    mc = [0] * args.games
    overrides = [0] * args.games
    active = list(range(args.games))

    it = 0
    while active and it < args.max_moves:
        single = {g: game.to_tensor(states[g]) for g in active}
        legal = {g: game.legal_actions(states[g]) for g in active}
        mcts_g = [g for g in active if mc[g] >= n_random]
        roots = {}
        fd_by_g = {}
        if mcts_g:
            obs_list = [stack_with_history(single[g], obs_hist[g], nf) for g in mcts_g]
            fda = None
            if args.root_terminal_draws:
                fd_by_g = {g: _forced_draw_actions(states[g].board, legal[g]) for g in mcts_g}
                fda = [fd_by_g[g] for g in mcts_g]   # in-search override: real mechanism
            rs = mcts.run_batch(obs_list, [legal[g] for g in mcts_g], add_noise=True,
                                forced_draw_actions=fda)
            roots = dict(zip(mcts_g, rs))
        for g in active:
            board = states[g].board
            if mc[g] < n_random:
                action = random.choice(legal[g]); rv = 0.0; tk = None
            else:
                root = roots[g]
                temp = temp_init if mc[g] < cfg.temperature_drop_step else cfg.temperature_final
                action, _ = select_action(root, temperature=temp)
                rv = float(root.value)
                if fd_by_g.get(g):
                    overrides[g] += 1   # a forced-draw option existed this ply (override active)
                tk = root_topk(root, board, args.topk)
                mv = _action_to_move(int(action), board)
                traces[g].append({
                    "ply": mc[g], "fen": board.fen(),
                    "halfmove_clock": board.halfmove_clock,
                    "rep2": board.is_repetition(2), "rep3": board.is_repetition(3),
                    "chosen": mv.uci() if mv is not None else None,
                    "temp": round(temp, 3), "root_value": round(rv, 3),
                    "n_legal": len(legal[g]), "top": tk,
                })
            obs_hist[g].append(single[g])
            st, _, _ = game.step(states[g], action)
            states[g] = st; mc[g] += 1
        active = [g for g in active if not states[g].done]
        it += 1

    # classify + dump threefold games
    summary = []
    n_tf = 0
    for g in range(args.games):
        b = states[g].board
        outcome = states[g].winner if states[g].done else None
        is_tf = bool(states[g].done and outcome == 0 and b.is_repetition(3))
        tag = ("threefold" if is_tf else
               ("draw_other" if (states[g].done and outcome == 0) else
                ("decisive" if states[g].done else "unfinished")))
        summary.append({"game": g, "plies": mc[g], "outcome": outcome, "type": tag,
                        "final_fen": b.fen(), "overrides": overrides[g]})
        if is_tf:
            n_tf += 1
            with open(os.path.join(args.out, f"game_{g:04d}.json"), "w") as f:
                json.dump({"game": g, "checkpoint": args.checkpoint, "plies": mc[g],
                           "final_fen": b.fen(), "moves": traces[g]}, f, indent=1)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    from collections import Counter
    tally = Counter(s["type"] for s in summary)
    decisive = tally.get("decisive", 0)
    avg_plies = sum(s["plies"] for s in summary) / max(1, len(summary))
    print(f"Done: sims={args.sims} temp_final={cfg.temperature_final} "
          f"dirichlet_alpha={cfg.dirichlet_alpha} games={args.games} "
          f"seed={args.seed} -> {args.out}")
    print(f"  TALLY: threefold={tally.get('threefold',0)} draw_other={tally.get('draw_other',0)} "
          f"decisive={decisive} unfinished={tally.get('unfinished',0)} | avg_plies={avg_plies:.0f}")
    if args.root_terminal_draws:
        fired = [s for s in summary if s["overrides"] > 0]
        oc = Counter(s["type"] for s in fired)
        tot_ov = sum(s["overrides"] for s in summary)
        print(f"  ROOT-TERMINAL-DRAWS: {tot_ov} move-overrides across {len(fired)}/{args.games} games; "
              f"outcome of overridden games -> decisive={oc.get('decisive',0)} "
              f"draw_other={oc.get('draw_other',0)} threefold={oc.get('threefold',0)} "
              f"unfinished={oc.get('unfinished',0)}")
    for s in summary:
        print(f"  game {s['game']:2d}: {s['type']:11s} plies={s['plies']:3d} outcome={s['outcome']} ov={s['overrides']}")


if __name__ == "__main__":
    main()
