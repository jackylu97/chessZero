"""Trace self-play games on the MATERIAL-ON checkpoint (chess_small) and, at the
threefold-decision positions, report the actual mechanism numbers:
 - root_value entering the repetition
 - repeating move's Q vs siblings, prior, visit fraction
 - within-position sibling-Q spread
 - value head WDL scalar, material head predicted margin vs ACTUAL STM material
 - moves-left head prediction
 - Stockfish reference eval (STM POV)

Loads material-head checkpoints via build_network (eval_checkpoint_health).
CPU-only to avoid GPU contention with the live run.
"""
import argparse, os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess, chess.engine

from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.mcts.mcts import BatchedMCTS, select_action
from src.training.replay_buffer import stack_with_history, _material_margin_stm
from src.model.utils import support_to_scalar, inverse_scalar_transform
from scripts.eval_checkpoint_health import build_network


def root_topk(root, board, k=12):
    acts = np.asarray(root.child_actions)
    vis = np.asarray(root.child_visits, dtype=float)
    vs = np.asarray(root.child_value_sums, dtype=float)
    rw = np.asarray(root.child_rewards, dtype=float)
    pri = np.asarray(root.child_priors, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        child_v = np.where(vis > 0, vs / np.maximum(vis, 1.0), 0.0)
    q_mover = rw - 1.0 * child_v
    order = np.argsort(-vis)[:k]
    tot = vis.sum() or 1.0
    out = []
    for i in order:
        mv = _action_to_move(int(acts[i]), board)
        out.append({
            "uci": mv.uci() if mv is not None else None,
            "visits": int(vis[i]), "frac": round(float(vis[i] / tot), 3),
            "q_mover": round(float(q_mover[i]), 4), "prior": round(float(pri[i]), 4),
        })
    return out, q_mover, vis, acts


def material_pred(net, hidden):
    """Decode the material head's predicted STM margin (pawns)."""
    logits = net.predict_material(hidden)  # (1, 2K+1)
    probs = torch.softmax(logits, dim=1)
    s = support_to_scalar(probs, net.material_head_support_size)  # transformed scalar
    return float(inverse_scalar_transform(s).item())


def moves_left_pred(net, hidden):
    logits = net.predict_moves_left(hidden)
    probs = torch.softmax(logits, dim=1)
    K = net.moves_left_support_size
    bins = torch.arange(0, 2 * K + 1, device=logits.device, dtype=torch.float32)
    # moves-left support is plies-to-end; head trained on scalar plies via support.
    # We report the expected support index (informational ordering of bins).
    return float((probs[0] * bins).sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--game", default="chess_small")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--max-moves", type=int, default=160)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stockfish", default="tools/stockfish/stockfish")
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--out", default="/tmp/claude-0/-workspace-chessZero/scratchpad/mat_trace")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    cfg = get_config(args.game); cfg.device = args.device; cfg.num_simulations = args.sims
    game = ChessGame()
    torch.serialization.add_safe_globals([type(cfg)])
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    net = build_network(ckpt, game, cfg, args.device)
    net.eval()
    print(f"loaded {args.checkpoint}  material_head={net.material_head is not None} "
          f"moves_left={net.moves_left_head is not None} value_head={cfg.value_head_type}")

    mcts = BatchedMCTS(net, game, cfg, args.device)
    nf = cfg.history_frames; n_random = cfg.random_opening_plies
    temp_init = cfg.temperature_init

    states = [game.reset() for _ in range(args.games)]
    obs_hist = [[] for _ in range(args.games)]
    # store per-ply (board_fen, hidden, root, topk) for threefold games
    rec = [[] for _ in range(args.games)]
    mc = [0] * args.games
    active = list(range(args.games))

    it = 0
    while active and it < args.max_moves:
        single = {g: game.to_tensor(states[g]) for g in active}
        legal = {g: game.legal_actions(states[g]) for g in active}
        mcts_g = [g for g in active if mc[g] >= n_random]
        roots = {}
        if mcts_g:
            obs_list = [stack_with_history(single[g], obs_hist[g], nf) for g in mcts_g]
            rs = mcts.run_batch(obs_list, [legal[g] for g in mcts_g], add_noise=True)
            roots = dict(zip(mcts_g, rs))
        for g in active:
            board = states[g].board
            if mc[g] < n_random:
                action = random.choice(legal[g])
            else:
                root = roots[g]
                temp = temp_init if mc[g] < cfg.temperature_drop_step else cfg.temperature_final
                action, _ = select_action(root, temperature=temp)
                tk, q_all, vis_all, acts_all = root_topk(root, board, 12)
                # net readouts at this position (full history stack)
                obs = stack_with_history(single[g], obs_hist[g], nf).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    hidden, _, vscalar = net.initial_inference(obs)
                    mat_pred = material_pred(net, hidden) if net.material_head is not None else None
                    ml_pred = moves_left_pred(net, hidden) if net.moves_left_head is not None else None
                actual_mat = _material_margin_stm(single[g])
                mv = _action_to_move(int(action), board)
                # would the chosen move complete a threefold / 50-move?
                board.push(mv); chosen_draws = board.is_repetition(3) or board.halfmove_clock >= 100; board.pop()
                rec[g].append({
                    "ply": mc[g], "fen": board.fen(), "stm": "w" if board.turn else "b",
                    "rep2": board.is_repetition(2), "rep3": board.is_repetition(3),
                    "chosen": mv.uci() if mv else None, "chosen_completes_draw": bool(chosen_draws),
                    "root_value": round(float(root.value), 4),
                    "value_head": round(float(vscalar.item()), 4),
                    "material_actual": round(actual_mat, 2),
                    "material_pred": round(mat_pred, 2) if mat_pred is not None else None,
                    "moves_left_pred": round(ml_pred, 2) if ml_pred is not None else None,
                    "n_legal": len(legal[g]), "top": tk,
                })
            obs_hist[g].append(single[g])
            st, _, _ = game.step(states[g], action)
            states[g] = st; mc[g] += 1
        active = [g for g in active if not states[g].done]
        it += 1

    # classify; analyze threefold games at their repetition-decision plies
    sf = None
    try:
        sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    except Exception as e:
        print(f"(stockfish unavailable: {e})")

    summary = []
    tf_games = []
    for g in range(args.games):
        b = states[g].board
        outcome = states[g].winner if states[g].done else None
        is_tf = bool(states[g].done and outcome == 0 and b.is_repetition(3))
        tag = ("threefold" if is_tf else ("draw_other" if (states[g].done and outcome == 0)
               else ("decisive" if states[g].done else "unfinished")))
        summary.append({"game": g, "plies": mc[g], "type": tag, "final_fen": b.fen()})
        if is_tf:
            tf_games.append(g)

    from collections import Counter
    tally = Counter(s["type"] for s in summary)
    print(f"\nTALLY ({args.games} games, sims={args.sims}): {dict(tally)}  "
          f"avg_plies={sum(s['plies'] for s in summary)/max(1,len(summary)):.0f}")

    # at the decision plies (first chosen_completes_draw) per threefold game,
    # aggregate the mechanism numbers
    agg = {"root_value": [], "chosen_q": [], "best_other_q": [], "q_spread": [],
           "chosen_is_topq": [], "chosen_prior": [], "chosen_frac": [], "chosen_is_topvisit": [],
           "value_head": [], "mat_actual": [], "mat_pred": [], "ml_pred": [], "sf_eval": []}
    detail_lines = []
    for g in tf_games:
        # decision plies: the ply where the chosen move completes the threefold,
        # plus the surrounding rep2 plies (shuffle region)
        decisions = [r for r in rec[g] if r["chosen_completes_draw"]]
        if not decisions:
            decisions = [r for r in rec[g] if r["rep2"]][-3:]
        for r in decisions:
            top = r["top"]
            if not top:
                continue
            chosen = r["chosen"]
            crow = next((t for t in top if t["uci"] == chosen), None)
            if crow is None:
                continue
            qs = [t["q_mover"] for t in top]
            other_q = [t["q_mover"] for t in top if t["uci"] != chosen]
            chosen_q = crow["q_mover"]
            best_other = max(other_q) if other_q else chosen_q
            q_spread = max(qs) - min(qs)
            is_topq = chosen_q >= max(qs) - 1e-9
            is_topvisit = top[0]["uci"] == chosen
            sf_eval = None
            if sf is not None:
                try:
                    info = sf.analyse(chess.Board(r["fen"]), chess.engine.Limit(depth=args.sf_depth))
                    sc = info["score"].pov(chess.Board(r["fen"]).turn)
                    sf_eval = sc.score(mate_score=10000)
                except Exception:
                    sf_eval = None
            agg["root_value"].append(r["root_value"]); agg["chosen_q"].append(chosen_q)
            agg["best_other_q"].append(best_other); agg["q_spread"].append(q_spread)
            agg["chosen_is_topq"].append(is_topq); agg["chosen_prior"].append(crow["prior"])
            agg["chosen_frac"].append(crow["frac"]); agg["chosen_is_topvisit"].append(is_topvisit)
            agg["value_head"].append(r["value_head"]); agg["mat_actual"].append(r["material_actual"])
            if r["material_pred"] is not None: agg["mat_pred"].append(r["material_pred"])
            if r["moves_left_pred"] is not None: agg["ml_pred"].append(r["moves_left_pred"])
            if sf_eval is not None: agg["sf_eval"].append(sf_eval)
            detail_lines.append(
                f"  g{g} ply{r['ply']} stm={r['stm']} rootV={r['root_value']:+.3f} "
                f"vhead={r['value_head']:+.3f} mat_act={r['material_actual']:+.1f} "
                f"mat_pred={r['material_pred']} ml={r['moves_left_pred']} "
                f"sfcp={sf_eval} | chosen={chosen} q={chosen_q:+.3f} pri={crow['prior']:.3f} "
                f"frac={crow['frac']:.2f} topQ={is_topq} topVis={is_topvisit} "
                f"bestOtherQ={best_other:+.3f} spread={q_spread:.3f}")
    if sf is not None:
        sf.quit()

    def m(x): return (sum(x)/len(x)) if x else float("nan")
    print(f"\n=== THREEFOLD DECISION-PLY AGGREGATE ({len(agg['root_value'])} positions across {len(tf_games)} tf games) ===")
    print(f"  root_value entering rep      : mean {m(agg['root_value']):+.3f}  "
          f"(min {min(agg['root_value']) if agg['root_value'] else 0:+.3f}, max {max(agg['root_value']) if agg['root_value'] else 0:+.3f})")
    print(f"  value_head scalar            : mean {m(agg['value_head']):+.3f}")
    print(f"  chosen(repeat) Q             : mean {m(agg['chosen_q']):+.3f}")
    print(f"  best OTHER move Q            : mean {m(agg['best_other_q']):+.3f}")
    print(f"  Q(chosen) - Q(best_other)    : mean {m(agg['chosen_q'])-m(agg['best_other_q']):+.3f}")
    print(f"  within-pos Q spread          : mean {m(agg['q_spread']):.3f}")
    print(f"  chosen is TOP-Q              : {sum(agg['chosen_is_topq'])}/{len(agg['chosen_is_topq'])}")
    print(f"  chosen is TOP-VISIT          : {sum(agg['chosen_is_topvisit'])}/{len(agg['chosen_is_topvisit'])}")
    print(f"  chosen prior                 : mean {m(agg['chosen_prior']):.3f}")
    print(f"  chosen visit frac            : mean {m(agg['chosen_frac']):.3f}")
    print(f"  --- MATERIAL signal ---")
    print(f"  actual STM material margin   : mean {m(agg['mat_actual']):+.2f} pawns")
    print(f"  material-head predicted      : mean {m(agg['mat_pred']):+.2f} pawns")
    print(f"  moves-left head pred (bin)   : mean {m(agg['ml_pred']):.2f}")
    print(f"  Stockfish eval (STM, cp)     : mean {m(agg['sf_eval']):+.0f}")
    print("\n=== PER-POSITION DETAIL ===")
    for ln in detail_lines:
        print(ln)

    with open(os.path.join(args.out, "agg.json"), "w") as f:
        json.dump({"tally": dict(tally), "agg": agg, "detail": detail_lines}, f, indent=1)


if __name__ == "__main__":
    main()
