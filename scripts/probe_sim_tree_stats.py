"""How much does 4x sims actually buy? Run MCTS at several sim budgets on a fixed
set of positions and report tree depth/width, move stability, and (optionally)
Stockfish move agreement — to find a compute-efficient sims sweet spot.

Per sim budget: max/mean tree depth, root width (# moves visited), top-move visit
fraction, visit entropy, fraction of positions whose chosen move MATCHES the
800-sim choice (if it already matches at 200, the extra sims are wasted), and SF
best-move agreement.

Run: .venv/bin/python scripts/probe_sim_tree_stats.py --checkpoint <ckpt.pt> \
        --game chess_small --positions 30 --sims 50 100 200 400 800
"""
import argparse, os, sys
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move
from src.mcts.mcts import BatchedMCTS
from src.training.replay_buffer import stack_with_history
from scripts.eval_checkpoint_health import build_network, random_playout


def tree_stats(root):
    """max_depth, mean node depth, total nodes from a BatchedMCTS root Node."""
    nodes, max_d, depth_sum = 0, 0, 0
    dq = deque([(root, 0)])
    while dq:
        node, d = dq.popleft()
        nodes += 1; max_d = max(max_d, d); depth_sum += d
        for ch in getattr(node, "children", []):
            if ch is not None:
                dq.append((ch, d + 1))
    cv = np.asarray(root.child_visits, dtype=np.float64) if root.child_visits is not None else np.zeros(1)
    tot = cv.sum()
    width = int((cv > 0).sum())
    top = float(cv.max() / tot) if tot > 0 else 0.0
    p = cv[cv > 0] / tot if tot > 0 else np.array([])
    ent = float(-(p * np.log(p)).sum()) if p.size else 0.0
    top_action = int(root.child_actions[int(cv.argmax())]) if tot > 0 else -1
    return max_d, depth_sum / max(1, nodes), nodes, width, top, ent, top_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--game", default="chess_small")
    ap.add_argument("--positions", type=int, default=30)
    ap.add_argument("--sims", type=int, nargs="+", default=[50, 100, 200, 400, 800])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stockfish", default="tools/stockfish/stockfish")
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--fens-file", default=None,
                    help="CSV/text with a FEN per line (or last comma field); use these "
                         "positions instead of random middlegame playouts (e.g. endgame FENs).")
    ap.add_argument("--max-pieces", type=int, default=99,
                    help="keep only FENs with <= this many pieces (filter for endgames).")
    args = ap.parse_args()

    dev = args.device
    game = ChessGame()
    cfg = get_config(args.game); cfg.device = dev
    HF = getattr(cfg, "history_frames", 1)
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev); net.eval()

    # Fixed positions (same across sim budgets): random middlegames, or FENs.
    obs_list, legal_list, boards = [], [], []
    if args.fens_file:
        import chess as _chess
        fens = []
        with open(args.fens_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cand = line.split(",")[-1].strip() if "," in line else line
                try:
                    b = _chess.Board(cand)
                    if len(b.piece_map()) <= args.max_pieces and not b.is_game_over():
                        fens.append(cand)
                except Exception:
                    continue
        fens = fens[: args.positions]
        for fen in fens:
            s = game.reset_from_fen(fen)
            obs_list.append(stack_with_history(game.to_tensor(s), [], HF))
            legal_list.append(game.legal_actions(s))
            boards.append(s.board)
    else:
        for i in range(args.positions):
            s, frames = random_playout(game, 10 + (i % 30), seed=1000 + i)
            obs_list.append(stack_with_history(game.to_tensor(s), frames, HF))
            legal_list.append(game.legal_actions(s))
            boards.append(s.board)
    print(f"net step {ckpt.get('step','?')} | {len(obs_list)} fixed positions | sims {args.sims}\n")

    # Stockfish best move per position (optional).
    sf_best = [None] * len(boards)
    if os.path.exists(args.stockfish):
        try:
            import chess.engine
            eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)
            for i, b in enumerate(boards):
                try:
                    r = eng.play(b, chess.engine.Limit(depth=args.sf_depth))
                    sf_best[i] = r.move
                except Exception:
                    pass
            eng.quit()
        except Exception as e:
            print(f"(stockfish unavailable: {e})")

    rows = {}
    chosen = {}  # sims -> [top_action per position]
    for S in args.sims:
        cfg.num_simulations = S
        mcts = BatchedMCTS(net, game, cfg, dev)
        roots = mcts.run_batch(obs_list, legal_list, add_noise=False)
        md, mnd, wid, topf, ent, acts = [], [], [], [], [], []
        for root in roots:
            a, b, n, w, t, e, ta = tree_stats(root)
            md.append(a); mnd.append(b); wid.append(w); topf.append(t); ent.append(e); acts.append(ta)
        chosen[S] = acts
        rows[S] = (np.mean(md), np.mean(mnd), np.mean(wid), np.mean(topf), np.mean(ent))

    ref = max(args.sims)
    print(f"{'sims':>5} {'maxDepth':>9} {'meanDepth':>10} {'rootWidth':>10} "
          f"{'topMove%':>9} {'visitEntropy':>13} {'agree@'+str(ref):>9} {'SFmatch':>8}")
    for S in args.sims:
        md, mnd, wid, topf, ent = rows[S]
        agree = np.mean([chosen[S][i] == chosen[ref][i] for i in range(len(obs_list))])
        sf_ok = [chosen[S][i] for i in range(len(boards)) if sf_best[i] is not None]
        sf_match = np.mean([
            _action_to_move(chosen[S][i], boards[i]) == sf_best[i]
            for i in range(len(boards)) if sf_best[i] is not None and chosen[S][i] >= 0
        ]) if any(sf_best) else float('nan')
        print(f"{S:>5} {md:>9.2f} {mnd:>10.2f} {wid:>10.2f} {topf:>9.1%} {ent:>13.3f} "
              f"{agree:>9.1%} {(sf_match if not np.isnan(sf_match) else 0):>8.1%}")
    print("\n  agree@N = fraction of positions whose argmax-visit move matches the N-sim choice")
    print("  (if 200 already matches 800, the extra 4x sims rarely change the move → wasted)")


if __name__ == "__main__":
    main()
