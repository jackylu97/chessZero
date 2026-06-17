"""Simulation-scaling probe: does MORE MCTS search rank moves better than the
raw (coarse) value head does, and does the model improve over training?

The sibling probe showed the value head can't rank sibling moves (Spearman ~0,
value-greedy best-move agreement ~10%). MCTS does lookahead, so more simulations
*might* recover move quality the 1-ply value misses. This probe runs real MCTS at
several simulation budgets on a FIXED set of real middlegame positions (identical
across checkpoints, so the comparison is controlled), and scores the move MCTS
actually picks (argmax visits) against Stockfish.

Metrics per (checkpoint, num_simulations), aggregated over the fixed positions:
  - best-move agreement: MCTS top move == Stockfish best move (%)
  - SF-rank of MCTS top move (1 = best; median)
  - ACPL: centipawn loss of MCTS top move vs SF best (mean)
  - Spearman(visit counts, SF eval) over the legal moves (within-position; median)

Run: .venv/bin/python scripts/probe_sim_scaling.py \
        --checkpoints 8k=<p1> 16k=<p2> 74k=<p3> --sims 50 200 800
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess, chess.engine
from scipy.stats import spearmanr

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move
from src.model.muzero_net import MuZeroNetwork
from src.mcts.mcts import MCTS
from src.training.replay_buffer import stack_with_history

# Fixed, realistic middlegame/opening positions (several reasonable moves each).
FENS = [
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkb1r/pp2pppp/3p1n2/2pP4/4P3/2N5/PPP2PPP/R1BQKBNR b KQkq - 0 4",
    "r2q1rk1/pp1bbppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9",
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "2rq1rk1/pb1nbppp/1p2pn2/2pp4/3P4/1P1BPN2/PBPN1PPP/R2Q1RK1 w - - 0 11",
    "rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 2 5",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
]


def build_net(path, cfg, game, device):
    torch.serialization.add_safe_globals([MuZeroConfig])
    sd = torch.load(path, map_location=device, weights_only=True)["model_state_dict"]
    net = MuZeroNetwork(
        observation_channels=game.num_planes * cfg.history_frames,
        action_space_size=game.action_space_size, hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w,
        input_h=8, input_w=8, fc_hidden=cfg.fc_hidden,
        value_support_size=cfg.value_support_size, reward_support_size=cfg.reward_support_size,
        use_consistency_loss=any(k.startswith("projection.") for k in sd),
        use_scalar_transform=cfg.use_scalar_transform, value_target_scale=cfg.value_target_scale,
        value_head_type=cfg.value_head_type, draw_score=cfg.draw_score,
        use_inverse_dynamics_loss=any(k.startswith("inverse_dynamics_head.") for k in sd),
    ).to(device)
    net.load_state_dict(sd); net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="label=path ...")
    ap.add_argument("--sims", type=int, nargs="+", default=[50, 200, 800])
    ap.add_argument("--sf-depth", type=int, default=14)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg = get_config("chess"); cfg.device = args.device
    game = ChessGame()
    eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)

    # Precompute Stockfish ground truth per position (depends only on the FEN).
    sf = []
    for fen in FENS:
        b = chess.Board(fen)
        legal = list(b.legal_moves)
        info = eng.analyse(b, chess.engine.Limit(depth=args.sf_depth), multipv=len(legal))
        cp = {d["pv"][0]: d["score"].pov(b.turn).score(mate_score=10000) for d in info}
        ranked = sorted(cp, key=cp.get, reverse=True)
        sf.append({"board": b, "cp": cp, "best": ranked[0]})
    print(f"Fixed positions: {len(FENS)} | SF depth {args.sf_depth} | sims {args.sims}\n")
    print(f"{'checkpoint':>12} {'sims':>5} {'best-agree':>11} {'med SF-rank':>12} "
          f"{'mean ACPL':>10} {'med Spearman':>13}")

    for spec in args.checkpoints:
        label, path = spec.split("=", 1)
        net = build_net(path, cfg, game, args.device)
        mcts = MCTS(net, game, cfg, args.device)
        for N in args.sims:
            cfg.num_simulations = N
            agree, ranks, acpls, sps = [], [], [], []
            for fen, gt in zip(FENS, sf):
                st = game.reset(); st.board = gt["board"].copy()
                legal_actions = game.legal_actions(st)
                obs = stack_with_history(game.to_tensor(st), [], cfg.history_frames)
                root = mcts.run(obs, legal_actions, add_noise=False)
                acts = np.asarray(root.child_actions)
                vis = np.asarray(root.child_visits, dtype=float)
                # decode each visited action to a move; pair with SF eval
                vv, cc = [], []
                top_v, top_move = -1.0, None
                for a, v in zip(acts.tolist(), vis.tolist()):
                    mv = _action_to_move(int(a), gt["board"])
                    if mv is None or mv not in gt["cp"]:
                        continue
                    vv.append(v); cc.append(gt["cp"][mv])
                    if v > top_v:
                        top_v, top_move = v, mv
                if top_move is None:
                    continue
                agree.append(1.0 if top_move == gt["best"] else 0.0)
                ranked = sorted(gt["cp"], key=gt["cp"].get, reverse=True)
                ranks.append(ranked.index(top_move) + 1)
                acpls.append(max(0, gt["cp"][gt["best"]] - gt["cp"][top_move]))
                if len(vv) >= 3 and np.std(vv) > 0:
                    sps.append(spearmanr(vv, cc).correlation)
            print(f"{label:>12} {N:>5} {np.mean(agree):>10.0%} {int(np.median(ranks)):>12} "
                  f"{np.mean(acpls):>10.0f} {np.median(sps):>13.2f}")
        print()
    eng.quit()


if __name__ == "__main__":
    main()
