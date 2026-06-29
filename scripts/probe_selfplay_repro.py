"""Reproduce the ACTUAL self-play loop at step 30000 and compare the played
ply-0 action distribution to the stored policy target. If the live loop also
produces ~99% e2e4 at ply 0, there is a temperature/selection bug in the
GPU-resident path; if it produces ~65% (matching the soft policy), the buffer
discrepancy comes from elsewhere.
"""
import argparse, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess

from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.training.self_play import play_games_parallel_gpu_resident
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from eval_checkpoint_health import build_network


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--step", type=int, default=30000)
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()

    device = "cuda"
    cfg = get_config("chess_small")
    cfg.device = device
    cfg.num_self_play_games = args.n
    cfg.num_parallel_games = args.n
    cfg.max_plies = 2  # only need ply-0 selection; cap to finish instantly
    game = ChessGame()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    net = build_network(ckpt, game, cfg, device)

    from src.training.self_play import get_temperature
    print(f"step={args.step} get_temperature={get_temperature(args.step, cfg)} "
          f"temp_drop={cfg.temperature_drop_step} random_open={cfg.random_opening_plies}")

    torch.manual_seed(0); np.random.seed(0)
    games = play_games_parallel_gpu_resident(net, cfg, args.n, device=device,
                                             training_step=args.step)
    print(f"produced {len(games)} games")

    # ply-0 action distribution + stored policy mass on the played action
    act0 = Counter()
    mass_on_played = []
    top_mass = []
    argmax_rate = []
    for g in games:
        if len(g.actions) == 0:
            continue
        a = int(g.actions[0])
        act0[a] += 1
        p = np.asarray(g.policies[0], dtype=float)
        if p.sum() > 0:
            mass_on_played.append(float(p[a]))
            top_mass.append(float(p.max()))
            argmax_rate.append(1.0 if a == int(np.argmax(p)) else 0.0)
    tot = sum(act0.values())
    top_a, top_n = act0.most_common(1)[0]
    mv = _action_to_move(int(top_a), chess.Board())
    ent = -sum((c / tot) * np.log2(c / tot) for c in act0.values())
    print(f"\nLIVE ply-0 ACTION dist: top={mv.uci() if mv else top_a} ({top_n/tot:.1%})  "
          f"entropy={ent:.2f}b  unique={len(act0)}")
    print(f"LIVE ply-0 POLICY: mean top_mass={np.mean(top_mass):.1%}  "
          f"mean mass_on_played={np.mean(mass_on_played):.1%}  "
          f"argmax_rate={np.mean(argmax_rate):.1%}")
    print("(If argmax_rate >> top_mass, the live loop plays sharper than T=1.0 -> bug.)")
    tops = [(_action_to_move(int(a), chess.Board()).uci(), f"{n/tot:.0%}")
            for a, n in act0.most_common(6)]
    print("top sampled:", tops)


if __name__ == "__main__":
    main()
