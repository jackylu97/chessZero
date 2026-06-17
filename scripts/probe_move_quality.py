"""Move-quality probe: on the model's OWN self-play positions (on-distribution),
score the move the model actually plays (MCTS) against Stockfish.

Addresses two issues with eval-vs-random + the strict top-1/top-3 check:
  - positions are the model's own (8 random opening plies then MCTS), not OOD
    random-opponent positions;
  - the bar is "is the model's move in Stockfish's top-K" + centipawn loss (ACPL),
    not "exactly matches Stockfish's single best move".

Run: .venv/bin/python scripts/probe_move_quality.py --checkpoint <path.pt>
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess, chess.engine

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.mcts.mcts import MCTS, select_action
from src.training.replay_buffer import stack_with_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--positions", type=int, default=25)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--multipv", type=int, default=15)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg = get_config("chess"); cfg.device = "cpu"; cfg.num_simulations = args.sims
    game = ChessGame()
    torch.serialization.add_safe_globals([MuZeroConfig])
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    sd = ck["model_state_dict"]
    net = MuZeroNetwork(
        observation_channels=game.num_planes * cfg.history_frames,
        action_space_size=game.action_space_size, hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w,
        input_h=8, input_w=8, fc_hidden=cfg.fc_hidden,
        value_support_size=cfg.value_support_size, reward_support_size=cfg.reward_support_size,
        use_consistency_loss=any(k.startswith("projection.") for k in sd),
        proj_hid=cfg.proj_hid, proj_out=cfg.proj_out, pred_hid=cfg.pred_hid, pred_out=cfg.pred_out,
        use_scalar_transform=cfg.use_scalar_transform, value_target_scale=cfg.value_target_scale,
        value_head_type=cfg.value_head_type, draw_score=cfg.draw_score,
        use_inverse_dynamics_loss=any(k.startswith("inverse_dynamics_head.") for k in sd),
        inverse_dynamics_hidden=getattr(cfg, "inverse_dynamics_hidden", 256),
    )
    net.load_state_dict(sd); net.eval()
    mcts = MCTS(net, game, cfg, "cpu")
    eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    print(f"Loaded {args.checkpoint} (step {ck.get('step','?')}); sims={args.sims}, "
          f"SF depth={args.sf_depth} multipv={args.multipv}\n")

    state = game.reset()
    frames = []
    n_open = 8
    ranks, cp_losses, in_topk, outside = [], [], 0, 0
    collected = 0
    ply = 0
    while collected < args.positions and ply < 300:
        legal = game.legal_actions(state)
        if len(legal) < 2:
            break
        if ply < n_open:                                  # random opening (matches self-play)
            action = int(np.random.choice(legal))
        else:
            obs = stack_with_history(game.to_tensor(state), frames, cfg.history_frames)
            root = mcts.run(obs, legal, add_noise=False)
            action, _ = select_action(root, temperature=0)
            # Stockfish scoring of the move the model is about to play.
            board = state.board.copy()
            info = eng.analyse(board, chess.engine.Limit(depth=args.sf_depth), multipv=args.multipv)
            sf = [(d["pv"][0], d["score"].pov(board.turn).score(mate_score=10000)) for d in info]
            best_cp = sf[0][1]
            frames.append(game.to_tensor(state))
            nxt, _, done = game.step(state, action)
            mv = nxt.board.move_stack[-1]
            rank = next((i for i, (m, _) in enumerate(sf) if m == mv), None)
            if rank is not None:
                in_topk += 1; ranks.append(rank + 1)
                cp_losses.append(max(0, best_cp - sf[rank][1]))
                tag = f"rank {rank+1}/{len(sf)}  cp_loss {max(0,best_cp-sf[rank][1])}"
            else:
                outside += 1; tag = f">{args.multipv} (not in SF top-{args.multipv})"
            print(f"  ply {ply:3d}  model={board.san(mv):6s}  SF#1={board.san(sf[0][0]):6s}  {tag}")
            collected += 1
            state = nxt
            if done: state = game.reset(); frames = []; ply = 0; continue
            ply += 1
            continue
        frames.append(game.to_tensor(state))
        state, _, done = game.step(state, action)
        if done: state = game.reset(); frames = []; ply = 0; continue
        ply += 1
    eng.quit()

    n = collected
    print(f"\n=== {n} model moves on its own (on-distribution) positions ===")
    print(f"  in Stockfish top-{args.multipv}: {in_topk}/{n} = {in_topk/n:.0%}   (outside: {outside})")
    if ranks:
        print(f"  rank among SF moves (top-{args.multipv} only): median {int(np.median(ranks))}  mean {np.mean(ranks):.1f}")
    if cp_losses:
        allcp = cp_losses + [300]*outside  # floor outside-top-K at ~300cp for a coarse ACPL
        print(f"  centipawn loss vs SF best (top-{args.multipv}): median {int(np.median(cp_losses))}  mean {np.mean(cp_losses):.0f}")
        print(f"  coarse ACPL (outside-top-{args.multipv} floored at 300cp): {np.mean(allcp):.0f}")


if __name__ == "__main__":
    main()
