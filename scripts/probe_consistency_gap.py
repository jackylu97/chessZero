#!/usr/bin/env python3
"""Does the consistency loss (projection-space) leave the RAW dyn latent far from
repr(next) in the dimensions the VALUE HEAD reads?

For sibling moves at a root, compare:
  cos(dyn(h,a), repr(next_obs))           — RAW latent agreement (value head reads this)
  cos(project(dyn(h,a)), project(repr))   — PROJECTION agreement (what consistency trains)
  |V(dyn(h,a)) - V(repr(next_obs))|       — value the head actually outputs on each
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F, chess
from src.config import MuZeroConfig, get_config
from src.games.chess import ChessGame, _move_to_action
from src.games.base import GameState
from src.model.utils import wdl_to_scalar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_checkpoint_health import build_network

CKDIR = "checkpoints/chess/2026_06_19_cold2_pc"
FENS = [
    ("white up a queen", "r3k3/8/8/8/8/8/8/R2QK3 w - - 0 1"),
    ("white winning KP", "8/8/8/4k3/8/4P3/4K3/8 w - - 0 1"),
    ("tactical sicilian","r2q1rk1/pp1bppbp/2np1np1/8/3NP3/2N1BP2/PPPQB1PP/2KR3R w - - 0 11"),
]


def stack_obs(game, state, hf):
    cur = game.to_tensor(state)
    return torch.cat([cur] + [torch.zeros_like(cur) for _ in range(hf - 1)], 0)


@torch.no_grad()
def run(path, game, cfg, dev):
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(path, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev)
    hf = cfg.history_frames
    raw_cos, proj_cos, vgap = [], [], []
    for name, fen in FENS:
        board = chess.Board(fen)
        state = GameState(board=board, current_player=1 if board.turn == chess.WHITE else -1)
        obs = stack_obs(game, state, hf).unsqueeze(0).to(dev)
        h = net.representation(obs)
        for m in list(board.legal_moves)[:20]:
            a = _move_to_action(m, board.turn)
            at = torch.tensor([a], dtype=torch.long, device=dev)
            h_dyn, _ = net.dynamics(h, at)
            nb = board.copy(); nb.push(m)
            ns = GameState(board=nb, current_player=1 if nb.turn == chess.WHITE else -1)
            h_rep = net.representation(stack_obs(game, ns, hf).unsqueeze(0).to(dev))
            raw_cos.append(F.cosine_similarity(h_dyn.flatten(), h_rep.flatten(), dim=0).item())
            if net.projection is not None:
                p_dyn = net.project(h_dyn, with_grad=False)
                p_rep = net.project(h_rep, with_grad=False)
                proj_cos.append(F.cosine_similarity(p_dyn.flatten(), p_rep.flatten(), dim=0).item())
            _, vd = net.prediction(h_dyn)
            _, vr = net.prediction(h_rep)
            vgap.append(abs(wdl_to_scalar(vd.float(), cfg.draw_score).item()
                            - wdl_to_scalar(vr.float(), cfg.draw_score).item()))
    return np.mean(raw_cos), np.mean(proj_cos) if proj_cos else float("nan"), np.mean(vgap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = args.device
    print(f"{'step':>8}{'raw cos(dyn,repr)':>20}{'proj cos':>12}{'|V_dyn - V_repr|':>18}")
    for st in [1000, 6000, 30000]:
        r, p, v = run(os.path.join(CKDIR, f"checkpoint_{st}.pt"), game, cfg, args.device)
        print(f"{st:>8}{r:>20.4f}{p:>12.4f}{v:>18.4f}")


if __name__ == "__main__":
    main()
