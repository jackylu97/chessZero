#!/usr/bin/env python3
"""FAST sibling-value blindness probe — no buffer reconstruction.

For a handful of constructed FENs spanning material imbalance, compute, at each
checkpoint, for the legal moves at that root:
  (A) via DYNAMICS:  V_a = prediction_value(dynamics(repr(obs), a))
  (B) via REPRESENTATION (ground truth path the net would see): V'_a =
      prediction_value(repr(obs_after_move))
and report the std of V_a and V'_a across siblings, plus their correlation.

If std(V_a) << std(V'_a) the WITHIN-position signal is lost in the DYNAMICS+VALUE
path (what MCTS actually uses), even though the representation CAN tell the moves
apart. If BOTH stds are tiny, the VALUE HEAD itself is position-blind.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F, chess
from src.config import MuZeroConfig, get_config
from src.games.chess import ChessGame, _move_to_action, _action_to_move
from src.games.base import GameState
from src.model.muzero_net import MuZeroNetwork
from src.model.utils import wdl_to_scalar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_checkpoint_health import build_network

CKDIR = "checkpoints/chess/2026_06_19_cold2_pc"
FENS = [
    ("white up a queen", "r3k3/8/8/8/8/8/8/R2QK3 w - - 0 1"),
    ("white up a rook",  "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"),
    ("balanced midgame", "r1bq1rk1/ppp2ppp/2n1pn2/2bp4/2BP4/2N1PN2/PPP1QPPP/R1B2RK1 w - - 0 8"),
    ("white winning KP", "8/8/8/4k3/8/4P3/4K3/8 w - - 0 1"),
    ("white losing",     "3qk3/8/8/8/8/8/8/4K3 w - - 0 1"),
    ("tactical sicilian","r2q1rk1/pp1bppbp/2np1np1/8/3NP3/2N1BP2/PPPQB1PP/2KR3R w - - 0 11"),
]


def stack_obs(game, state, hf):
    cur = game.to_tensor(state)
    frames = [cur] + [torch.zeros_like(cur) for _ in range(hf - 1)]
    return torch.cat(frames, 0)


@torch.no_grad()
def run(path, game, cfg, dev):
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(path, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev)
    hf = cfg.history_frames
    dyn_stds, repr_stds, corrs, root_vs = [], [], [], []
    for name, fen in FENS:
        board = chess.Board(fen)
        state = GameState(board=board, current_player=1 if board.turn == chess.WHITE else -1)
        obs = stack_obs(game, state, hf).unsqueeze(0).to(dev)
        h = net.representation(obs)
        _, vl = net.prediction(h)
        rootv = wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).item()
        root_vs.append(rootv)
        legal = list(board.legal_moves)
        acts = [_move_to_action(m, board.turn) for m in legal]
        # via dynamics
        at = torch.tensor(acts, dtype=torch.long, device=dev)
        hn = h.expand(len(acts), *h.shape[1:]).contiguous()
        dyn, _ = net.dynamics(hn, at)
        _, vla = net.prediction(dyn)
        v_dyn = wdl_to_scalar(vla.float(), draw_score=cfg.draw_score).cpu().numpy()
        # via representation of actual next position (negate: opponent POV)
        v_repr = []
        for m in legal:
            nb = board.copy(); nb.push(m)
            ns = GameState(board=nb, current_player=1 if nb.turn == chess.WHITE else -1)
            nobs = stack_obs(game, ns, hf).unsqueeze(0).to(dev)
            _, nvl = net.prediction(net.representation(nobs))
            # next position is opponent-to-move; negate to root STM POV
            v_repr.append(-wdl_to_scalar(nvl.float(), draw_score=cfg.draw_score).item())
        v_repr = np.asarray(v_repr)
        dyn_stds.append(float(np.std(v_dyn)))
        repr_stds.append(float(np.std(v_repr)))
        if v_dyn.std() > 1e-9 and v_repr.std() > 1e-9:
            corrs.append(float(np.corrcoef(v_dyn, v_repr)[0, 1]))
        print(f"    {name:20s} root V={rootv:+.3f} nlegal={len(legal):2d}  "
              f"std(V_dyn)={np.std(v_dyn):.4f}  std(V_repr_next)={np.std(v_repr):.4f}  "
              f"range_repr=[{v_repr.min():+.3f},{v_repr.max():+.3f}]")
    return (np.mean(dyn_stds), np.mean(repr_stds),
            np.mean(corrs) if corrs else float("nan"), np.std(root_vs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", default="1000,6000,30000")
    args = ap.parse_args()
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = args.device
    rows = []
    for st in [int(s) for s in args.steps.split(",")]:
        print(f"\n=== step {st} ===")
        d, r, c, rvs = run(os.path.join(CKDIR, f"checkpoint_{st}.pt"), game, cfg, args.device)
        rows.append((st, d, r, c, rvs))
    print(f"\n{'step':>8}{'mean std(V_dyn)':>18}{'mean std(V_repr)':>18}{'corr dyn/repr':>16}{'cross-pos rootV std':>22}")
    for st, d, r, c, rvs in rows:
        print(f"{st:>8}{d:>18.4f}{r:>18.4f}{c:>16.3f}{rvs:>22.4f}")


if __name__ == "__main__":
    main()
