"""Quantify representation collapse over training on a FIXED set of clearly-
distinct positions (varied material/structure). Reports per checkpoint:
 - per-dim latent std (variance across positions)
 - mean pairwise cosine similarity (1.0 => all positions map to same direction)
 - effective rank of the latent matrix (participation ratio of singular values)

A latent that increasingly maps distinct positions to one point (cos->1,
eff-rank->1, std->0) is a collapsed representation: the backbone has stopped
encoding position identity. Run on a diverse fixed FEN set so the input
diversity is held constant — only the network changes.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
import chess

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from eval_checkpoint_health import build_network

CKDIR = "checkpoints/chess/2026_06_19_cold2_pc"
DEV = "cpu"

# Diverse, clearly distinct positions (material + structure vary widely).
FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 3",
    "4k3/8/8/8/8/8/8/3QK3 w - - 0 1",          # K+Q vs K (winning)
    "3qk3/8/8/8/8/8/8/4K3 w - - 0 1",          # K vs K+Q (losing)
    "8/5k2/8/8/8/8/2K5/8 w - - 0 1",           # bare kings
    "r3k2r/ppp2ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPP2PPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  # rook endgame
    "rnbq1rk1/pp2bppp/4pn2/2pp4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 1",
    "2kr3r/ppp1qppp/2n5/3p4/3P4/2P2N2/PP3PPP/R1BQ1RK1 b - - 0 1",
    "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",         # K+P vs K
    "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",     # balanced K+3P
    "r4rk1/1bp1qppp/p1np1n2/1p2p3/4P3/1BPP1N1P/PP3PP1/RNBQR1K1 w - - 0 1",
    "8/8/8/3k4/8/3K4/8/7R w - - 0 1",          # K+R vs K (winning)
    "8/8/8/3k4/8/3K4/8/7r w - - 0 1",          # K vs K+R (losing)
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
]


def eff_rank(X):
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))  # participation ratio (entropy form)


def main():
    torch.serialization.add_safe_globals([MuZeroConfig])
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = DEV
    hf = cfg.history_frames
    # Build single-frame obs, then zero-pad history (constant across checkpoints).
    obs = []
    for fen in FENS:
        st = game.reset_from_fen(fen)
        cur = game.to_tensor(st)
        stacked = torch.cat([cur] + [torch.zeros_like(cur)] * (hf - 1), dim=0)
        obs.append(stacked)
    X_in = torch.stack(obs)

    print(f"{len(FENS)} distinct positions; tracking latent collapse over training\n")
    print(f"{'step':>6} {'perdim_std':>11} {'mean_cos':>9} {'eff_rank':>9} {'Vstd':>7} {'Vrange':>16}")
    for step in [1000, 2000, 4000, 6000, 10000, 20000, 30000]:
        ck = torch.load(f"{CKDIR}/checkpoint_{step}.pt", map_location=DEV, weights_only=True)
        net = build_network(ck, game, cfg, DEV)
        with torch.no_grad():
            H = net.representation(X_in)
            _, vl = net.prediction(H)
            from src.model.utils import wdl_to_scalar
            V = wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).cpu().numpy()
        Hf = H.flatten(1).cpu().numpy()
        per_dim_std = Hf.std(0).mean()
        Hn = F.normalize(torch.from_numpy(Hf), dim=-1)
        C = (Hn @ Hn.T).numpy()
        mean_cos = C[~np.eye(len(C), dtype=bool)].mean()
        er = eff_rank(Hf)
        print(f"{step:>6} {per_dim_std:>11.4f} {mean_cos:>9.3f} {er:>9.2f} {V.std():>7.3f} "
              f"[{V.min():+.2f},{V.max():+.2f}]")


if __name__ == "__main__":
    main()
