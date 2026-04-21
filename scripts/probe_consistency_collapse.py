"""Probe whether the EfficientZero consistency loss is collapsing or healthy.

Loads a chess checkpoint, feeds a diverse set of board states through the
representation / projection / predictor, and reports pairwise cosine
similarity at each layer. High similarity across distinct inputs =>
collapse. Moderate/low similarity => healthy differentiation.

Run: .venv/bin/python scripts/probe_consistency_collapse.py --checkpoint <path>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
import torch
import torch.nn.functional as F

from src.config import get_config
from src.games.chess import ChessGame
from src.games.base import GameState
from src.model.muzero_net import MuZeroNetwork


DIVERSE_FENS = [
    # Opening, middlegame, endgame, positions across the game
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",                   # start
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",        # italian
    "r1bqk2r/pp1pbppp/2n2n2/2p1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 b kq - 5 5",      # castled italian
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",       # kiwipete
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                                   # endgame R vs R
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",   # complex middlegame
    "8/8/8/4k3/8/8/4K3/8 w - - 0 1",                                               # K vs K
    "4k3/4P3/4K3/8/8/8/8/8 b - - 0 1",                                             # K+P vs K
    "r3k3/8/8/8/8/8/8/R3K2R w KQq - 0 1",                                          # castling positions
    "8/8/8/3p4/3P4/8/8/8 w - - 0 1",                                               # empty-ish with pawns
    "q7/8/8/8/8/8/8/K6k w - - 0 1",                                                # Q vs nothing
    "8/8/8/8/8/8/k7/K1R5 w - - 0 1",                                               # K+R vs K
]


def cos_sim_matrix(x: torch.Tensor) -> np.ndarray:
    """Pairwise cosine similarity over rows of x (N, D)."""
    return F.cosine_similarity(x.unsqueeze(0), x.unsqueeze(1), dim=-1).cpu().numpy()


def summarize(name: str, sim: np.ndarray) -> None:
    n = sim.shape[0]
    off_diag = sim[~np.eye(n, dtype=bool)]
    print(f"\n{name}")
    print(f"  mean off-diag cos sim: {off_diag.mean():+.4f}")
    print(f"  max  off-diag cos sim: {off_diag.max():+.4f}")
    print(f"  min  off-diag cos sim: {off_diag.min():+.4f}")
    print(f"  std  off-diag cos sim: {off_diag.std():+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config = get_config("chess")
    game = ChessGame()

    network = MuZeroNetwork(
        observation_channels=game.num_planes,
        action_space_size=game.action_space_size,
        hidden_planes=config.hidden_planes,
        num_blocks=config.num_residual_blocks,
        latent_h=config.latent_h, latent_w=config.latent_w,
        input_h=game.board_size[0], input_w=game.board_size[1],
        fc_hidden=config.fc_hidden,
        value_support_size=config.value_support_size,
        reward_support_size=config.reward_support_size,
        use_consistency_loss=config.use_consistency_loss,
        proj_hid=config.proj_hid, proj_out=config.proj_out,
        pred_hid=config.pred_hid, pred_out=config.pred_out,
        use_scalar_transform=config.use_scalar_transform,
        value_target_scale=config.value_target_scale,
    ).to(args.device)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    network.load_state_dict(ckpt["model_state_dict"])
    network.eval()
    print(f"Loaded {args.checkpoint} (step {ckpt.get('step', '?')})")

    # Build diverse observation batch
    states = []
    for fen in DIVERSE_FENS:
        board = chess.Board(fen)
        states.append(GameState(
            board=board,
            current_player=1 if board.turn == chess.WHITE else -1,
        ))
    obs_batch = torch.stack([game.to_tensor(s) for s in states]).to(args.device)
    print(f"\nProbe batch: {len(states)} diverse positions")
    print(f"Obs shape: {tuple(obs_batch.shape)}")

    with torch.no_grad():
        # 1) Representation latents
        latent = network.representation(obs_batch)            # (N, C, H, W)
        latent_flat = latent.reshape(latent.shape[0], -1)     # (N, C*H*W)
        sim_repr = cos_sim_matrix(latent_flat)
        summarize("Representation latent (flattened)", sim_repr)

        # 2) Projection output (target branch, pre-detach)
        proj = network.projection(latent_flat)                # (N, proj_out)
        sim_proj = cos_sim_matrix(proj)
        summarize("Projection output (target branch)", sim_proj)

        # 3) Predictor output (online branch)
        pred = network.prediction_head(proj)                  # (N, pred_out)
        sim_pred = cos_sim_matrix(pred)
        summarize("Predictor output (online branch)", sim_pred)

    print("\nInterpretation:")
    print("  Healthy:  mean off-diag cos sim < ~0.5 (latents differentiate between inputs)")
    print("  Degenerate data (not SimSiam collapse):")
    print("            latent sim moderate/low, but projections near 1 — training data")
    print("            is uniform enough that all rolled states look alike.")
    print("  SimSiam collapse:")
    print("            latent sim low but projection sim ~1 — projector learned constant.")
    print("  Representation collapse (worst):")
    print("            latent sim ~1 across all inputs — representation net lost signal.")


if __name__ == "__main__":
    main()
