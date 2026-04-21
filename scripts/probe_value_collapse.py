"""Probe whether the value head has collapsed to outputting 'draw' for all positions.

Loads a chess checkpoint, runs initial_inference_logits on a diverse set of
positions, and prints each position's value distribution + decoded scalar.

If the value head has collapsed to the draw basin, all positions — including
obviously won/lost ones (K+R vs K, K alone vs Q) — produce near-identical
distributions peaked at the 0-bucket. If it has learned chess value,
won/lost positions shift probability mass away from 0.

Run: .venv/bin/python scripts/probe_value_collapse.py --checkpoint <path>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
import torch
import torch.nn.functional as F

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.games.base import GameState
from src.model.muzero_net import MuZeroNetwork


# (fen, label, expected_sign) — expected_sign from white-to-move perspective:
#   +1 = white should evaluate clearly winning
#    0 = balanced / drawish
#   -1 = white should evaluate clearly losing
LABELED_FENS = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",               "start",           0),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",    "italian",         0),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",   "kiwipete",        0),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                              "endgame R vs R",  0),
    ("8/8/8/4k3/8/8/4K3/8 w - - 0 1",                                          "K vs K (draw)",   0),
    ("8/8/8/8/8/8/k7/K1R5 w - - 0 1",                                          "K+R vs K (wins)", +1),
    ("8/8/8/8/4k3/8/3Q4/3K4 w - - 0 1",                                        "K+Q vs K (wins)", +1),
    ("q7/8/8/8/8/8/8/K6k w - - 0 1",                                           "K vs K+Q (loses)",-1),
    ("8/8/8/8/8/k7/p7/K7 w - - 0 1",                                           "K vs K+P (loses)",-1),
    ("8/P7/8/8/8/8/k7/K7 w - - 0 1",                                           "K+P vs K (wins)", +1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--topk", type=int, default=5, help="top-K support bins to show")
    args = ap.parse_args()

    config = get_config("chess")
    game = ChessGame()

    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    state_dict = ckpt["model_state_dict"]
    has_consistency = any(k.startswith("projection.") for k in state_dict)

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
        use_consistency_loss=has_consistency,
        proj_hid=config.proj_hid, proj_out=config.proj_out,
        pred_hid=config.pred_hid, pred_out=config.pred_out,
        use_scalar_transform=config.use_scalar_transform,
        value_target_scale=config.value_target_scale,
    ).to(args.device)
    network.load_state_dict(state_dict)
    network.eval()
    print(f"Loaded {args.checkpoint} (step {ckpt.get('step', '?')})")
    n_bins = 2 * config.value_support_size + 1
    print(f"value_support_size = {config.value_support_size} ({n_bins} bins, indices 0..{n_bins-1}, 0-value at index {config.value_support_size})")

    # Build batch
    states = []
    for fen, _, _ in LABELED_FENS:
        board = chess.Board(fen)
        states.append(GameState(board=board, current_player=1 if board.turn == chess.WHITE else -1))
    obs_batch = torch.stack([game.to_tensor(s) for s in states]).to(args.device)

    with torch.no_grad():
        _, _, value_logits = network.initial_inference_logits(obs_batch)
        probs = F.softmax(value_logits, dim=-1).cpu().numpy()   # (N, 21)
        _, _, value_scalar = network.initial_inference(obs_batch)
        value_scalar = value_scalar.squeeze(-1).cpu().numpy()

    center = config.value_support_size
    n_bins = 2 * config.value_support_size + 1

    # Per-position report
    print(f"\n{'position':<26} {'expect':>6}  {'scalar':>7}  {'argmax':>6}  top-{args.topk} (idx:prob)")
    print("-" * 110)
    for i, (fen, label, expected) in enumerate(LABELED_FENS):
        p = probs[i]
        argmax = int(np.argmax(p))
        top_idx = np.argsort(-p)[: args.topk]
        top_str = "  ".join(f"{int(k)-center:+d}:{p[k]:.2f}" for k in top_idx)
        print(f"{label:<26} {expected:>+6d}  {value_scalar[i]:>+7.3f}  {argmax-center:>+6d}  {top_str}")

    # Collapse diagnostics
    print("\nCollapse diagnostics:")
    argmaxes = probs.argmax(axis=1)
    all_same_argmax = len(set(argmaxes.tolist())) == 1
    print(f"  all positions argmax to same bin?  {all_same_argmax}  (argmax set: {sorted(set(argmaxes.tolist() - np.int64(center)))})")

    mean_dist = probs.mean(axis=0)
    tv_from_mean = 0.5 * np.abs(probs - mean_dist[None, :]).sum(axis=1)
    print(f"  mean total-variation from batch-mean distribution: {tv_from_mean.mean():.4f}  (0 = every position has identical dist)")
    print(f"    max TV from mean: {tv_from_mean.max():.4f}  min: {tv_from_mean.min():.4f}")

    # Spearman-style: are value_scalars correlated with expected signs?
    expected = np.array([e for _, _, e in LABELED_FENS], dtype=np.float32)
    if expected.std() > 0 and value_scalar.std() > 0:
        corr = float(np.corrcoef(expected, value_scalar)[0, 1])
        print(f"  correlation (expected_sign vs predicted scalar): {corr:+.3f}  (1.0 = perfect; <0.2 = value head ignores material)")

    # Mass concentration around 0-bin
    near_zero_mass = probs[:, [center-1, center, center+1]].sum(axis=1)
    print(f"  mean prob mass on {{-1, 0, +1}} bins: {near_zero_mass.mean():.3f}  (close to 1.0 = value head sits at 0)")

    print("\nInterpretation:")
    print("  Healthy: scalar predictions track expected signs, correlation > 0.5,")
    print("           losing/winning positions have argmax != 0 bin.")
    print("  Collapsed: all argmaxes on 0 bin, near-zero correlation, TV-from-mean < 0.05.")


if __name__ == "__main__":
    main()
