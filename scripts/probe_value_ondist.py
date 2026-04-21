"""Probe the value head on in-distribution positions (sampled from the replay buffer).

The OOD probe (probe_value_collapse.py) uses hand-picked endgames the network
has likely never seen. This probe instead samples random positions from the
actual training distribution (the replay buffer) and asks:
  - Does the predicted value spread at all across the distribution?
  - Does it correlate with the *actual* game outcome from that position?
  - Does argmax ever leave the 0-bin on positions the network was trained on?

Run: .venv/bin/python scripts/probe_value_ondist.py --checkpoint <path.pt>
     (expects <path.buf> next to it)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.training.replay_buffer import ReplayBuffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    buf_path = Path(args.checkpoint).with_suffix(".buf")
    if not buf_path.exists():
        print(f"ERROR: replay buffer not found at {buf_path}")
        return
    buf = ReplayBuffer(max_size=10**9)
    buf.load(buf_path)

    print(f"Loaded {args.checkpoint} (step {ckpt.get('step', '?')})")
    print(f"Loaded buffer {buf_path} ({len(buf)} games, total_games_seen={buf.total_games})")

    # Sample (game_idx, step_idx) pairs uniformly over positions, not games
    game_lens = np.array([len(g) for g in buf.buffer])
    total_positions = int(game_lens.sum())
    n = min(args.num_samples, total_positions)
    print(f"Total positions: {total_positions}. Sampling {n}.")

    # Flatten into (game_idx, step_idx) using a CDF trick
    cdf = np.cumsum(game_lens)
    flat_pos = np.random.choice(total_positions, size=n, replace=False)
    game_idx = np.searchsorted(cdf, flat_pos, side="right")
    step_idx = flat_pos - np.where(game_idx > 0, cdf[np.clip(game_idx - 1, 0, None)], 0)

    obs_list = []
    outcomes_current_player = []   # +1 win, 0 draw, -1 loss FROM THE CURRENT PLAYER AT THAT STATE
    outcomes_game = []             # game outcome from p1 perspective: +1 p1 won, -1 p2 won, 0 draw
    for gi, si in zip(game_idx, step_idx):
        g = buf.buffer[int(gi)]
        obs_list.append(g.observations[int(si)])
        # Current player at step si: self-play alternates. g.current_player NOT recorded;
        # assume step 0 is white (p1) and strictly alternating.
        cur_player = 1 if int(si) % 2 == 0 else -1
        outcome = float(g.game_outcome)   # from p1 (white) perspective
        outcomes_game.append(outcome)
        # From current player's perspective
        outcomes_current_player.append(outcome * cur_player)

    obs_batch = torch.stack(obs_list).to(args.device)
    outcomes_cur = np.array(outcomes_current_player, dtype=np.float32)
    outcomes_game_arr = np.array(outcomes_game, dtype=np.float32)

    # Forward
    with torch.no_grad():
        _, _, value_logits = network.initial_inference_logits(obs_batch)
        probs = F.softmax(value_logits, dim=-1).cpu().numpy()
        _, _, value_scalar_t = network.initial_inference(obs_batch)
        value_scalar = value_scalar_t.squeeze(-1).cpu().numpy()

    center = config.value_support_size     # index of the "0" bin
    num_bins = 2 * config.value_support_size + 1
    argmaxes = probs.argmax(axis=1)

    print(f"\n--- In-distribution value probe ({n} positions) ---")

    # Predicted scalar spread
    print(f"predicted scalar: mean={value_scalar.mean():+.4f}  std={value_scalar.std():.4f}  "
          f"min={value_scalar.min():+.4f}  max={value_scalar.max():+.4f}")

    # Argmax distribution over bins
    bin_counts = np.bincount(argmaxes, minlength=num_bins)
    print(f"\nargmax bin distribution (support idx offset from center; center={center}):")
    for b in range(num_bins):
        if bin_counts[b] > 0:
            offset = b - center
            pct = 100 * bin_counts[b] / n
            print(f"  {offset:+3d}  {bin_counts[b]:5d}  ({pct:5.1f}%)")

    frac_at_zero = float((argmaxes == center).mean())
    print(f"\nfraction of positions with argmax at 0-bin: {frac_at_zero:.3f}")
    print(f"number of distinct argmax bins used: {int((bin_counts > 0).sum())} / {num_bins}")

    # Prob mass on {-1, 0, +1}
    near_zero_mass = probs[:, [center - 1, center, center + 1]].sum(axis=1)
    print(f"mean prob mass on {{-1,0,+1}} bins: {near_zero_mass.mean():.3f}  (close to 1 = all mass at center)")

    # Distribution of game outcomes in the sample (sanity)
    pct_draws = float((outcomes_game_arr == 0).mean())
    pct_wins_p1 = float((outcomes_game_arr > 0).mean())
    pct_wins_p2 = float((outcomes_game_arr < 0).mean())
    print(f"\nsampled positions' game outcomes (from p1 perspective): "
          f"draws={pct_draws:.2%}  p1 wins={pct_wins_p1:.2%}  p2 wins={pct_wins_p2:.2%}")

    # Correlation of predicted value with actual outcome (from current player's POV)
    # Positions from games that end in a loss for current player should have value < 0 on avg.
    if outcomes_cur.std() > 0:
        corr = float(np.corrcoef(outcomes_cur, value_scalar)[0, 1])
        print(f"correlation(predicted scalar, actual outcome from current player POV): {corr:+.3f}")
        print(f"  (1.0 = perfect; 0 = value head ignores game outcome; <0 = anti-correlated)")

    # Mean predicted value conditioned on actual outcome
    for label, mask in [("decisive-win for cur player", outcomes_cur > 0.5),
                        ("decisive-loss for cur player", outcomes_cur < -0.5),
                        ("draw", np.abs(outcomes_cur) < 0.5)]:
        if mask.sum() > 0:
            print(f"  {label:<32} n={int(mask.sum()):4d}  mean predicted scalar={value_scalar[mask].mean():+.4f}")


if __name__ == "__main__":
    main()
