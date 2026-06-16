"""Policy-diffuseness + policy/value consistency probe.

For a sample of in-distribution positions (from the replay buffer next to a
checkpoint), answer:

  A. Is the predicted policy uniform-over-LEGAL (preference not learned) or is it
     leaking mass onto ILLEGAL moves? Compare full-softmax entropy vs
     legal-only-renormalized entropy, and how much mass sits on illegal moves.

  B. Does the model see any move as better than a draw? For every legal move,
     use the dynamics+prediction to compute the model's own action value
     Q(s,a) = reward(s,a) - discount * value(g(s,a))   [mover POV; matches MCTS]
     and report the spread across legal moves (max-min, std).

  C. Does the policy CHOOSE the moves it values? Correlate policy mass with Q,
     and report the value of the policy's top move vs the best/worst legal move.

CPU by default — safe to run alongside a live GPU training run.

Run: .venv/bin/python scripts/probe_policy_value_consistency.py \
        --checkpoint checkpoints/chess/<run>/checkpoint_<N>.pt
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
from src.training.replay_buffer import ReplayBuffer, stack_with_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-games", type=int, default=40)
    ap.add_argument("--max-plies", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = get_config("chess")
    game = ChessGame()
    discount = config.discount

    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    state_dict = ckpt["model_state_dict"]
    has_consistency = any(k.startswith("projection.") for k in state_dict)

    network = MuZeroNetwork(
        observation_channels=game.num_planes * config.history_frames,
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
        value_head_type=config.value_head_type,
        draw_score=config.draw_score,
        use_inverse_dynamics_loss=getattr(config, "use_inverse_dynamics_loss", False),
        inverse_dynamics_hidden=getattr(config, "inverse_dynamics_hidden", 256),
    ).to(args.device)
    network.load_state_dict(state_dict)
    network.eval()

    print(f"Loaded {args.checkpoint} (step {ckpt.get('step','?')})")
    print(f"Sourcing positions by ON-POLICY rollouts (model samples its own moves).")

    HF = config.history_frames
    full_ent, legal_ent, illegal_mass, top1_legal = [], [], [], []
    q_spread, q_std, q_top_move, q_best, q_worth, pv_corr = [], [], [], [], [], []
    n_legal_list = []
    root_vals = []

    n_games = args.num_games
    max_plies = args.max_plies
    rng = np.random.default_rng(args.seed)
    collected = 0

    for gi in range(n_games):
        state = game.reset()
        prior_obs = []
        for ply in range(max_plies):
            legal = game.legal_actions(state)
            if not legal or len(legal) < 2:
                break
            cur_obs = game.to_tensor(state)
            obs = stack_with_history(cur_obs, prior_obs, HF).unsqueeze(0).to(args.device)
            with torch.no_grad():
                hidden, policy_logits, root_v = network.initial_inference(obs)
                p_full = F.softmax(policy_logits, dim=-1)[0].cpu().numpy()

                # (A) diffuseness
                fe = -(p_full * np.log(p_full + 1e-12)).sum()
                lp = p_full[legal]
                lmass = lp.sum()
                lpn = lp / (lmass + 1e-12)
                le = -(lpn * np.log(lpn + 1e-12)).sum()

                # (B,C) per-legal-move Q via dynamics
                acts = torch.tensor(legal, dtype=torch.long, device=args.device)
                hid_rep = hidden.expand(len(legal), *hidden.shape[1:]).contiguous()
                _, reward, _, next_v = network.recurrent_inference(hid_rep, acts)
                q = (reward.squeeze(-1).cpu().numpy()
                     - discount * next_v.squeeze(-1).cpu().numpy())

            full_ent.append(fe); legal_ent.append(le)
            illegal_mass.append(1.0 - float(lmass))
            top1_legal.append(float(lpn.max()))
            n_legal_list.append(len(legal))
            root_vals.append(float(root_v.item()))
            q_spread.append(float(q.max() - q.min()))
            q_std.append(float(q.std()))
            choose = int(np.argmax(lpn))           # move the policy most prefers
            q_top_move.append(float(q[choose]))
            q_best.append(float(q.max())); q_worth.append(float(q.min()))
            if lpn.std() > 1e-9 and q.std() > 1e-9:
                pv_corr.append(float(np.corrcoef(lpn, q)[0, 1]))
            collected += 1

            # advance the game by SAMPLING from the model's own policy over legal moves
            a = int(rng.choice(legal, p=lpn))
            prior_obs.append(cur_obs)
            state, _, done = game.step(state, a)
            if done:
                break

    print(f"Collected {collected} on-policy positions over {n_games} games.")

    def stat(x):
        x = np.array(x)
        return f"mean={x.mean():+.3f} median={np.median(x):+.3f} std={x.std():.3f}"

    print(f"\n=== {len(full_ent)} positions, avg {np.mean(n_legal_list):.0f} legal moves "
          f"(log(avg-legal)={np.log(np.mean(n_legal_list)):.2f}) ===")

    print("\n--- A. POLICY DIFFUSENESS ---")
    print(f"full-softmax entropy   : {stat(full_ent)}")
    print(f"legal-only entropy     : {stat(legal_ent)}   (0=one-hot, log(#legal)≈uniform-over-legal)")
    print(f"mass on ILLEGAL moves  : {stat(illegal_mass)}")
    print(f"top legal-move prob    : {stat(top1_legal)}   (1.0=decisive, ~1/#legal=uniform)")
    print("  -> if legal-only entropy ≈ log(#legal) AND illegal-mass ≈ 0:"
          " uniform-over-legal (preference NOT learned)")
    print("  -> if full >> legal-only AND illegal-mass large: leaking onto illegal moves")

    print("\n--- B. DOES THE MODEL SEE ANY MOVE AS BETTER THAN DRAW? ---")
    print(f"root value             : {stat(root_vals)}")
    print(f"Q spread (max-min/pos) : {stat(q_spread)}   (0=all moves look identical)")
    print(f"Q std across legal/pos : {stat(q_std)}")
    print(f"best legal-move Q      : {stat(q_best)}")
    print(f"worst legal-move Q     : {stat(q_worth)}")

    print("\n--- C. DOES THE POLICY CHOOSE THE MOVES IT VALUES? ---")
    print(f"policy-mass vs Q corr  : {stat(pv_corr)}   (+1=policy prefers high-Q, 0=ignores value)")
    print(f"Q of policy's top move : {stat(q_top_move)}")
    gap = np.array(q_best) - np.array(q_top_move)
    print(f"Q(best) - Q(chosen)    : mean={gap.mean():+.3f} median={np.median(gap):+.3f}  "
          f"(0=policy picks the move it values most)")


if __name__ == "__main__":
    main()
