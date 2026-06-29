"""Isolate each loss term's gradient contribution to the BODY (repr+dynamics).

The first probe showed the value head is alive and calibrated across positions.
This probe asks the sharper question the user cares about: how much signal does
the VALUE loss alone push into the representation/dynamics body vs the other
losses (policy, consistency, inverse, moves_left, reward)? If the value loss's
body-gradient is dwarfed by consistency/policy, the body never reorganizes its
latent to make value rankable WITHIN a position — even though the head itself
fits the across-position target.

Also: re-do the run with value_loss ONLY, and measure how much the value head's
gradient reaches the representation (the cold-start-Jacobian concern).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from scripts.eval_checkpoint_health import build_network
from src.training.trainer import MuZeroTrainer, _negative_cosine_similarity

CKPT_DIR = "checkpoints/chess/2026_06_19_cold2_pc"


def body_grad_norm(net):
    """grad L2 over representation + dynamics body only (the 'world model')."""
    sq = 0.0
    sq_repr = 0.0
    for name, p in net.named_parameters():
        if p.grad is None:
            continue
        if name.startswith("representation."):
            sq_repr += p.grad.detach().float().norm().item() ** 2
            sq += p.grad.detach().float().norm().item() ** 2
        elif name.startswith(("dynamics.conv_in.", "dynamics.bn_in.", "dynamics.blocks.")):
            sq += p.grad.detach().float().norm().item() ** 2
    return sq ** 0.5, sq_repr ** 0.5


BATCH = 64  # smaller batch for CPU speed; gradient-balance ratios are batch-size invariant


def build_batch(trainer):
    cfg = trainer.config
    beta = cfg.per_beta_init + (1.0 - cfg.per_beta_init) * (trainer.global_step / max(1, cfg.training_steps))
    batch, gi, isw = trainer.replay_buffer.sample_batch(
        BATCH, cfg.num_unroll_steps, cfg.td_steps, cfg.discount,
        trainer.game.action_space_size, alpha=cfg.per_alpha, beta=beta,
        value_head_type=cfg.value_head_type, history_frames=cfg.history_frames,
        eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
        warmstart_sample_frac=cfg.warmstart_sample_frac, decisive_sample_frac=cfg.decisive_sample_frac,
        q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio, selfplay_q_ratio=cfg.selfplay_q_ratio,
        repetition_penalty=cfg.repetition_penalty,
        repetition_penalty_window=int(cfg.repetition_penalty_window),
        repetition_penalty_decay=float(cfg.repetition_penalty_decay),
        build_legal_masks=False,
    )
    return batch, isw


def forward_losses(trainer, batch, isw):
    """Return per-term per-sample loss tensors (with graph attached) for the SAME forward."""
    cfg = trainer.config
    net = trainer.network
    dev = trainer.device
    obs = batch["observations"].to(dev)
    actions = batch["actions"].to(dev)
    target_values = batch["target_values"].to(dev)
    target_rewards = batch["target_rewards"].to(dev)
    target_policies = batch["target_policies"].to(dev)
    target_obs_mask = batch["target_obs_mask"].to(dev)
    target_observations = batch["target_observations"].to(dev)
    target_moves_left = batch["target_moves_left"].to(dev) if "target_moves_left" in batch else None
    num_planes = getattr(trainer.game, "num_planes", None)
    K = cfg.num_unroll_steps
    unroll_scale = 1.0
    outer_scale = 1.0 / (K + 1)

    terms = {}
    with autocast(device_type=dev.split(":")[0], enabled=cfg.use_amp):
        hidden, policy_logits, value_logits = net.initial_inference_logits(obs)
        policy_loss = trainer._policy_loss(policy_logits, target_policies[:, 0])
        value_loss = trainer._value_loss(value_logits, target_values[:, 0], net.value_support_size)
        reward_loss = torch.zeros(obs.shape[0], device=dev)
        consistency_loss = torch.zeros(obs.shape[0], device=dev)
        inverse_loss = torch.zeros(obs.shape[0], device=dev)
        moves_left_loss = trainer._moves_left_loss(net.predict_moves_left(hidden), target_moves_left[:, 0])
        for k in range(K):
            hidden_in = hidden
            hidden, reward_logits, policy_logits, value_logits = net.recurrent_inference_logits(hidden, actions[:, k])
            hidden.register_hook(lambda grad: grad * 0.5)
            mask_k = target_obs_mask[:, k + 1]
            policy_loss = policy_loss + unroll_scale * trainer._policy_loss(policy_logits, target_policies[:, k + 1]) * mask_k
            value_loss = value_loss + unroll_scale * trainer._value_loss(value_logits, target_values[:, k + 1], net.value_support_size) * mask_k
            reward_loss = reward_loss + unroll_scale * trainer._reward_loss(reward_logits, target_rewards[:, k + 1], net.reward_support_size) * mask_k
            tobs = target_observations[:, k + 1]
            if num_planes is not None:
                sf = torch.zeros_like(tobs); sf[:, :num_planes] = tobs[:, :num_planes]; tobs = sf
            dyn_proj = net.project(hidden, with_grad=True)
            with torch.no_grad():
                tgt_proj = net.project(net.representation(tobs), with_grad=False)
            consistency_loss = consistency_loss + unroll_scale * (_negative_cosine_similarity(dyn_proj, tgt_proj) * mask_k)
            inv_logits = net.predict_inverse_action(hidden_in, hidden)
            inverse_loss = inverse_loss + unroll_scale * (F.cross_entropy(inv_logits, actions[:, k], reduction="none") * mask_k)
            moves_left_loss = moves_left_loss + unroll_scale * (trainer._moves_left_loss(net.predict_moves_left(hidden), target_moves_left[:, k + 1]) * mask_k)

    isw_t = torch.tensor(isw, device=dev)
    vw = cfg.value_loss_weight_warmstart  # all self-play here, but weight is 1.0 either way
    terms["policy"] = (isw_t * outer_scale * policy_loss).mean()
    terms["value"] = (isw_t * outer_scale * 1.0 * value_loss).mean()  # value_weight=1.0
    terms["reward"] = (isw_t * outer_scale * reward_loss).mean()
    terms["consistency"] = (isw_t * outer_scale * cfg.consistency_loss_weight * consistency_loss).mean()
    terms["inverse"] = (isw_t * outer_scale * cfg.inverse_dynamics_loss_weight * inverse_loss).mean()
    terms["moves_left"] = (isw_t * outer_scale * cfg.moves_left_loss_weight * moves_left_loss).mean()
    return terms


def main():
    dev = "cpu"
    game = ChessGame()
    cfg = get_config("chess_small"); cfg.device = dev
    torch.serialization.add_safe_globals([MuZeroConfig])

    print(f"loss weights: consistency={cfg.consistency_loss_weight} inverse={cfg.inverse_dynamics_loss_weight} "
          f"moves_left={cfg.moves_left_loss_weight} value(warm/self)={cfg.value_loss_weight_warmstart}/{cfg.value_loss_weight_selfplay}")

    for step in [6000, 30000]:
        ckpt = torch.load(os.path.join(CKPT_DIR, f"checkpoint_{step}.pt"), map_location=dev, weights_only=True)
        net = build_network(ckpt, game, cfg, dev)
        trainer = MuZeroTrainer(cfg, game, net, run_id=f"diag{step}", device=dev,
                                log_dir="/tmp/diag_runs", checkpoints_dir="/tmp/diag_ckpts")
        trainer.global_step = step
        trainer.replay_buffer.load(os.path.join(CKPT_DIR, f"checkpoint_{step}.buf"), game=game)
        net.train()

        batch, isw = build_batch(trainer)

        print(f"\n{'='*78}\nCHECKPOINT {step}  (buffer {len(trainer.replay_buffer)} games, "
              f"{sum(1 for g in trainer.replay_buffer.buffer if g.external_values)} warmstart)\n{'='*78}")

        # Per-term BODY gradient (retain graph; backward each term separately).
        terms = forward_losses(trainer, batch, isw)
        keys = list(terms.keys())
        print(f"  {'term':12s} {'loss_value':>12s} {'body_gradL2':>12s} {'repr_gradL2':>12s}")
        for i, kterm in enumerate(keys):
            trainer.optimizer.zero_grad()
            retain = (i < len(keys) - 1)
            terms[kterm].backward(retain_graph=retain)
            bg, rg = body_grad_norm(net)
            print(f"  {kterm:12s} {float(terms[kterm].item()):12.5f} {bg:12.5f} {rg:12.5f}")

        # Value-ONLY: how much does value loss reorganize the body relative to itself?
        # And what fraction of value's total grad lands in the head vs the body?
        terms2 = forward_losses(trainer, batch, isw)
        trainer.optimizer.zero_grad()
        terms2["value"].backward()
        head_sq = 0.0; body_sq = 0.0
        for name, p in net.named_parameters():
            if p.grad is None:
                continue
            n = p.grad.detach().float().norm().item() ** 2
            if name.startswith("prediction.value_head."):
                head_sq += n
            elif name.startswith(("representation.", "dynamics.conv_in.", "dynamics.bn_in.", "dynamics.blocks.")):
                body_sq += n
        hv = head_sq ** 0.5; bv = body_sq ** 0.5
        print(f"  [value-only backward]  head_gradL2={hv:.5f}  body_gradL2={bv:.5f}  "
              f"body/head ratio={bv/max(hv,1e-9):.3f}")


if __name__ == "__main__":
    main()
