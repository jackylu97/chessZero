"""Value-head gradient-flow probe.

Loads checkpoint_6000 + its real .buf, runs an ACTUAL train step through the
real MuZeroTrainer._train_step machinery (real sample_batch, real losses, real
0.5 hidden hook, real AMP), and dumps per-parameter-group gradient L2 norms with
fine granularity. Answers: is the value head dead/starved? Is the 0.5 hook
double-applied? What's the GradScaler skip rate? Is weight_decay eroding the
value head? What's the value-head init?

Run on CPU (safe alongside GPU training).
"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from scripts.eval_checkpoint_health import build_network

CKPT_DIR = "checkpoints/chess/2026_06_19_cold2_pc"


def fine_grad_norms(net):
    """Per-parameter grad L2 norm, keyed by full name."""
    out = {}
    for name, p in net.named_parameters():
        if p.grad is None:
            out[name] = None
        else:
            out[name] = p.grad.detach().float().norm().item()
    return out


def group_grad_norms(net):
    """Sub-network grad norms by coarse prefix (mirrors trainer._GRAD_GROUPS plus value-head internals)."""
    groups = {
        "representation": ("representation.",),
        "dynamics_body": ("dynamics.conv_in.", "dynamics.bn_in.", "dynamics.blocks."),
        "dynamics_action_embed": ("dynamics.action_embedding.",),
        "dynamics_reward_head": ("dynamics.reward_head.",),
        "prediction_policy_head": ("prediction.policy_head.",),
        "prediction_value_head": ("prediction.value_head.",),
        "moves_left_head": ("moves_left_head.",),
        "projection": ("projection.", "prediction_head."),
        "inverse_dynamics_head": ("inverse_dynamics_head.",),
    }
    acc = {k: 0.0 for k in groups}
    nparam = {k: 0 for k in groups}
    none_count = {k: 0 for k in groups}
    for name, p in net.named_parameters():
        for key, prefixes in groups.items():
            if name.startswith(prefixes):
                nparam[key] += 1
                if p.grad is None:
                    none_count[key] += 1
                else:
                    acc[key] += p.grad.detach().float().norm().item() ** 2
                break
    return {k: (acc[k] ** 0.5, nparam[k], none_count[k]) for k in groups}


def run_one_train_step(trainer, capture_grads=True, n_hook_calls_box=None):
    """Replicate trainer._train_step EXACTLY through backward, then return grads.

    Returns (loss_dict, fine_grads_after_unscale_before_clip, grad_norm_total,
             value_logits_k0, target_values_k0, scaler_scale_before, step_taken).
    """
    cfg = trainer.config
    net = trainer.network
    net.train()

    beta = cfg.per_beta_init + (1.0 - cfg.per_beta_init) * (
        trainer.global_step / max(1, cfg.training_steps)
    )
    batch, game_indices, is_weights = trainer.replay_buffer.sample_batch(
        cfg.batch_size, cfg.num_unroll_steps, cfg.td_steps, cfg.discount,
        trainer.game.action_space_size,
        alpha=cfg.per_alpha, beta=beta,
        value_head_type=getattr(cfg, "value_head_type", "support"),
        history_frames=getattr(cfg, "history_frames", 1),
        eval_to_wdl_alpha=getattr(cfg, "eval_to_wdl_alpha", 4.0),
        eval_to_wdl_beta=getattr(cfg, "eval_to_wdl_beta", 2.0),
        warmstart_sample_frac=getattr(cfg, "warmstart_sample_frac", 0.0),
        decisive_sample_frac=getattr(cfg, "decisive_sample_frac", 0.0),
        q_ratio=getattr(cfg, "q_ratio", 0.0),
        warmstart_q_ratio=getattr(cfg, "warmstart_q_ratio", None),
        selfplay_q_ratio=getattr(cfg, "selfplay_q_ratio", None),
        repetition_penalty=getattr(cfg, "repetition_penalty", 0.0),
        repetition_penalty_window=int(getattr(cfg, "repetition_penalty_window", 0)),
        repetition_penalty_decay=float(getattr(cfg, "repetition_penalty_decay", 0.0)),
        build_legal_masks=bool(getattr(cfg, "mask_illegal_policy", False)),
    )
    dev = trainer.device
    obs = batch["observations"].to(dev)
    actions = batch["actions"].to(dev)
    target_values = batch["target_values"].to(dev)
    target_rewards = batch["target_rewards"].to(dev)
    target_policies = batch["target_policies"].to(dev)
    is_weights_t = torch.tensor(is_weights, device=dev)
    target_obs_mask = batch["target_obs_mask"].to(dev)
    is_warm = batch["is_warmstart"].to(dev)

    use_consistency = bool(getattr(cfg, "use_consistency_loss", False))
    single_frame_consistency = bool(getattr(cfg, "consistency_single_frame_target", False))
    use_inverse = bool(getattr(cfg, "use_inverse_dynamics_loss", False))
    inverse_weight = float(getattr(cfg, "inverse_dynamics_loss_weight", 1.0))
    num_planes = getattr(trainer.game, "num_planes", None)
    if use_consistency:
        target_observations = batch["target_observations"].to(dev)

    w_warm = getattr(cfg, "value_loss_weight_warmstart", None)
    w_self = getattr(cfg, "value_loss_weight_selfplay", None)
    if w_warm is not None and w_self is not None:
        value_weight = torch.where(is_warm, w_warm, w_self)
    else:
        value_weight = cfg.value_loss_weight

    use_moves_left = (bool(getattr(cfg, "use_moves_left", False))
                      and getattr(net, "moves_left_head", None) is not None)
    moves_left_weight = float(getattr(cfg, "moves_left_loss_weight", 0.0))
    target_moves_left = (batch["target_moves_left"].to(dev)
                         if use_moves_left and "target_moves_left" in batch else None)

    K = cfg.num_unroll_steps
    use_root_heavy = bool(getattr(cfg, "use_root_heavy_loss", False))
    unroll_scale = (1.0 / K if K > 0 else 0.0) if use_root_heavy else 1.0
    outer_scale = 1.0 if use_root_heavy else 1.0 / (K + 1)

    policy_loss_fn = trainer._policy_loss

    # Track hidden-hook invocations to detect double-apply.
    hook_calls = []

    from torch.amp import autocast
    with autocast(device_type=dev.split(":")[0], enabled=cfg.use_amp):
        hidden, policy_logits, value_logits = net.initial_inference_logits(obs)
        value_logits_k0 = value_logits

        policy_loss, illegal_loss = trainer._policy_terms(
            policy_logits, target_policies[:, 0], None, policy_loss_fn)
        value_loss = trainer._value_loss(value_logits, target_values[:, 0], net.value_support_size)
        reward_loss = torch.zeros(obs.shape[0], device=dev)
        consistency_loss = torch.zeros(obs.shape[0], device=dev)
        inverse_loss = torch.zeros(obs.shape[0], device=dev)
        moves_left_loss = torch.zeros(obs.shape[0], device=dev)
        if use_moves_left:
            moves_left_loss = trainer._moves_left_loss(
                net.predict_moves_left(hidden), target_moves_left[:, 0])

        value_loss_root_only = value_loss.detach().mean().item()

        for k in range(cfg.num_unroll_steps):
            hidden_in = hidden
            hidden, reward_logits, policy_logits, value_logits = \
                net.recurrent_inference_logits(hidden, actions[:, k])

            def make_hook(idx):
                def _h(grad):
                    hook_calls.append(idx)
                    return grad * 0.5
                return _h
            hidden.register_hook(make_hook(k))

            mask_k = target_obs_mask[:, k + 1]
            pl_k, im_k = trainer._policy_terms(
                policy_logits, target_policies[:, k + 1], None, policy_loss_fn)
            policy_loss = policy_loss + unroll_scale * pl_k * mask_k
            value_loss = value_loss + unroll_scale * trainer._value_loss(
                value_logits, target_values[:, k + 1], net.value_support_size) * mask_k
            reward_loss = reward_loss + unroll_scale * trainer._reward_loss(
                reward_logits, target_rewards[:, k + 1], net.reward_support_size) * mask_k

            if use_consistency:
                target_obs_k = target_observations[:, k + 1]
                if single_frame_consistency and num_planes is not None:
                    sf = torch.zeros_like(target_obs_k)
                    sf[:, :num_planes] = target_obs_k[:, :num_planes]
                    target_obs_k = sf
                from src.training.trainer import _negative_cosine_similarity
                dyn_proj = net.project(hidden, with_grad=True)
                with torch.no_grad():
                    target_hidden = net.representation(target_obs_k)
                    tgt_proj = net.project(target_hidden, with_grad=False)
                consistency_loss = consistency_loss + unroll_scale * (
                    _negative_cosine_similarity(dyn_proj, tgt_proj) * target_obs_mask[:, k + 1])

            if use_inverse:
                inv_logits = net.predict_inverse_action(hidden_in, hidden)
                inverse_loss = inverse_loss + unroll_scale * (
                    F.cross_entropy(inv_logits, actions[:, k], reduction="none") * mask_k)

            if use_moves_left:
                moves_left_loss = moves_left_loss + unroll_scale * (
                    trainer._moves_left_loss(
                        net.predict_moves_left(hidden), target_moves_left[:, k + 1]) * mask_k)

        per_sample_loss = outer_scale * (
            policy_loss
            + value_weight * value_loss
            + reward_loss
            + cfg.consistency_loss_weight * consistency_loss
            + inverse_weight * inverse_loss
            + moves_left_weight * moves_left_loss
        )
        total_loss = (is_weights_t * per_sample_loss).mean()

    trainer.optimizer.zero_grad()
    trainer.scaler.scale(total_loss).backward()
    scale_before = float(trainer.scaler.get_scale())
    trainer.scaler.unscale_(trainer.optimizer)

    fine = fine_grad_norms(net) if capture_grads else None
    groups = group_grad_norms(net)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))

    if n_hook_calls_box is not None:
        n_hook_calls_box["calls"] = list(hook_calls)

    loss_dict = {
        "total_loss": float(total_loss.item()),
        "policy_loss": float((outer_scale * policy_loss.mean()).item()),
        "value_loss": float((outer_scale * value_loss.mean()).item()),
        "value_loss_root_only": value_loss_root_only,
        "reward_loss": float((outer_scale * reward_loss.mean()).item()),
        "consistency_loss": float((outer_scale * cfg.consistency_loss_weight * consistency_loss.mean()).item()),
        "inverse_loss": float((outer_scale * inverse_weight * inverse_loss.mean()).item()),
        "moves_left_loss": float((outer_scale * moves_left_weight * moves_left_loss.mean()).item()) if use_moves_left else float("nan"),
        "value_weight_mean": float(value_weight.mean()) if isinstance(value_weight, torch.Tensor) else float(value_weight),
        "n_warm": int(is_warm.sum().item()),
        "n_sp": int((~is_warm).numel() - is_warm.sum().item()),
    }
    return loss_dict, fine, grad_norm, groups, value_logits_k0.detach(), target_values[:, 0].detach(), scale_before


def main():
    dev = "cpu"
    game = ChessGame()
    cfg = get_config("chess_small")
    cfg.device = dev
    torch.serialization.add_safe_globals([MuZeroConfig])

    ckpt_path = os.path.join(CKPT_DIR, "checkpoint_6000.pt")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev)

    print(f"=== checkpoint_6000 loaded | value_head_type={net.value_head_type} "
          f"value_support_size={net.value_support_size} ===")
    print(f"cfg: value_loss_weight_warmstart={cfg.value_loss_weight_warmstart} "
          f"value_loss_weight_selfplay={cfg.value_loss_weight_selfplay} "
          f"weight_decay={cfg.weight_decay} value_head_init_std={getattr(cfg,'value_head_init_std',None)} "
          f"use_amp={cfg.use_amp} K={cfg.num_unroll_steps}")

    # --- Static: value-head weight stats (is it dead / collapsed to constant?) ---
    print("\n=== VALUE HEAD WEIGHT STATS ===")
    for name, p in net.named_parameters():
        if name.startswith("prediction.value_head."):
            w = p.detach().float()
            print(f"  {name:55s} shape={tuple(p.shape)} "
                  f"norm={w.norm().item():.5f} std={w.std().item():.5f} "
                  f"absmax={w.abs().max().item():.5f}")

    # --- Build a trainer to get the real sample_batch + optimizer + scaler + buffer ---
    from src.training.trainer import MuZeroTrainer
    trainer = MuZeroTrainer(cfg, game, net, run_id="diag", device=dev,
                            log_dir="/tmp/diag_runs", checkpoints_dir="/tmp/diag_ckpts")
    # Restore optimizer state so weight_decay/Adam moments match the live run.
    trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    trainer.global_step = ckpt["step"]
    buf_path = os.path.join(CKPT_DIR, "checkpoint_6000.buf")
    trainer.replay_buffer.load(buf_path, game=game)
    print(f"\nLoaded buffer: {len(trainer.replay_buffer)} games")
    n_warm_games = sum(1 for g in trainer.replay_buffer.buffer if g.external_values)
    print(f"  warmstart(external_values) games: {n_warm_games} / {len(trainer.replay_buffer)}")

    # --- Adam value-head state (is the head actually being updated step-to-step?) ---
    print("\n=== ADAM STATE FOR VALUE HEAD (exp_avg / exp_avg_sq norms) ===")
    name_by_param = {p: n for n, p in net.named_parameters()}
    for p, st in trainer.optimizer.state.items():
        nm = name_by_param.get(p, "?")
        if nm.startswith("prediction.value_head."):
            ea = st.get("exp_avg"); eas = st.get("exp_avg_sq"); steps = st.get("step")
            print(f"  {nm:55s} step={float(steps) if steps is not None else '?'} "
                  f"|exp_avg|={ea.norm().item():.3e} |exp_avg_sq|={eas.norm().item():.3e}")

    # --- Run several real train steps; aggregate grad norms + hook calls ---
    torch.manual_seed(0)
    np.random.seed(0)
    N_STEPS = 5
    agg_groups = None
    hook_box = {}
    fine_first = None
    val_logits = None
    tgt_vals = None
    print(f"\n=== RUNNING {N_STEPS} REAL TRAIN STEPS (grad capture pre-clip, post-unscale) ===")
    for step_i in range(N_STEPS):
        ld, fine, gnorm, groups, vlog, tv, scale_before = run_one_train_step(
            trainer, capture_grads=(step_i == 0), n_hook_calls_box=hook_box)
        if step_i == 0:
            fine_first = fine
            val_logits = vlog
            tgt_vals = tv
        print(f"\n-- step {step_i} (global {trainer.global_step}) | total={ld['total_loss']:.4f} "
              f"value={ld['value_loss']:.4f} (root_only={ld['value_loss_root_only']:.4f}) "
              f"policy={ld['policy_loss']:.4f} inv={ld['inverse_loss']:.4f} "
              f"cons={ld['consistency_loss']:.4f} ml={ld['moves_left_loss']:.4f} | "
              f"grad_norm(total,pre-clip)={gnorm:.4f} | n_warm={ld['n_warm']} n_sp={ld['n_sp']} "
              f"value_w={ld['value_weight_mean']:.3f} | scaler_scale={scale_before:.1f}")
        for k, (gn, npar, nnone) in groups.items():
            print(f"      {k:26s} gradL2={gn:10.5f}   ({npar} params, {nnone} with grad=None)")
        if agg_groups is None:
            agg_groups = {k: [v[0]] for k, v in groups.items()}
        else:
            for k, v in groups.items():
                agg_groups[k].append(v[0])
        # Optimizer step so subsequent steps reflect real progression.
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()

    print("\n=== HOOK CALL AUDIT (0.5 hidden-grad hook) ===")
    calls = hook_box.get("calls", [])
    print(f"  hook fired {len(calls)} times in last step's backward; expected == K = {cfg.num_unroll_steps}")
    print(f"  per-unroll-index call counts: "
          f"{[(i, calls.count(i)) for i in range(cfg.num_unroll_steps)]}")
    print("  (each index should appear EXACTLY once; >1 => double-apply of 0.5)")

    print("\n=== GROUP GRAD NORM SUMMARY (mean over steps) ===")
    means = {k: float(np.mean(v)) for k, v in agg_groups.items()}
    total_sq = sum(m ** 2 for m in means.values())
    for k in sorted(means, key=lambda x: -means[x]):
        frac = (means[k] ** 2) / total_sq if total_sq > 0 else 0.0
        print(f"  {k:26s} mean_gradL2={means[k]:10.5f}   energy_frac={frac:6.2%}")

    # --- Fine grad: value head per-parameter, first step ---
    print("\n=== FINE GRAD: value-head params (step 0) ===")
    for name, gn in fine_first.items():
        if name.startswith("prediction.value_head."):
            print(f"  {name:55s} grad_norm={gn}")

    # --- Is the value-head OUTPUT layer (last linear) getting gradient? ---
    print("\n=== VALUE-HEAD LAST-LINEAR GRAD (the W that maps to WDL logits) ===")
    last_lin = None
    for name, p in net.named_parameters():
        if name.startswith("prediction.value_head.") and name.endswith(".weight") and p.dim() == 2:
            last_lin = (name, p)
    if last_lin is not None:
        nm, p = last_lin
        print(f"  last linear param: {nm} shape={tuple(p.shape)} "
              f"grad_norm={fine_first.get(nm)}")

    # --- Value calibration on THIS batch: does pred track target? ---
    print("\n=== VALUE HEAD OUTPUT vs TARGET (step-0 batch, root k=0) ===")
    from src.model.utils import wdl_to_scalar
    pred_v = wdl_to_scalar(val_logits.float(), draw_score=cfg.draw_score).cpu().numpy()
    if cfg.value_head_type == "wdl":
        tv = tgt_vals.float().cpu().numpy()  # (B,3)
        true_v = tv[:, 0] - tv[:, 2]
    else:
        true_v = tgt_vals.float().cpu().numpy()
    print(f"  pred_v:  mean={pred_v.mean():+.4f} std={pred_v.std():.4f} "
          f"min={pred_v.min():+.4f} max={pred_v.max():+.4f}")
    print(f"  true_v:  mean={true_v.mean():+.4f} std={true_v.std():.4f} "
          f"min={true_v.min():+.4f} max={true_v.max():+.4f}")
    if pred_v.std() > 1e-9 and true_v.std() > 1e-9:
        corr = float(np.corrcoef(pred_v, true_v)[0, 1])
        print(f"  corr(pred,true)={corr:+.4f}   MAE={np.abs(pred_v-true_v).mean():.4f}")
    pred_wdl = F.softmax(val_logits.float(), dim=-1).cpu().numpy()
    print(f"  pred P(W) mean={pred_wdl[:,0].mean():.3f}  P(D) mean={pred_wdl[:,1].mean():.3f} "
          f"P(L) mean={pred_wdl[:,2].mean():.3f}")
    print(f"  pred P(D) range [{pred_wdl[:,1].min():.3f},{pred_wdl[:,1].max():.3f}] "
          f"(near-constant ~1 => draw-saturated head)")


if __name__ == "__main__":
    main()
