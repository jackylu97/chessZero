"""Gradient-flow probe: is the VALUE head actually learning?

Loads a checkpoint + a real .buf, runs an instrumented real train step, and
dumps per-parameter-group grad norms. Reports:
  - per-param-group grad norms (repr / dyn / value-head / policy-head / reward / aux)
  - per-LAYER value-head grad norms (is the OUTPUT layer dead?)
  - whether the value head passes grad BACK to the body (Jacobian zero?)
  - AMP/GradScaler state, weight_decay, value-head weight stats
  - the 0.5 hidden-state grad hook effect (single vs double application)
  - constant/zero-grad layers
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.training.trainer import MuZeroTrainer
from scripts.eval_checkpoint_health import build_network


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--buf", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    torch.serialization.add_safe_globals([MuZeroConfig])
    dev = args.device
    game = ChessGame()
    cfg = get_config("chess_small")
    cfg.device = dev
    cfg.batch_size = args.batch_size
    # Force AMP off on CPU; it's a no-op there but GradScaler under CPU misbehaves.
    if dev == "cpu":
        cfg.use_amp = False

    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev)
    print(f"=== checkpoint step {ckpt.get('step')} | device {dev} ===")
    print(f"value_head_type={cfg.value_head_type} value_head_init_std={getattr(cfg,'value_head_init_std',0.0)} "
          f"weight_decay={cfg.weight_decay} use_amp={cfg.use_amp}")
    print(f"value_loss_weight_warmstart={cfg.value_loss_weight_warmstart} "
          f"value_loss_weight_selfplay={cfg.value_loss_weight_selfplay} "
          f"num_unroll_steps={cfg.num_unroll_steps}")

    # ---- value-head weight stats (is it still near zero-init?) --------------
    print("\n--- VALUE HEAD WEIGHT STATS (from checkpoint) ---")
    sd = net.state_dict()
    for k in sd:
        if k.startswith("prediction.value_head"):
            t = sd[k].float()
            print(f"  {k:42s} shape={tuple(t.shape)} mean={t.mean():+.4e} "
                  f"std={t.std():.4e} absmax={t.abs().max():.4e}")

    # Build a trainer to reuse its exact _train_step machinery + grad groups.
    trainer = MuZeroTrainer(cfg, game, net, run_id="gradprobe", device=dev,
                            log_dir="/tmp/gradprobe_runs", checkpoints_dir="/tmp/gradprobe_ckpts")
    trainer.replay_buffer.load(args.buf, game=game)
    print(f"\nLoaded buffer: {len(trainer.replay_buffer)} games "
          f"(warmstart={sum(1 for g in trainer.replay_buffer.buffer if g.external_values)})")
    trainer.global_step = ckpt.get("step", 0)

    # Make the trainer log THIS step so _subnetwork_grad_norms fires.
    trainer.config.log_interval = 1

    # ---- Run the instrumented real train step ------------------------------
    # Register full per-parameter grad capture by monkeypatching nothing — we
    # just re-run _train_step but capture grads right after backward via a hook
    # on the optimizer? Simpler: replicate the inner of _train_step but keep
    # grads around. We instead wrap clip_grad_norm to snapshot grads first.
    captured = {}
    orig_clip = torch.nn.utils.clip_grad_norm_

    def clip_spy(params, *a, **k):
        # snapshot all grads BEFORE clip (these are post-unscale)
        for name, p in net.named_parameters():
            if p.grad is not None:
                captured[name] = (p.grad.detach().float().norm().item(),
                                  float(p.grad.detach().float().abs().max()),
                                  p.numel())
            else:
                captured[name] = (None, None, p.numel())
        return orig_clip(params, *a, **k)

    torch.nn.utils.clip_grad_norm_ = clip_spy
    try:
        info = trainer._train_step()
    finally:
        torch.nn.utils.clip_grad_norm_ = orig_clip

    print("\n--- TRAIN STEP LOSS INFO ---")
    for k in ["total_loss", "policy_loss", "value_loss", "reward_loss",
              "consistency_loss", "inverse_loss", "moves_left_loss",
              "grad_norm", "amp_scale", "value_mae", "value_target_std",
              "value_loss_weight", "batch_warmstart_frac"]:
        if k in info:
            print(f"  {k:24s} {info[k]}")

    # ---- Per-subnetwork grad norms (the trainer's own groups) --------------
    print("\n--- PER-SUBNETWORK GRAD NORMS (trainer._GRAD_GROUPS) ---")
    for k, v in sorted(info.items()):
        if k.startswith("gnorm_"):
            print(f"  {k[len('gnorm_'):]:24s} {v:.6e}")

    # ---- Per-layer value-head grad norms -----------------------------------
    print("\n--- PER-LAYER VALUE-HEAD GRAD NORMS (is the OUTPUT layer dead?) ---")
    for name, (gn, gmax, ne) in captured.items():
        if name.startswith("prediction.value_head"):
            s = f"{gn:.6e}" if gn is not None else "NO GRAD"
            mx = f"{gmax:.3e}" if gmax is not None else "-"
            print(f"  {name:42s} gnorm={s:14s} gmax={mx:10s} n={ne}")

    # ---- Per-group totals incl. moves_left (NOT in trainer groups) ---------
    print("\n--- FULL PER-PREFIX GRAD NORM ROLLUP (manual) ---")
    groups = {
        "representation": "representation.",
        "dyn.action_embed": "dynamics.action_embedding.",
        "dyn.body": ("dynamics.conv_in.", "dynamics.bn_in.", "dynamics.blocks."),
        "dyn.reward_head": "dynamics.reward_head.",
        "pred.policy_head": "prediction.policy_head.",
        "pred.value_head": "prediction.value_head.",
        "ssl.projection": "projection.",
        "ssl.prediction_head": "prediction_head.",
        "inverse_head": "inverse_dynamics_head.",
        "moves_left_head": "moves_left_head.",
    }
    for gname, pref in groups.items():
        prefs = pref if isinstance(pref, tuple) else (pref,)
        acc = 0.0; nparam = 0; ng = 0; nz = 0
        for name, (gn, gmax, ne) in captured.items():
            if name.startswith(prefs):
                nparam += 1
                if gn is None:
                    ng += 1
                else:
                    acc += gn ** 2
                    if gn == 0.0:
                        nz += 1
        print(f"  {gname:22s} L2={acc**0.5:.6e}  params={nparam} nograd={ng} zerograd={nz}")

    # ---- Constant / zero-grad layers anywhere ------------------------------
    print("\n--- ZERO-OR-MISSING GRAD PARAMETERS (anywhere) ---")
    any_zero = False
    for name, (gn, gmax, ne) in captured.items():
        if gn is None or gn == 0.0:
            any_zero = True
            print(f"  {name:50s} grad={'NONE' if gn is None else '0.0'}")
    if not any_zero:
        print("  (none — every parameter received nonzero grad)")

    # ---- Does the value head pass grad BACK to the body? -------------------
    # Probe directly: zero grads, forward representation -> value_head only,
    # backprop a value loss, check grad on representation params.
    print("\n--- VALUE-HEAD -> BODY JACOBIAN PROBE ---")
    net.zero_grad(set_to_none=True)
    net.train()
    batch, gi, isw = trainer.replay_buffer.sample_batch(
        args.batch_size, cfg.num_unroll_steps, cfg.td_steps, cfg.discount,
        game.action_space_size, alpha=cfg.per_alpha, beta=1.0,
        value_head_type=cfg.value_head_type, history_frames=cfg.history_frames,
        eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
        q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
        selfplay_q_ratio=cfg.selfplay_q_ratio,
        repetition_penalty=cfg.repetition_penalty,
        repetition_penalty_window=int(getattr(cfg, "repetition_penalty_window", 0)),
        repetition_penalty_decay=float(getattr(cfg, "repetition_penalty_decay", 0.0)),
    )
    obs = batch["observations"].to(dev)
    tv = batch["target_values"].to(dev)
    hidden = net.representation(obs)
    _, value_logits = net.prediction(hidden)
    # WDL CE at root
    vloss = -(tv[:, 0] * F.log_softmax(value_logits, dim=1)).sum(dim=1).mean()
    vloss.backward()
    repr_gn = sum((p.grad.float().norm().item() ** 2)
                  for n, p in net.named_parameters()
                  if n.startswith("representation.") and p.grad is not None) ** 0.5
    vh_gn = sum((p.grad.float().norm().item() ** 2)
                for n, p in net.named_parameters()
                if n.startswith("prediction.value_head") and p.grad is not None) ** 0.5
    print(f"  value-only loss = {vloss.item():.6f}")
    print(f"  grad reaching representation body from value loss: {repr_gn:.6e}")
    print(f"  grad on value head itself:                          {vh_gn:.6e}")
    print(f"  -> body {'RECEIVES' if repr_gn > 1e-9 else 'BLOCKED FROM'} value-head gradient")


if __name__ == "__main__":
    main()
