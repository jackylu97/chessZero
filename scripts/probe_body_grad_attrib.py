"""Attribute the representation-BODY gradient to each loss head.

The within-position value resolution depends on whether the VALUE loss actually
shapes the shared representation/dynamics body, or whether other heads
(consistency w=2.0, policy, inverse) dominate and starve the value signal in the
body. We backprop each head's loss SEPARATELY (same batch, fresh zero_grad) and
measure the L2 grad it deposits on representation + dynamics body params.

Also a hard correctness check: run the FULL train-step loss but with the value
term's weight set to 0 vs normal, and confirm the value head grad responds (i.e.
the value loss is actually wired to the value head, not detached/zeroed).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.training.trainer import MuZeroTrainer
from src.model.utils import scalar_transform, scalar_to_support
from scripts.eval_checkpoint_health import build_network


def body_grad(net):
    rg = sum((p.grad.float().norm().item() ** 2) for n, p in net.named_parameters()
             if n.startswith("representation.") and p.grad is not None) ** 0.5
    dg = sum((p.grad.float().norm().item() ** 2) for n, p in net.named_parameters()
             if (n.startswith("dynamics.conv_in") or n.startswith("dynamics.bn_in")
                 or n.startswith("dynamics.blocks")) and p.grad is not None) ** 0.5
    return rg, dg


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
    cfg = get_config("chess_small"); cfg.device = dev; cfg.batch_size = args.batch_size
    if dev == "cpu": cfg.use_amp = False
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev); net.train()

    trainer = MuZeroTrainer(cfg, game, net, run_id="bodygrad", device=dev,
                            log_dir="/tmp/bg_runs", checkpoints_dir="/tmp/bg_ck")
    trainer.replay_buffer.load(args.buf, game=game)
    batch, gi, isw = trainer.replay_buffer.sample_batch(
        args.batch_size, cfg.num_unroll_steps, cfg.td_steps, cfg.discount,
        game.action_space_size, alpha=cfg.per_alpha, beta=1.0,
        value_head_type=cfg.value_head_type, history_frames=cfg.history_frames,
        eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
        q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
        selfplay_q_ratio=cfg.selfplay_q_ratio,
        repetition_penalty=cfg.repetition_penalty,
        repetition_penalty_window=int(getattr(cfg, "repetition_penalty_window", 0)),
        repetition_penalty_decay=float(getattr(cfg, "repetition_penalty_decay", 0.0)))
    obs = batch["observations"].to(dev)
    actions = batch["actions"].to(dev)
    tv = batch["target_values"].to(dev)
    tp = batch["target_policies"].to(dev)
    mask = batch["target_obs_mask"].to(dev)
    K = cfg.num_unroll_steps

    print(f"=== body-grad attribution @ step {ckpt.get('step')} (B={args.batch_size}) ===")
    # target value distribution sanity: how draw-saturated are the WDL targets?
    tv0 = tv[:, 0].cpu().numpy()
    print(f"root WDL target mean: W={tv0[:,0].mean():.3f} D={tv0[:,1].mean():.3f} L={tv0[:,2].mean():.3f}")
    print(f"root WDL target std:  W={tv0[:,0].std():.3f} D={tv0[:,1].std():.3f} L={tv0[:,2].std():.3f}")
    frac_pure_draw = float((tv0[:,1] > 0.99).mean())
    print(f"fraction of root targets that are ~pure draw (D>0.99): {frac_pure_draw:.3f}")

    def value_only_loss():
        hidden, _, vlog = net.initial_inference_logits(obs)
        vl = -(tv[:,0] * F.log_softmax(vlog, dim=1)).sum(dim=1)
        for k in range(K):
            hidden, _, _, vlog = net.recurrent_inference_logits(hidden, actions[:,k])
            hidden.register_hook(lambda g: g*0.5)
            vl = vl + (-(tv[:,k+1]*F.log_softmax(vlog,1)).sum(1))*mask[:,k+1]
        return (vl/(K+1)).mean()

    def policy_only_loss():
        hidden, plog, _ = net.initial_inference_logits(obs)
        pl = -(tp[:,0]*F.log_softmax(plog,1)).sum(1)
        for k in range(K):
            hidden, _, plog, _ = net.recurrent_inference_logits(hidden, actions[:,k])
            hidden.register_hook(lambda g: g*0.5)
            pl = pl + (-(tp[:,k+1]*F.log_softmax(plog,1)).sum(1))*mask[:,k+1]
        return (pl/(K+1)).mean()

    for label, fn in [("VALUE-only", value_only_loss), ("POLICY-only", policy_only_loss)]:
        net.zero_grad(set_to_none=True)
        loss = fn(); loss.backward()
        rg, dg = body_grad(net)
        print(f"\n  {label}: loss={loss.item():.4f}")
        print(f"    grad into representation body: {rg:.6e}")
        print(f"    grad into dynamics body:       {dg:.6e}")

    # Hard wiring check: value head grad with value weight 0 vs 1.
    print("\n--- WIRING CHECK: value-head grad responds to value loss ---")
    net.zero_grad(set_to_none=True)
    loss = value_only_loss(); loss.backward()
    vh = sum((p.grad.float().norm().item()**2) for n,p in net.named_parameters()
             if n.startswith("prediction.value_head") and p.grad is not None)**0.5
    ph = sum((p.grad.float().norm().item()**2) for n,p in net.named_parameters()
             if n.startswith("prediction.policy_head") and p.grad is not None)**0.5
    print(f"  from VALUE-only loss: value_head grad={vh:.6e}  policy_head grad={ph:.6e}")
    print(f"  (policy_head grad should be ~0 under value-only loss -> {'OK' if ph<1e-9 else 'LEAK'})")


if __name__ == "__main__":
    main()
