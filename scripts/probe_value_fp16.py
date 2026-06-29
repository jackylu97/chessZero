"""fp16 numerical precision probe for the VALUE head + within-position resolution.

Two questions:
  (A) Does fp16 autocast (the trainer's training dtype) degrade the value head's
      WDL logits / scalar V vs fp32 enough to wipe out within-position resolution?
  (B) WITHIN-POSITION value resolution: take a root, expand to all legal moves via
      dynamics, read value-head V per child. What's the std of V ACROSS siblings
      (the 'sibling-Q' signal MCTS needs)? Compare to ACROSS-position std. Is the
      head's within-position blindness present in the RAW value head (pre-MCTS),
      and does fp16 make it worse?
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move
from src.model.utils import wdl_to_scalar
from src.training.replay_buffer import stack_with_history
from scripts.eval_checkpoint_health import build_network, random_playout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.serialization.add_safe_globals([MuZeroConfig])
    dev = args.device
    game = ChessGame()
    cfg = get_config("chess_small"); cfg.device = dev
    HF = cfg.history_frames
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev)
    net.eval()
    print(f"=== fp16/value-resolution probe @ step {ckpt.get('step')} ===")

    # Build several midgame roots.
    roots = []
    for i, n in enumerate([6, 14, 22, 30, 40, 50]):
        s, frames = random_playout(game, n, seed=200 + i)
        if s.done:
            continue
        obs = stack_with_history(game.to_tensor(s), frames, HF)
        roots.append((f"mid@{n}", s, s.board, obs, game.legal_actions(s)))

    # (A) fp16 vs fp32 on the value head at each root.
    print("\n--- (A) fp16 autocast vs fp32 value-head divergence (root V) ---")
    root_V_fp32 = []
    for name, s, board, obs, legal in roots:
        x = obs.unsqueeze(0).to(dev)
        with torch.no_grad():
            h32 = net.representation(x)
            _, vl32 = net.prediction(h32)
            v32 = wdl_to_scalar(vl32.float(), draw_score=cfg.draw_score).item()
            root_V_fp32.append(v32)
            # autocast fp16 (only meaningful on cuda; on cpu autocast is bf16-ish — still informative)
            dt = torch.float16 if dev.startswith("cuda") else torch.bfloat16
            with torch.autocast(device_type=dev.split(":")[0], dtype=dt):
                h16 = net.representation(x)
                _, vl16 = net.prediction(h16)
            v16 = wdl_to_scalar(vl16.float(), draw_score=cfg.draw_score).item()
            wdl32 = F.softmax(vl32.float(), -1)[0].tolist()
            wdl16 = F.softmax(vl16.float(), -1)[0].tolist()
        print(f"  {name:8s} V32={v32:+.5f} V16={v16:+.5f} dV={v16-v32:+.2e} "
              f"WDL32=({wdl32[0]:.3f},{wdl32[1]:.3f},{wdl32[2]:.3f}) "
              f"WDL16=({wdl16[0]:.3f},{wdl16[1]:.3f},{wdl16[2]:.3f})")
    print(f"  across-position root-V std (fp32): {np.std(root_V_fp32):.4f}")

    # (B) within-position sibling-V resolution via dynamics.
    print("\n--- (B) WITHIN-position sibling V std (dynamics -> value head) ---")
    all_within = []
    for name, s, board, obs, legal in roots:
        x = obs.unsqueeze(0).to(dev)
        with torch.no_grad():
            h = net.representation(x)
            probe = legal[:40]
            acts = torch.tensor(probe, device=dev)
            hn = h.expand(len(probe), *h.shape[1:]).contiguous()
            dyn, _ = net.dynamics(hn, acts)
            _, vl = net.prediction(dyn)
            sib_V = wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).cpu().numpy()
            # negamax: child value is from opponent POV; flip for parent-relative ranking
            sib_V_neg = -sib_V
        within_std = float(np.std(sib_V_neg))
        all_within.append(within_std)
        print(f"  {name:8s} n_moves={len(probe):3d} sibling-V(neg) "
              f"mean={sib_V_neg.mean():+.4f} std={within_std:.5f} "
              f"range=[{sib_V_neg.min():+.4f},{sib_V_neg.max():+.4f}]")
    print(f"\n  mean WITHIN-position sibling-V std: {np.mean(all_within):.5f}")
    print(f"  ACROSS-position root-V std:          {np.std(root_V_fp32):.5f}")
    ratio = np.std(root_V_fp32) / max(1e-9, np.mean(all_within))
    print(f"  across/within ratio: {ratio:.1f}x  "
          f"(>>1 => head ranks positions but not sibling moves = within-position blindness)")


if __name__ == "__main__":
    main()
