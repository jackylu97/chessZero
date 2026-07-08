"""Within-position value RESOLUTION of the value head (the crux of the user's question).

For real buffer positions, expand the root latent across its LEGAL sibling actions
via dynamics, then run the VALUE HEAD on each sibling latent. Report:
  - dyn_value_std: std of value-head scalar V across siblings (within-position spread)
  - cross_action_cos: cosine sim of dynamics outputs across siblings (1=action-blind)
  - across_position_std: std of root V across DIFFERENT positions (the thing that's fine)

If within-position std << across-position std, the value head has across-position
calibration but no within-position resolution. We then localize WHERE it's lost:
dynamics collapse (cross_action_cos~1) vs value-head insensitivity (distinct latents
-> same V).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.model.utils import wdl_to_scalar
from scripts.eval_checkpoint_health import build_network

CKPT_DIR = "checkpoints/chess/2026_06_19_cold2_pc"


@torch.no_grad()
def main():
    dev = "cpu"
    game = ChessGame()
    cfg = get_config("chess_small"); cfg.device = dev
    torch.serialization.add_safe_globals([MuZeroConfig])
    hf = cfg.history_frames

    from src.training.replay_buffer import ReplayBuffer

    for step in [6000, 30000]:
        ckpt = torch.load(os.path.join(CKPT_DIR, f"checkpoint_{step}.pt"), map_location=dev, weights_only=True)
        net = build_network(ckpt, game, cfg, dev)
        rb = ReplayBuffer(cfg.replay_buffer_size)
        rb.load(os.path.join(CKPT_DIR, f"checkpoint_{step}.buf"), game=game)

        rng = np.random.default_rng(1)
        # Pick 40 real positions with >=4 legal moves.
        root_Vs = []
        within_std = []
        cross_cos = []
        head_within_std_given_distinct = []  # value-head spread on the SAME distinct latents
        picked = 0
        tries = 0
        while picked < 40 and tries < 4000:
            tries += 1
            g = rb.buffer[rng.integers(len(rb.buffer))]
            if len(g.policies) < 1:
                continue
            ply = int(rng.integers(len(g.policies)))
            if ply >= len(g.legal_actions_list):
                continue
            legal = g.legal_actions_list[ply]
            if not legal or len(legal) < 4:
                continue
            obs = g._stack_history(ply, hf).unsqueeze(0).to(dev)
            h = net.representation(obs)
            # root value
            _, vl = net.prediction(h)
            root_Vs.append(wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).item())
            # siblings
            probe = legal[:24]
            acts = torch.tensor(probe, device=dev)
            hn = h.expand(len(probe), *h.shape[1:]).contiguous()
            dyn, _ = net.dynamics(hn, acts)
            dv = F.normalize(dyn.reshape(len(probe), -1), dim=-1)
            cc = (dv @ dv.T)[~torch.eye(len(probe), dtype=bool)].mean().item()
            cross_cos.append(cc)
            _, svl = net.prediction(dyn)
            sib_V = wdl_to_scalar(svl.float(), draw_score=cfg.draw_score).cpu().numpy()
            within_std.append(float(sib_V.std()))
            picked += 1

        root_Vs = np.array(root_Vs); within_std = np.array(within_std); cross_cos = np.array(cross_cos)
        print(f"\n{'='*70}\nCHECKPOINT {step}  ({picked} positions, >=4 legal moves)")
        print(f"  ACROSS-position root V std      = {root_Vs.std():.4f}   "
              f"(range [{root_Vs.min():+.3f},{root_Vs.max():+.3f}])")
        print(f"  WITHIN-position sibling V std   = {within_std.mean():.4f}   "
              f"(mean over positions; max={within_std.max():.4f})")
        print(f"  dynamics cross-action cos       = {cross_cos.mean():.4f}   "
              f"(1.0 = action-blind dynamics)")
        ratio = root_Vs.std() / max(within_std.mean(), 1e-9)
        print(f"  across/within ratio             = {ratio:.1f}x   "
              f"(>>1 => value ranks positions but not sibling moves)")


if __name__ == "__main__":
    main()
