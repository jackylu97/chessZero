"""Does DEEPER search ERASE the one-step value signal?

For a few real positions, run reference MCTS at increasing sim budgets and track,
for the root's children:
  - sibling std of child.value (search, child POV)  -- the signal MCTS uses
  - sibling std of the NETWORK's DIRECT value at each child latent (1-step dynNet_V)
  - spearman(child.value_search, dynNet_V) -- does backed-up value still track the
    1-step value, or does deep-subtree draw-averaging wash it out?

If sibling-std(child.value_search) SHRINKS as sims grow while dynNet_V std stays,
then deep search is averaging the thin 1-step signal into draw -> more sims HURT.
This is the mechanism behind 'search amplifies miscalibration / more sims worse'.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.training.replay_buffer import GameHistory, stack_with_history
from src.mcts.mcts import MCTS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_checkpoint_health import build_network
from probe_mcts_internal_consistency import spearman

def load(buf, game, mx):
    import pickle
    gs = []
    with open(buf, "rb") as f:
        first = pickle.load(f); ver = first["version"]; n = first["n_records"]
        for _ in range(n):
            rec, _ = pickle.load(f)
            if ver == 3: rec = GameHistory.from_compact_dict(rec, game)
            gs.append(rec)
            if len(gs) >= mx: break
    return gs

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-dir", default="checkpoints/chess/2026_06_19_cold2_pc")
ap.add_argument("--device", default="cpu")
ap.add_argument("--n-positions", type=int, default=5)
ap.add_argument("--seed", type=int, default=3)
args = ap.parse_args()

dev = args.device
cfg = get_config("chess_small"); cfg.device = dev
HF = cfg.history_frames
game = ChessGame(); torch.serialization.add_safe_globals([MuZeroConfig])
ckpt = torch.load(os.path.join(args.ckpt_dir, "checkpoint_30000.pt"), map_location=dev, weights_only=True)
net = build_network(ckpt, game, cfg, dev)

gs = [g for g in load(os.path.join(args.ckpt_dir, "checkpoint_30000.buf"), game, 200)
      if not g.external_values and len(g.actions) >= 12]
rng = np.random.default_rng(args.seed); rng.shuffle(gs)
samples = []
for g in gs:
    ply = int(rng.integers(2, len(g.actions) - 1))
    if ply < len(g.policies): samples.append((g, ply))
    if len(samples) >= args.n_positions: break

SIM_GRID = [25, 100, 400]
print(f"{'ply':>4s} {'sims':>5s} {'rootV':>7s} {'std(childV_srch)':>16s} "
      f"{'std(dynNetV)':>13s} {'sp(srch,dyn)':>12s} {'topVisFr':>8s}", flush=True)
print("-"*80, flush=True)
for (g, ply) in samples:
    cur = g.observations[ply]; prior = g.observations[:ply]
    obs = stack_with_history(cur, prior, HF)
    legal = g.legal_actions_list[ply]
    # one-step dynNet_V per legal action (independent of sim count)
    with torch.no_grad():
        h, _, _ = net.initial_inference(obs.unsqueeze(0).to(dev))
        acts = torch.tensor(legal, device=dev)
        hn = h.expand(len(legal), *h.shape[1:]).contiguous()
        _, _, _, vch = net.recurrent_inference(hn, acts)
        dyn_v = vch.view(-1).cpu().numpy()  # child POV one-step value
    for sims in SIM_GRID:
        cfg.num_simulations = sims
        mcts = MCTS(net, game, cfg, dev)
        root = mcts.run(obs, legal, add_noise=False)
        visits = root.child_visits
        vmask = visits > 0
        # map root child actions to their dyn_v
        act_to_dyn = {int(a): dyn_v[i] for i, a in enumerate(legal)}
        child_v = np.array([(root.children[i].value if root.children[i] is not None else 0.0)
                            for i in range(len(root.child_actions))])
        dyn_aligned = np.array([act_to_dyn.get(int(a), np.nan) for a in root.child_actions])
        sv = float(child_v[vmask].std()) if vmask.sum() >= 2 else float("nan")
        sd = float(dyn_aligned[vmask].std()) if vmask.sum() >= 2 else float("nan")
        spc = spearman(child_v[vmask], dyn_aligned[vmask]) if vmask.sum() >= 2 else float("nan")
        topfr = float(visits.max()/visits.sum()) if visits.sum() > 0 else float("nan")
        print(f"{ply:4d} {sims:5d} {root.value:+7.3f} {sv:16.5f} {sd:13.5f} "
              f"{spc:12.3f} {topfr:8.3f}", flush=True)
    print(flush=True)
