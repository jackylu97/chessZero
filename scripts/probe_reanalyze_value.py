"""Quantify reanalyze's benefit: how much does re-running MCTS with the CURRENT
net actually CHANGE the stored targets? If the delta is small (targets already
fresh — small buffer cycles fast, net improves slowly), reanalyze's ~46%-of-MCTS
compute is mostly wasted.

Mirrors trainer._reanalyze: sample self-play positions, rebuild the T-frame obs,
run BatchedMCTS (200 sims, no noise), compare fresh root.value / policy to the
STORED root_values[pos] / policies[pos].

Run: .venv/bin/python scripts/probe_reanalyze_value.py \
        --checkpoint <ckpt.pt> --buf <ckpt.buf> --game chess_small --positions 200
"""
import argparse, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.mcts.mcts import BatchedMCTS, select_action
from src.training.replay_buffer import ReplayBuffer, _densify_policy
from scripts.eval_checkpoint_health import build_network


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--buf", required=True)
    ap.add_argument("--game", default="chess_small")
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    dev = args.device
    game = ChessGame()
    cfg = get_config(args.game); cfg.device = dev
    n_frames = getattr(cfg, "history_frames", 1)
    A = game.action_space_size

    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev); net.eval()
    print(f"net step {ckpt.get('step','?')} | buffer {args.buf}")

    rb = ReplayBuffer(max_size=10_000_000); rb.load(args.buf, game=game)
    sp = [g for g in rb.buffer if not getattr(g, "external_values", [])]
    print(f"  {len(sp)} self-play games; sampling {args.positions} positions")

    # sample positions
    items = []
    tries = 0
    while len(items) < args.positions and tries < args.positions * 40:
        tries += 1
        g = random.choice(sp)
        if not g.policies or pos_cap(g) < 1:
            continue
        pos = random.randint(0, pos_cap(g) - 1)
        try:
            obs = g._stack_history(pos, n_frames)
            legal = g.legal_actions_list[pos]
        except Exception:
            continue
        items.append((obs, legal, g, pos))

    mcts = BatchedMCTS(net, game, cfg, dev)
    chunk = 64
    dv, sv, agree, kl_list, topp_stored, topp_fresh = [], [], 0, [], [], []
    n = 0
    for start in range(0, len(items), chunk):
        batch = items[start:start + chunk]
        roots = mcts.run_batch([b[0] for b in batch], [b[1] for b in batch], add_noise=False)
        for (_, _, g, pos), root in zip(batch, roots):
            if float(root.child_visits.sum()) <= 0:
                continue
            fresh_v = float(root.value)
            stored_v = float(g.root_values[pos])
            _, fresh_probs = select_action(root, temperature=1.0)
            fresh_pol = np.zeros(A, dtype=np.float32); fresh_pol[:len(fresh_probs)] = fresh_probs
            stored_pol = _densify_policy(g.policies[pos], A)
            stored_pol = np.asarray(stored_pol, dtype=np.float32)
            if stored_pol.sum() > 0:
                stored_pol = stored_pol / stored_pol.sum()
            dv.append(fresh_v - stored_v); sv.append(stored_v)
            # top-move agreement
            if int(np.argmax(fresh_pol)) == int(np.argmax(stored_pol)):
                agree += 1
            # KL(stored || fresh) over union support
            eps = 1e-8
            m = (stored_pol > 0)
            kl = float(np.sum(stored_pol[m] * np.log((stored_pol[m] + eps) / (fresh_pol[m] + eps))))
            kl_list.append(kl)
            topp_stored.append(float(stored_pol.max())); topp_fresh.append(float(fresh_pol.max()))
            n += 1

    dv = np.array(dv); sv = np.array(sv)
    print(f"\n=== reanalyze DELTA over {n} positions (fresh net vs stored targets) ===")
    print(f"  VALUE: corr(stored, fresh) = {np.corrcoef(sv, sv+dv)[0,1]:+.3f}")
    print(f"         mean |Δvalue| = {np.mean(np.abs(dv)):.3f}   (value range [-1,1]; stored std={sv.std():.3f})")
    print(f"         mean Δvalue (bias) = {np.mean(dv):+.3f}")
    print(f"  POLICY: top-move agreement (fresh vs stored) = {agree/max(1,n):.1%}")
    print(f"          mean KL(stored||fresh) = {np.mean(kl_list):.3f} nats")
    print(f"          mean top-move prob: stored={np.mean(topp_stored):.3f} -> fresh={np.mean(topp_fresh):.3f}")
    print("\n  (small Δ / high corr / high agreement => reanalyze barely changes targets => low value)")


def pos_cap(g):
    return min(len(g.policies), len(getattr(g, "legal_actions_list", [])), len(g.root_values))


if __name__ == "__main__":
    main()
