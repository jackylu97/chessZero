"""Opening-MCTS collapse probe.

For each checkpoint, run the REFERENCE numpy MCTS (src/mcts/mcts.py) from the
standard start position and report:
  - root policy PRIOR sharpness (entropy, top-move share) — what the net wants
    before search
  - root value V(s0)
  - per-child Q spread (sibling differentiation) — does search/value separate moves?
  - resulting VISIT distribution entropy + top-move share — what gets sampled

Tells us whether the empirically-observed opening collapse (ply-0 entropy 0.13b
at step 30k) is driven by a SHARP PRIOR (policy learned) or by VALUE BLINDNESS
(MCTS just follows the noised prior because Q can't separate moves).

CPU. Run:
  .venv/bin/python scripts/probe_opening_mcts_collapse.py \
     checkpoints/chess/2026_06_19_cold2_pc/checkpoint_{1000,6000,30000}.pt
"""
import argparse, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess

from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.mcts.mcts import MCTS
from src.training.replay_buffer import stack_with_history
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from eval_checkpoint_health import build_network


def entropy_bits(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def analyze(ckpt_path, game, cfg, device, n_sims, seeds=3):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    net = build_network(ckpt, game, cfg, device)

    state = game.reset()
    single = game.to_tensor(state)
    obs = stack_with_history(single, [], cfg.history_frames)  # zero-pad history (ply-0, faithful)
    legal = game.legal_actions(state)

    # --- raw net prior + value at root (no search) ---
    with torch.no_grad():
        hidden, policy_logits, value = net.initial_inference(obs.unsqueeze(0).to(device))
    pl = policy_logits.squeeze(0).cpu().numpy()
    mask = np.full_like(pl, -1e30)
    mask[legal] = pl[legal]
    prior = np.exp(mask - mask.max())
    prior = prior / prior.sum()
    prior_legal = prior[legal]
    # root value (scalar). WDL head returns scalar expectation.
    try:
        v0 = float(value.item())
    except Exception:
        v0 = float(np.asarray(value.detach().cpu()).ravel()[0])

    print(f"\n=== {os.path.basename(ckpt_path)}  step={ckpt.get('step')} ===")
    print(f"  #legal at start = {len(legal)}")
    print(f"  ROOT V(s0) = {v0:+.4f}")
    print(f"  PRIOR entropy = {entropy_bits(prior_legal):.2f} bits "
          f"(max {math.log2(len(legal)):.2f})  top-prior-share = {prior_legal.max():.1%}")
    # top prior moves
    order = np.argsort(prior)[::-1][:5]
    tops = []
    for a in order:
        mv = _action_to_move(int(a), chess.Board())
        tops.append((mv.uci() if mv else str(a), f"{prior[a]:.1%}"))
    print(f"  top-prior moves: {tops}")

    # --- run reference MCTS a few seeds, collect visit + Q stats ---
    visit_top_shares = []
    visit_entropies = []
    q_spreads = []
    chosen = []
    for s in range(seeds):
        np.random.seed(s); torch.manual_seed(s)
        mcts = MCTS(net, game, cfg, device=device)
        root = mcts.run(obs, legal, add_noise=True)
        # gather per-child visits and Q
        actions = root.child_actions
        visits = np.asarray(root.child_visits, dtype=float)
        vs = np.asarray(root.child_value_sums, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(visits > 0, vs / np.maximum(visits, 1), np.nan)
        tot = visits.sum()
        vfrac = visits / max(tot, 1)
        visit_top_shares.append(vfrac.max())
        visit_entropies.append(entropy_bits(vfrac))
        # Q spread over visited children (sibling differentiation)
        qv = q[visits > 0]
        if qv.size > 1:
            q_spreads.append(float(np.nanstd(qv)))
        bi = int(np.argmax(visits))
        mv = _action_to_move(int(actions[bi]), chess.Board())
        chosen.append(mv.uci() if mv else str(actions[bi]))

    print(f"  MCTS visit entropy = {np.mean(visit_entropies):.2f} bits  "
          f"top-visit-share = {np.mean(visit_top_shares):.1%}")
    print(f"  MCTS sibling-Q std (visited children) = {np.mean(q_spreads):.5f}  "
          f"(n_seeds with >1 visited: {len(q_spreads)})")
    print(f"  MCTS chosen move across {seeds} seeds: {chosen}")
    return dict(step=ckpt.get("step"), v0=v0,
                prior_ent=entropy_bits(prior_legal),
                prior_top=float(prior_legal.max()),
                visit_ent=float(np.mean(visit_entropies)),
                visit_top=float(np.mean(visit_top_shares)),
                q_std=float(np.mean(q_spreads)) if q_spreads else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    device = "cpu"
    game = ChessGame()
    cfg = get_config("chess_small")
    cfg.device = device
    if args.sims:
        cfg.num_simulations = args.sims
    print(f"num_simulations={cfg.num_simulations}  dirichlet_alpha={cfg.dirichlet_alpha} "
          f"eps={cfg.dirichlet_epsilon}  sample_k={cfg.sample_k}")

    for c in args.ckpts:
        analyze(c, game, cfg, device, cfg.num_simulations, seeds=args.seeds)


if __name__ == "__main__":
    main()
