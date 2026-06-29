"""Prosecutor verification: within- vs across-position value spread on real buffer
positions, using the reference numpy MCTS. Determines fundamental-vs-bug."""
import argparse, pickle, sys, math
import numpy as np
import torch

sys.path.insert(0, ".")
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.training.replay_buffer import GameHistory
from src.mcts.mcts import MCTS, MinMaxStats
from scripts.eval_checkpoint_health import build_network


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--buf", required=True)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--n-pos", type=int, default=40)
    ap.add_argument("--min-ply", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    torch.serialization.add_safe_globals([MuZeroConfig])
    cfg = get_config("chess_small"); cfg.device = "cpu"
    game = ChessGame()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    net = build_network(ckpt, game, cfg, "cpu")
    print(f"step={ckpt['step']} value_head={cfg.value_head_type} draw_score={cfg.draw_score} sims={args.sims}")

    with open(args.buf, "rb") as f:
        header = pickle.load(f)
        recs = [pickle.load(f)[0] for _ in range(header["n_records"])]
    sp = [d for d in recs if len(d.get("external_values", [])) == 0]
    print(f"buf: {len(recs)} games, {len(sp)} self-play")

    # Pick games and plies. Reconstruct full-history observation via from_compact_dict.
    order = list(range(len(sp))); rng.shuffle(order)
    root_vals = []          # across-position: net root scalar V per sampled position
    head_wdl = []           # raw head WDL per position
    sib_q_stds = []         # within-position: std of child Q (mover POV)
    sib_margins = []        # best - median child Q
    collected = 0
    HF = cfg.history_frames
    for gi in order:
        if collected >= args.n_pos:
            break
        d = sp[gi]
        try:
            gh = GameHistory.from_compact_dict(d, game)
        except Exception:
            continue
        n = len(gh.actions)
        if n <= args.min_ply + 2:
            continue
        ply = int(rng.integers(args.min_ply, n - 1))
        # Build stacked observation with history (newest first), like self-play.
        frames = []
        for k in range(HF):
            idx = ply - k
            if idx >= 0:
                frames.append(gh.observations[idx])
            else:
                frames.append(torch.zeros_like(gh.observations[0]))
        obs = torch.cat(frames, dim=0).float()  # [HF*C, H, W] (MCTS.run unsqueezes)
        legal = gh.legal_actions_list[ply] if ply < len(gh.legal_actions_list) else None
        if not legal:
            continue
        # head WDL from initial_inference_logits (returns raw value logits)
        with torch.no_grad():
            _h, _p, vlog = net.initial_inference_logits(obs.unsqueeze(0))
        if vlog.shape[-1] == 3:
            wdl = torch.softmax(vlog, dim=-1)[0].tolist()
        else:
            wdl = None
        # run MCTS
        mcts = MCTS(net, game, cfg)
        root = mcts.run(obs, legal, add_noise=False)
        root_vals.append(float(root.value))
        if wdl is not None:
            head_wdl.append(wdl)
        # within-position child Q (mover POV): reward - discount*child.value
        disc = cfg.discount
        cv = np.asarray(root.child_visits, dtype=np.float64)
        cvs = np.asarray(root.child_value_sums, dtype=np.float64)
        cr = np.asarray(root.child_rewards, dtype=np.float64)
        mask = cv > 0
        child_value = np.zeros_like(cvs)
        child_value[mask] = cvs[mask] / cv[mask]
        q_all = cr - disc * child_value
        qs = q_all[mask].tolist()
        if len(qs) >= 3:
            qs = np.array(qs)
            sib_q_stds.append(float(qs.std()))
            sib_margins.append(float(qs.max() - np.median(qs)))
        collected += 1

    rv = np.array(root_vals)
    print(f"\n=== ACROSS-position spread (n={len(rv)}) ===")
    print(f"  root.value  mean={rv.mean():+.3f} std={rv.std():.3f} min={rv.min():+.3f} max={rv.max():+.3f}")
    if head_wdl:
        hw = np.array(head_wdl)
        print(f"  head WDL    mean=[{hw[:,0].mean():.3f},{hw[:,1].mean():.3f},{hw[:,2].mean():.3f}]  P(D)>0.8 frac={np.mean(hw[:,1]>0.8):.2f}")
        scal = hw[:,0]-hw[:,2]+cfg.draw_score*hw[:,1]
        print(f"  head scalar mean={scal.mean():+.3f} std={scal.std():.3f}")
    sq = np.array(sib_q_stds)
    sm = np.array(sib_margins)
    print(f"\n=== WITHIN-position sibling-Q spread (n={len(sq)}) ===")
    print(f"  sibling-Q std   mean={sq.mean():.4f} median={np.median(sq):.4f} p90={np.percentile(sq,90):.4f}")
    print(f"  best-median margin mean={sm.mean():.4f} median={np.median(sm):.4f}")
    print(f"  frac positions sibling-Q std < 0.02: {np.mean(sq<0.02):.2f}")
    if len(sq):
        print(f"\n  RATIO within/across = {sq.mean()/max(rv.std(),1e-9):.3f}")


if __name__ == "__main__":
    main()
