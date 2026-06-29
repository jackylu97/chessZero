"""Probe what the VALUE and POLICY heads learn OVER training.

Compares checkpoint_{1000,6000,30000} on a SHARED, fixed set of positions
sampled from the step-6000 buffer (self-play, mid-collapse) and graded by
Stockfish. For each checkpoint reports:

  VALUE:
   - WDL output distribution (mean P(W)/P(D)/P(L)) and scalar V spread
   - calibration vs Stockfish eval (corr, slope)
   - per-bucket (SF win/draw/loss) mean V and draw-prob saturation
   - "is it stuck at draw from the start, or does it degrade?"

  POLICY:
   - entropy over LEGAL moves (vs uniform-legal entropy)
   - top-1 / top-3 move probability mass
   - mode-collapse check: does the argmax move repeat across distinct positions?

Distinguishes "never learned" vs "learned then collapsed".

Run: .venv/bin/python scripts/probe_head_evolution.py
"""
import os, sys, pickle, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
import chess, chess.engine

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move, _move_to_action
from src.model.utils import wdl_to_scalar
from src.training.replay_buffer import GameHistory, stack_with_history

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from eval_checkpoint_health import build_network

CKDIR = "checkpoints/chess/2026_06_19_cold2_pc"
SF = "/usr/games/stockfish"
DEV = "cpu"


def load_games(buf_path, game, limit=None):
    """Load full GameHistory objects (with observations replayed)."""
    games = []
    with open(buf_path, "rb") as f:
        header = pickle.load(f)
        n = header["n_records"]
        for _ in range(n):
            d, _prio = pickle.load(f)
            try:
                gh = GameHistory.from_compact_dict(d, game)
            except Exception:
                continue
            games.append(gh)
            if limit and len(games) >= limit:
                break
    return games


def sample_positions(games, n_pos, hf, min_ply=8, seed=0):
    """Sample (obs, board, legal_actions, ply, policy_target, root_value, game_outcome)
    from self-play games. Reconstruct boards by replaying actions."""
    rng = random.Random(seed)
    sp = [g for g in games if not g.external_values]
    out = []
    tries = 0
    while len(out) < n_pos and tries < n_pos * 50:
        tries += 1
        g = rng.choice(sp)
        if len(g.actions) <= min_ply + 1:
            continue
        ply = rng.randint(min_ply, len(g.actions) - 1)
        # reconstruct board at ply
        board = chess.Board()
        ok = True
        for a in g.actions[:ply]:
            mv = _action_to_move(int(a), board)
            if mv is None or mv not in board.legal_moves:
                ok = False
                break
            board.push(mv)
        if not ok or board.is_game_over():
            continue
        obs = g.observations[ply] if ply < len(g.observations) else None
        if obs is None:
            continue
        # stacked-history obs via the stored GameHistory helper
        stacked = g._stack_history(ply, hf)
        legal = [_move_to_action(m, board.turn) for m in board.legal_moves]
        pol = g.policies[ply] if ply < len(g.policies) else None
        rv = g.root_values[ply] if ply < len(g.root_values) else None
        out.append(dict(board=board.copy(), obs=stacked, legal=legal, ply=ply,
                        policy=pol, root_value=rv, outcome=float(g.game_outcome),
                        actions_after=g.actions[ply:ply+1]))
    return out


def sf_eval(engine, board, depth=12):
    """STM-relative eval in [-1,1]-ish via tanh of pawns; also raw cp."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].pov(board.turn)
    if score.is_mate():
        m = score.mate()
        return (1.0 if m > 0 else -1.0), (10000 if m > 0 else -10000)
    cp = score.score()
    return float(np.tanh(cp / 400.0)), cp


@torch.no_grad()
def eval_heads(net, cfg, obs_batch):
    xs = torch.stack(obs_batch).to(DEV)
    h = net.representation(xs)
    pl, vl = net.prediction(h)
    wdl = F.softmax(vl.float(), -1).cpu().numpy()        # (B,3) [W,D,L]
    V = wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).cpu().numpy()
    return pl.float().cpu().numpy(), wdl, V


def policy_stats(logits_row, legal):
    """Entropy over legal moves, top-1/top-3 mass, argmax action."""
    legal = list(set(int(a) for a in legal))
    z = logits_row[legal]
    p = np.exp(z - z.max()); p = p / p.sum()
    ent = -np.sum(p * np.log(p + 1e-12))
    uniform_ent = np.log(len(legal))
    order = np.argsort(-p)
    top1 = float(p[order[0]])
    top3 = float(p[order[:3]].sum())
    argmax_action = int(legal[order[0]])
    # mass on illegal moves (full softmax over all actions)
    full = np.exp(logits_row - logits_row.max()); full = full / full.sum()
    legal_mask = np.zeros_like(full); legal_mask[legal] = 1.0
    illegal_mass = float((full * (1 - legal_mask)).sum())
    return ent, uniform_ent, top1, top3, argmax_action, illegal_mass


def main():
    torch.serialization.add_safe_globals([MuZeroConfig])
    game = ChessGame()
    cfg = get_config("chess_small"); cfg.device = DEV
    hf = cfg.history_frames
    print(f"config: history_frames={hf} value_head_type={cfg.value_head_type} "
          f"draw_score={cfg.draw_score} num_planes={game.num_planes}")

    # Fixed shared position set from the step-6000 buffer (mid-collapse).
    print("\nLoading step-6000 buffer for shared position set...")
    g6 = load_games(f"{CKDIR}/checkpoint_6000.buf", game, limit=400)
    pos = sample_positions(g6, n_pos=300, hf=hf, seed=42)
    print(f"Sampled {len(pos)} self-play positions (ply>=8)")

    # Grade with Stockfish once.
    print("Grading with Stockfish (depth 12)...")
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    for p in pos:
        v, cp = sf_eval(eng, p["board"], depth=12)
        p["sf"] = v; p["sf_cp"] = cp
    eng.quit()
    sf_vals = np.array([p["sf"] for p in pos])
    print(f"SF eval dist: mean={sf_vals.mean():+.3f} std={sf_vals.std():.3f} "
          f"win(>+0.3)={np.mean(sf_vals>0.3):.0%} draw(|.|<0.3)={np.mean(np.abs(sf_vals)<0.3):.0%} "
          f"loss(<-0.3)={np.mean(sf_vals<-0.3):.0%}")

    obs_batch = [p["obs"] for p in pos]

    for step in [1000, 6000, 30000]:
        ckpt = torch.load(f"{CKDIR}/checkpoint_{step}.pt", map_location=DEV, weights_only=True)
        net = build_network(ckpt, game, cfg, DEV)
        pl, wdl, V = eval_heads(net, cfg, obs_batch)

        print(f"\n{'='*72}\nCHECKPOINT {step}\n{'='*72}")
        # ---- VALUE ----
        print(f"[VALUE] WDL mean: W={wdl[:,0].mean():.3f} D={wdl[:,1].mean():.3f} "
              f"L={wdl[:,2].mean():.3f}")
        print(f"[VALUE] scalar V: mean={V.mean():+.3f} std={V.std():.3f} "
              f"range=[{V.min():+.3f},{V.max():+.3f}]")
        print(f"[VALUE] P(D) per-pos: mean={wdl[:,1].mean():.3f} std={wdl[:,1].std():.3f} "
              f"max={wdl[:,1].max():.3f} | frac P(D)>0.9: {np.mean(wdl[:,1]>0.9):.0%}")
        if V.std() > 1e-6 and sf_vals.std() > 1e-6:
            corr = np.corrcoef(V, sf_vals)[0, 1]
            slope = np.polyfit(sf_vals, V, 1)[0]
        else:
            corr = slope = float("nan")
        print(f"[VALUE] corr(V, SF)={corr:+.3f}  slope={slope:+.3f}")
        # per-bucket
        for lbl, mask in [("SF win >+0.3", sf_vals > 0.3),
                          ("SF draw|.|<0.3", np.abs(sf_vals) < 0.3),
                          ("SF loss <-0.3", sf_vals < -0.3)]:
            if mask.sum() > 0:
                print(f"        {lbl:16s} n={mask.sum():3d}  V={V[mask].mean():+.3f}±{V[mask].std():.3f}  "
                      f"P(W)={wdl[mask,0].mean():.3f} P(D)={wdl[mask,1].mean():.3f} P(L)={wdl[mask,2].mean():.3f}")
        # ---- POLICY ----
        ents, top1s, top3s, illeg = [], [], [], []
        argmaxes = []
        uent = []
        for i, p in enumerate(pos):
            e, ue, t1, t3, am, im = policy_stats(pl[i], p["legal"])
            ents.append(e); uent.append(ue); top1s.append(t1); top3s.append(t3)
            illeg.append(im); argmaxes.append(am)
        ents = np.array(ents); uent = np.array(uent)
        print(f"[POLICY] entropy(legal): mean={ents.mean():.3f} "
              f"(uniform-legal={uent.mean():.3f}, ratio={ (ents/uent).mean():.2f})")
        print(f"[POLICY] top1 mass mean={np.mean(top1s):.3f}  top3 mass mean={np.mean(top3s):.3f}")
        print(f"[POLICY] illegal-move mass mean={np.mean(illeg):.3f}")
        # mode collapse: how concentrated are argmax actions across distinct positions?
        from collections import Counter
        c = Counter(argmaxes)
        topk = c.most_common(5)
        n_unique = len(c)
        print(f"[POLICY] argmax actions across {len(pos)} positions: {n_unique} unique; "
              f"top5 share={sum(v for _,v in topk)/len(pos):.0%} {topk}")

    # ---- POLICY TARGET sanity: visit-dist of stored targets at sampled plies ----
    print(f"\n{'='*72}\nPOLICY TARGET (stored visit dist) SANITY — step-6000 buffer\n{'='*72}")
    tgt_ent, tgt_top1, tgt_nlegal_mass = [], [], []
    for p in pos:
        pol = p["policy"]
        if pol is None:
            continue
        s = pol.sum()
        if s <= 0:
            continue
        q = pol / s
        nz = q[q > 0]
        tgt_ent.append(-np.sum(nz * np.log(nz + 1e-12)))
        tgt_top1.append(float(q.max()))
        legal = set(int(a) for a in p["legal"])
        mass_legal = sum(q[a] for a in legal)
        tgt_nlegal_mass.append(float(mass_legal))
    if tgt_ent:
        print(f"target entropy mean={np.mean(tgt_ent):.3f}  top1 mean={np.mean(tgt_top1):.3f}  "
              f"mass-on-legal mean={np.mean(tgt_nlegal_mass):.3f}  (n={len(tgt_ent)})")


if __name__ == "__main__":
    main()
