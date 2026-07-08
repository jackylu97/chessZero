"""Is the value head's DTZ ranking different at 3 vs 4 vs 5 pieces? The relabeling
fires at <=5 pieces; the aggregate corr / the 3-piece KQvK,KRvK conversion test may
mask a piece-count-dependent effect. Break corr(value,-DTZ) AND the latent linear
separability down by piece count, on winning (wdl=2) TB positions from the buffer.

Run: PYTHONPATH=. .venv/bin/python scripts/probe_corr_by_piececount.py \
        --buf <ckpt.buf> --checkpoint <ckpt.pt>
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, chess, chess.syzygy
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import ReplayBuffer
from scripts.eval_checkpoint_health import build_network


def ridge_cv(X, y, lam=10.0, folds=5, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X)); X, y = X[idx], y[idx]
    n = len(X); fs = n // folds; preds = np.zeros(n)
    for k in range(folds):
        te = slice(k*fs, (k+1)*fs if k < folds-1 else n)
        m = np.ones(n, bool); m[te] = False
        Xtr, ytr = X[m], y[m]; mu = Xtr.mean(0); sd = Xtr.std(0)+1e-6
        Xn = (Xtr-mu)/sd
        w = np.linalg.solve(Xn.T@Xn + lam*np.eye(Xn.shape[1]), Xn.T@(ytr-ytr.mean()))
        preds[te] = ((X[te]-mu)/sd)@w + ytr.mean()
    return np.corrcoef(preds, y)[0,1] if y.std() > 1e-6 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tb", default="data/syzygy")
    ap.add_argument("--per-pc", type=int, default=300)
    ap.add_argument("--max-games", type=int, default=3000)
    args = ap.parse_args()
    dev = "cpu"
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = dev; hf = cfg.history_frames
    torch.serialization.add_safe_globals([MuZeroConfig])
    rb = ReplayBuffer(max_size=10_000_000); rb.load(args.buf, game=game)
    sp = [g for g in rb.buffer if not getattr(g, "external_values", [])][: args.max_games]
    tb = chess.syzygy.open_tablebase(args.tb)

    # DTZ-stratify WITHIN each piece count so 5-piece isn't all DTZ=1.
    DZB = [(1, 1), (2, 4), (5, 10), (11, 25), (26, 80)]
    per_cell = max(1, args.per_pc // len(DZB))
    pc_cells = {pc: {b: [] for b in DZB} for pc in (3, 4, 5)}
    def full(pc): return all(len(pc_cells[pc][b]) >= per_cell for b in DZB)
    for g in sp:
        if all(full(pc) for pc in (3, 4, 5)):
            break
        b = chess.Board(getattr(g, "start_fen", None) or chess.STARTING_FEN)
        for ply, a in enumerate(g.actions):
            npc = len(b.piece_map())
            if npc in pc_cells:
                try:
                    if tb.probe_wdl(b) == 2:
                        d = abs(int(tb.probe_dtz(b)))
                        for (lo, hi) in DZB:
                            if lo <= d <= hi and len(pc_cells[npc][(lo, hi)]) < per_cell:
                                pc_cells[npc][(lo, hi)].append((g._stack_history(ply, hf), float(d)))
                                break
                except Exception:
                    pass
            mv = _action_to_move(int(a), b)
            if mv is None or mv not in b.legal_moves:
                break
            b.push(mv)
    tb.close()
    pc_pos = {pc: [p for cell in pc_cells[pc].values() for p in cell] for pc in (3, 4, 5)}

    from src.model.utils import wdl_to_scalar
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ck, game, cfg, dev); net.eval()
    print(f"checkpoint {os.path.basename(args.checkpoint)}\n")
    print(f"{'pieces':>7} {'n':>5} {'DTZ range':>12} {'HEAD corr':>10} {'LINEAR corr':>12}")
    for pc in (3, 4, 5):
        pos = pc_pos[pc]
        if len(pos) < 40:
            print(f"{pc:>7} {len(pos):>5}  (too few)"); continue
        obs = [p[0] for p in pos]; dtz = np.array([p[1] for p in pos]); y = -dtz
        with torch.no_grad():
            feats, Vs = [], []
            for i in range(0, len(obs), 64):
                xs = torch.stack(obs[i:i+64]); h = net.representation(xs)
                feats.append(h.flatten(1).cpu().numpy())
                _, vl = net.prediction(h)
                Vs.append(wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).cpu().numpy())
            X = np.concatenate(feats, 0); V = np.concatenate(Vs)
        head = np.corrcoef(V, y)[0,1] if y.std() > 1e-6 else float("nan")
        lin = ridge_cv(X, y)
        print(f"{pc:>7} {len(pos):>5} {f'[{int(dtz.min())},{int(dtz.max())}]':>12} "
              f"{head:>+10.3f} {lin:>+12.3f}")
    print("\n  HEAD = corr(value,-DTZ); LINEAR = corr(ridge(latent),-DTZ).")
    print("  If 5-piece HEAD corr is clearly > 3-piece, the relabel helps where it's active.")


if __name__ == "__main__":
    main()
