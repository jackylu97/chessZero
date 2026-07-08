"""Is the endgame REPRESENTATION diffuse? Does the latent encode distance-to-mate?

The value head reads the representation-net latent. If a frozen latent linearly
(or nonlinearly) encodes a winning endgame's DTZ, then a correct value head COULD
rank by it and the -0.41 corr is a HEAD/TARGET problem. If even an MLP probe can't
read DTZ from the latent, the endgames are 'diffuse' in the representation and no
head can fix it — the bottleneck is the world model.

Pulls DTZ-stratified winning TB positions, freezes net.representation(obs), and
probes latent -> (-DTZ) with ridge (linear) + a small MLP (nonlinear), per
checkpoint. Compares to the value HEAD's corr(V,-DTZ) on the same positions.

  linear/MLP probe HIGH, head LOW  => latent has DTZ; head/target is the problem
  linear LOW but MLP HIGH          => info is there but entangled (head too weak)
  both LOW                         => representation is DIFFUSE -> real bottleneck

Run: PYTHONPATH=. .venv/bin/python scripts/probe_endgame_latent_dtz.py \
        --buf <ckpt.buf> --checkpoints <a.pt> <b.pt>
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
        Xtr_n = (Xtr-mu)/sd
        w = np.linalg.solve(Xtr_n.T@Xtr_n + lam*np.eye(Xtr_n.shape[1]), Xtr_n.T@(ytr-ytr.mean()))
        preds[te] = ((X[te]-mu)/sd)@w + ytr.mean()
    return np.corrcoef(preds, y)[0,1] if y.std() > 1e-6 else float("nan")


def mlp_probe(X, y, hidden=64, epochs=300, seed=0):
    """Single 70/30 split, small MLP, weight decay; held-out corr."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]; ntr = int(0.7*len(X))
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0)+1e-6
    Xn = (X-mu)/sd
    Xt = torch.tensor(Xn[:ntr], dtype=torch.float32); yt = torch.tensor(y[:ntr], dtype=torch.float32)
    Xv = torch.tensor(Xn[ntr:], dtype=torch.float32); yv = y[ntr:]
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    ymu = yt.mean()
    for _ in range(epochs):
        opt.zero_grad(); p = net(Xt).squeeze(1)
        loss = ((p - (yt-ymu))**2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        pv = (net(Xv).squeeze(1) + ymu).numpy()
    return np.corrcoef(pv, yv)[0,1] if yv.std() > 1e-6 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--tb", default="data/syzygy")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--max-games", type=int, default=2500)
    args = ap.parse_args()
    dev = "cpu"
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = dev; hf = cfg.history_frames
    torch.serialization.add_safe_globals([MuZeroConfig])
    rb = ReplayBuffer(max_size=10_000_000); rb.load(args.buf, game=game)
    sp = [g for g in rb.buffer if not getattr(g, "external_values", [])][: args.max_games]
    tb = chess.syzygy.open_tablebase(args.tb)

    # DTZ-stratified winning endgame positions (avoid all-dtz=1)
    buckets = {(1,1): [], (2,3): [], (4,6): [], (7,12): [], (13,25): [], (26,200): []}
    per = max(1, args.n // len(buckets))
    for g in sp:
        if all(len(v) >= per for v in buckets.values()):
            break
        b = chess.Board(getattr(g, "start_fen", None) or chess.STARTING_FEN)
        for ply, a in enumerate(g.actions):
            if len(b.piece_map()) <= 5:
                try:
                    if tb.probe_wdl(b) == 2:
                        d = abs(int(tb.probe_dtz(b)))
                        for (lo, hi), lst in buckets.items():
                            if lo <= d <= hi and len(lst) < per:
                                lst.append((g._stack_history(ply, hf), float(d))); break
                except Exception:
                    pass
            mv = _action_to_move(int(a), b)
            if mv is None or mv not in b.legal_moves:
                break
            b.push(mv)
    tb.close()
    pos = [p for lst in buckets.values() for p in lst]
    if len(pos) < 60:
        print(f"only {len(pos)} winning TB positions found — too few"); return
    obs = [p[0] for p in pos]
    dtz = np.array([p[1] for p in pos])
    y = -dtz  # encode "-DTZ": closer to mate -> larger, matches the head-corr convention
    print(f"{len(pos)} winning endgame positions | DTZ range [{int(dtz.min())},{int(dtz.max())}] "
          f"mean {dtz.mean():.1f} std {dtz.std():.1f}")
    print(f"  DTZ-bucket counts: {[len(v) for v in buckets.values()]}\n")

    from src.model.utils import wdl_to_scalar
    print(f"{'checkpoint':>26} {'latent_std':>11} {'LINEAR':>8} {'MLP':>8} {'HEAD':>8}")
    for cp in args.checkpoints:
        ck = torch.load(cp, map_location=dev, weights_only=True)
        net = build_network(ck, game, cfg, dev); net.eval()
        with torch.no_grad():
            feats, Vs = [], []
            for i in range(0, len(obs), 64):
                xs = torch.stack(obs[i:i+64])
                h = net.representation(xs)
                feats.append(h.flatten(1).cpu().numpy())
                _, vl = net.prediction(h)
                Vs.append(wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).cpu().numpy())
            X = np.concatenate(feats, 0); V = np.concatenate(Vs)
        lin = ridge_cv(X, y); mlp = mlp_probe(X, y); head = np.corrcoef(V, y)[0,1]
        # shuffle-label control: held-out MLP on permuted DTZ should be ~0 if the
        # 0.80 is real signal and not overfitting the high-dim latent.
        y_shuf = np.random.default_rng(7).permutation(y)
        mlp_ctrl = mlp_probe(X, y_shuf)
        name = os.path.basename(cp)
        print(f"{name:>26} {X.std(0).mean():>11.4f} {lin:>+8.3f} {mlp:>+8.3f} {head:>+8.3f}  (shuffle-ctrl MLP {mlp_ctrl:+.3f})")
    print("\n  LINEAR/MLP = corr(probe(latent) -> -DTZ); HEAD = corr(value -> -DTZ).")
    print("  high probe + low head => head/target problem (latent HAS DTZ).")
    print("  both probes low      => representation is DIFFUSE (the real bottleneck).")


if __name__ == "__main__":
    main()
