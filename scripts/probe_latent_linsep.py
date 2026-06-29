"""Does the learned REPRESENTATION linearly encode position value (vs Stockfish)?

If the value target carries no signal (V^pi), the value HEAD can't learn calib.
But the deeper question: has the SHARED BACKBONE/representation learned a latent
that even *could* be read out into a correct value? Train a cheap linear probe
(ridge) from frozen latents -> Stockfish eval, report held-out R / corr.

 - High probe corr + low head corr  => head/target problem only; repr is fine
   (so a clean value target WOULD fix it) -> V^pi symptom, NOT a repr degeneracy.
 - Low probe corr too                => the representation itself is degenerate
   (collapsed / position-blind) -> a deeper mechanistic problem.

Compares checkpoint 1000 / 6000 / 30000 (does the latent's value-readability
improve, stay flat, or COLLAPSE over training?).

Run: .venv/bin/python scripts/probe_latent_linsep.py
"""
import os, sys, pickle, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess, chess.engine

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import GameHistory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from eval_checkpoint_health import build_network

CKDIR = "checkpoints/chess/2026_06_19_cold2_pc"
SF = "/usr/games/stockfish"
DEV = "cpu"


def load_games(buf_path, game, limit=None):
    games = []
    with open(buf_path, "rb") as f:
        header = pickle.load(f)
        for _ in range(header["n_records"]):
            d, _prio = pickle.load(f)
            try:
                games.append(GameHistory.from_compact_dict(d, game))
            except Exception:
                continue
            if limit and len(games) >= limit:
                break
    return games


def ridge_cv(X, y, lam=10.0, folds=5, seed=0):
    """Simple k-fold ridge; return mean held-out corr."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X = X[idx]; y = y[idx]
    n = len(X); fs = n // folds
    preds = np.zeros(n)
    for k in range(folds):
        te = slice(k*fs, (k+1)*fs if k < folds-1 else n)
        mask = np.ones(n, bool); mask[te] = False
        Xtr, ytr = X[mask], y[mask]
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        Xtr_n = (Xtr - mu) / sd
        A = Xtr_n.T @ Xtr_n + lam * np.eye(Xtr_n.shape[1])
        w = np.linalg.solve(A, Xtr_n.T @ (ytr - ytr.mean()))
        b = ytr.mean()
        Xte_n = (X[te] - mu) / sd
        preds[te] = Xte_n @ w + b
    c = np.corrcoef(preds, y)[0,1] if y.std() > 1e-6 else float("nan")
    return c


def main():
    torch.serialization.add_safe_globals([MuZeroConfig])
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = DEV
    hf = cfg.history_frames
    games = load_games(f"{CKDIR}/checkpoint_6000.buf", game, limit=500)
    sp = [g for g in games if not g.external_values]
    rng = random.Random(3)
    samples = []
    while len(samples) < 400 and len(samples) < 100000:
        g = rng.choice(sp)
        if len(g.actions) <= 10:
            continue
        ply = rng.randint(8, len(g.actions) - 1)
        board = chess.Board(); ok = True
        for a in g.actions[:ply]:
            mv = _action_to_move(int(a), board)
            if mv is None or mv not in board.legal_moves:
                ok = False; break
            board.push(mv)
        if not ok or board.is_game_over():
            continue
        samples.append((board.copy(), g._stack_history(ply, hf)))
        if len(samples) >= 400:
            break

    print(f"Grading {len(samples)} positions with Stockfish...")
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    y = []
    for board, _ in samples:
        info = eng.analyse(board, chess.engine.Limit(depth=12))
        sc = info["score"].pov(board.turn)
        y.append((1.0 if sc.mate() > 0 else -1.0) if sc.is_mate() else float(np.tanh(sc.score()/400.0)))
    eng.quit()
    y = np.array(y)
    obs = [s[1] for s in samples]
    print(f"SF eval std={y.std():.3f}")

    for step in [1000, 6000, 30000]:
        ck = torch.load(f"{CKDIR}/checkpoint_{step}.pt", map_location=DEV, weights_only=True)
        net = build_network(ck, game, cfg, DEV)
        with torch.no_grad():
            feats = []
            for i in range(0, len(obs), 64):
                xs = torch.stack(obs[i:i+64])
                h = net.representation(xs)
                feats.append(h.flatten(1).cpu().numpy())
            X = np.concatenate(feats, 0)
        # latent collapse stats
        per_dim_std = X.std(0).mean()
        probe_corr = ridge_cv(X, y, lam=10.0)
        # head corr for reference
        from src.model.utils import wdl_to_scalar
        import torch.nn.functional as F
        with torch.no_grad():
            Vs = []
            for i in range(0, len(obs), 64):
                xs = torch.stack(obs[i:i+64])
                _, vl = net.prediction(net.representation(xs))
                Vs.append(wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).cpu().numpy())
            V = np.concatenate(Vs)
        head_corr = np.corrcoef(V, y)[0,1] if V.std() > 1e-6 else float("nan")
        print(f"step {step:5d}: latent per-dim std={per_dim_std:.4f} | "
              f"LINEAR-PROBE corr(latent->SF)={probe_corr:+.3f} | head corr(V->SF)={head_corr:+.3f}")


if __name__ == "__main__":
    main()
