"""Confirm the in-distribution value-signal collapse and tie it to buffer
homogeneity. For each checkpoint, sample positions from ITS OWN buffer (the
data its value head was actually trained on at that time), grade with SF, and
report:
 - SF eval spread in that buffer (input diversity the head saw)
 - linear-probe corr(latent->SF) and head corr(V->SF)
 - latent per-dim std on those positions

If input diversity (SF std) shrinks AND probe corr shrinks together, the value
blindness tracks buffer homogenization (the self-reinforcing collapse), not a
fixed code bug. Uses fewer positions/lower SF depth to stay within time budget.
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
from probe_latent_linsep import ridge_cv

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


def sample_obs(games, hf, n, seed):
    rng = random.Random(seed)
    sp = [g for g in games if not g.external_values]
    out = []
    tries = 0
    while len(out) < n and tries < n * 80:
        tries += 1
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
        out.append((board.copy(), g._stack_history(ply, hf)))
    return out


def main():
    torch.serialization.add_safe_globals([MuZeroConfig])
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = DEV
    hf = cfg.history_frames
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    print(f"{'step':>6} {'bufSFstd':>9} {'latStd':>7} {'probe_corr':>11} {'head_corr':>10}")
    for step in [1000, 6000, 30000]:
        games = load_games(f"{CKDIR}/checkpoint_{step}.buf", game, limit=200)
        samples = sample_obs(games, hf, n=40, seed=step)
        y = []
        for board, _ in samples:
            info = eng.analyse(board, chess.engine.Limit(depth=6))
            sc = info["score"].pov(board.turn)
            y.append((1.0 if sc.mate() > 0 else -1.0) if sc.is_mate() else float(np.tanh(sc.score()/400.0)))
        y = np.array(y)
        obs = [s[1] for s in samples]
        ck = torch.load(f"{CKDIR}/checkpoint_{step}.pt", map_location=DEV, weights_only=True)
        net = build_network(ck, game, cfg, DEV)
        from src.model.utils import wdl_to_scalar
        with torch.no_grad():
            feats, Vs = [], []
            for i in range(0, len(obs), 64):
                xs = torch.stack(obs[i:i+64])
                h = net.representation(xs)
                feats.append(h.flatten(1).cpu().numpy())
                _, vl = net.prediction(h)
                Vs.append(wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).cpu().numpy())
            X = np.concatenate(feats); V = np.concatenate(Vs)
        pc = ridge_cv(X, y, lam=10.0)
        hc = np.corrcoef(V, y)[0,1] if V.std() > 1e-6 else float("nan")
        print(f"{step:>6} {y.std():>9.3f} {X.std(0).mean():>7.4f} {pc:>11.3f} {hc:>10.3f}")
    eng.quit()


if __name__ == "__main__":
    main()
