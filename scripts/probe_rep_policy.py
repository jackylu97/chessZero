"""At repetition plies, what does the POLICY TARGET (visit fractions) look like?
Does it concentrate on the repeating move => policy trained to shuffle. CPU."""
import argparse, os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import GameHistory


def load_buffer_games(path, game, limit=None):
    games = []
    with open(path, "rb") as f:
        header = pickle.load(f)
        version = header["version"]
        for i in range(header["n_records"]):
            record, priority = pickle.load(f)
            if version == 3:
                record = GameHistory.from_compact_dict(record, game)
            games.append(record)
            if limit and len(games) >= limit:
                break
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", default="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()
    cfg = get_config("chess_small"); cfg.device = "cpu"
    game = ChessGame()
    games = load_buffer_games(args.buf, game, args.limit)
    rep = [g for g in games if g.draw_by_repetition]

    # For each rep-draw, look at the visit fraction the PLAYED (repeating) action
    # gets in the policy target at the last 8 plies (where the loop closes).
    played_fracs_tail = []
    played_fracs_mid = []
    entropies_tail = []
    for g in rep:
        n = len(g.actions)
        for ply in range(n):
            pol = g.policies[ply] if ply < len(g.policies) else None
            if pol is None or len(pol) == 0:
                continue
            played = g.actions[ply]
            frac = float(pol[played]) if played < len(pol) else 0.0
            ent = -float(np.sum(pol[pol > 0] * np.log(pol[pol > 0] + 1e-12)))
            if ply >= n - 8:
                played_fracs_tail.append(frac)
                entropies_tail.append(ent)
            elif n // 4 <= ply <= 3 * n // 4:
                played_fracs_mid.append(frac)
    pt = np.array(played_fracs_tail); pm = np.array(played_fracs_mid)
    print(f"rep-draw games: {len(rep)}")
    print(f"PLAYED-move visit fraction in policy target:")
    print(f"   last-8-ply (loop region): mean={pt.mean():.3f} median={np.median(pt):.3f} "
          f"frac>0.5={np.mean(pt>0.5):.3f} frac>0.8={np.mean(pt>0.8):.3f}  (n={len(pt)})")
    print(f"   mid-game:                 mean={pm.mean():.3f} median={np.median(pm):.3f} "
          f"frac>0.5={np.mean(pm>0.5):.3f}  (n={len(pm)})")
    et = np.array(entropies_tail)
    print(f"   policy-target entropy (last-8): mean={et.mean():.3f} nats")

    # Print a couple concrete examples: top-3 of the policy target at the last few plies.
    print(f"\n--- concrete policy targets at last plies (3 games) ---")
    state0 = game.reset()
    for g in rep[:3]:
        # replay boards
        st = game.reset(); boards = [st]
        for a in g.actions:
            st, _, _ = game.step(st, a); boards.append(st)
        n = len(g.actions)
        print(f"\n  game len={n} outcome={g.game_outcome}")
        for ply in range(max(0, n - 4), n):
            pol = g.policies[ply]
            order = np.argsort(-pol)[:3]
            played = g.actions[ply]
            b = boards[ply].board
            parts = []
            for a in order:
                mv = _action_to_move(int(a), b)
                san = b.san(mv) if (mv and mv in b.legal_moves) else f"a{a}"
                tag = "*" if a == played else ""
                parts.append(f"{san}{tag}={pol[a]:.2f}")
            stm = "W" if ply % 2 == 0 else "B"
            print(f"    ply {ply}({stm}): {'  '.join(parts)}")


if __name__ == "__main__":
    main()
