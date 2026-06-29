"""Inspect the value TARGETS in a tb_value run buffer: is the DTZ relabeling live,
correct, and changing what the value head trains on?

Loads once and reports:
  (1) composition — self-play games, how many reached the tablebase (tablebase_values
      populated), how many TB plies total.
  (2) stored tablebase_values distribution — range, sign breakdown, DTZ-shape band.
  (3) CORRECTNESS — replays each TB ply's board and re-probes Syzygy fresh; the stored
      value must match the fresh _position_value (catches GPU-decode / POV / sign bugs).
  (4) RELABEL EFFECT — runs make_target with tb_value_weight 0 vs 1.0 at TB plies and
      shows the value-target delta, esp. for DRAWN-but-won games (the key fix).
Dumps a sample to JSON for independent adversarial verification.

Run: PYTHONPATH=. .venv/bin/python scripts/inspect_tbvalue_targets.py --buf <ckpt.buf>
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, chess
from src.games.chess import ChessGame, _action_to_move
from src.games.syzygy_probe import SyzygyRootProber
from src.training.replay_buffer import ReplayBuffer

A = 4672


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buf", required=True)
    ap.add_argument("--tb", default="data/syzygy")
    ap.add_argument("--max-games", type=int, default=4000)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--out", default="/tmp/claude-0/-workspace-chessZero/9fcf1781-b6f9-40ae-a54e-8c4ef808ba32/scratchpad/tbval_samples.json")
    args = ap.parse_args()

    game = ChessGame()
    rb = ReplayBuffer(max_size=10_000_000); rb.load(args.buf, game=game)
    sp = [g for g in rb.buffer if not getattr(g, "external_values", [])][: args.max_games]
    warm = [g for g in rb.buffer if getattr(g, "external_values", [])]
    n_with_tb = sum(1 for g in sp if getattr(g, "tablebase_values", []))
    print(f"buffer {args.buf}")
    print(f"  total games {len(rb.buffer)} | self-play {len(sp)} | warmstart {len(warm)}")
    print(f"  self-play games WITH tablebase_values: {n_with_tb} ({n_with_tb/max(1,len(sp)):.1%})\n")

    prober = SyzygyRootProber(args.tb, max_pieces=5, dtz_weight=1.0, value_dtz_shape=0.5)
    vals = []                 # all finite stored tablebase_values
    checked = matched = 0     # correctness vs fresh Syzygy
    mism = []                 # mismatches
    drawn_won = 0             # TB plies where game drew but position is won (relabel targets)
    samples = []
    scount = {}               # stratified sample counts per category
    for g in sp:
        tv = getattr(g, "tablebase_values", [])
        if not tv:
            continue
        b = chess.Board(getattr(g, "start_fen", None) or chess.STARTING_FEN)
        boards = [b.copy()]
        for a in g.actions:
            mv = _action_to_move(int(a), b)
            if mv is None or mv not in b.legal_moves:
                break
            b.push(mv); boards.append(b.copy())
        outcome = float(getattr(g, "game_outcome", 0.0))
        for ply, v in enumerate(tv):
            if v != v or ply >= len(boards):       # NaN or out of range
                continue
            vals.append(float(v))
            fresh = prober._position_value(boards[ply])
            if fresh == fresh:
                checked += 1
                if abs(fresh - float(v)) < 0.06:
                    matched += 1
                elif len(mism) < 12:
                    mism.append((boards[ply].fen(), float(v), float(fresh)))
            # drawn-but-won detector (the relabel's whole point)
            if outcome == 0.0 and float(v) > 0.4:
                drawn_won += 1
            cat = ("drawnwon" if (outcome == 0.0 and float(v) > 0.4)
                   else "win" if float(v) > 0.4 else "loss" if float(v) < -0.4 else "draw")
            if scount.get(cat, 0) < 10 and len(samples) < args.sample:
                vt0 = np.asarray(game_make(g, ply, 0.0)[0])
                vt1 = np.asarray(game_make(g, ply, 1.0)[0])
                samples.append({
                    "cat": cat, "fen": boards[ply].fen(), "ply": ply,
                    "stm": "w" if ply % 2 == 0 else "b",
                    "stored_tb_value": round(float(v), 4),
                    "fresh_syzygy_value": round(float(fresh), 4) if fresh == fresh else None,
                    "game_outcome": outcome,
                    "value_target_no_relabel": [round(float(x), 4) for x in vt0],
                    "value_target_relabel": [round(float(x), 4) for x in vt1],
                })
                scount[cat] = scount.get(cat, 0) + 1
    prober.close()

    vals = np.array(vals)
    print(f"=== stored tablebase_values (n={len(vals)} TB plies) ===")
    if len(vals):
        wins = vals[vals > 0.4]; losses = vals[vals < -0.4]; draws = vals[np.abs(vals) <= 0.4]
        print(f"  range [{vals.min():+.3f}, {vals.max():+.3f}]  mean {vals.mean():+.3f}")
        print(f"  wins(>+0.4) {len(wins)} ({len(wins)/len(vals):.0%}) | draws {len(draws)} "
              f"({len(draws)/len(vals):.0%}) | losses(<-0.4) {len(losses)} ({len(losses)/len(vals):.0%})")
        if len(wins):
            print(f"  win magnitudes: min {wins.min():.3f} max {wins.max():.3f} "
                  f"(expect [0.5,1.0] DTZ band) — spread {wins.max()-wins.min():.3f}")
    print(f"\n=== CORRECTNESS vs fresh Syzygy ===")
    print(f"  stored matches fresh _position_value: {matched}/{checked} = {matched/max(1,checked):.1%}")
    for fen, st, fr in mism[:8]:
        print(f"    MISMATCH stored={st:+.3f} fresh={fr:+.3f}  {fen}")
    print(f"\n=== RELABEL EFFECT ===")
    print(f"  TB plies in DRAWN games but position won (>+0.4): {drawn_won}  <- these get relabeled draw->win")

    with open(args.out, "w") as f:
        json.dump(samples, f, indent=1)
    print(f"\n  dumped {len(samples)} samples -> {args.out}")
    # show a few drawn-but-won relabels inline
    print("\n  sample relabels (W,D,L target before -> after):")
    shown = 0
    for s in samples:
        if s["game_outcome"] == 0.0 and s["stored_tb_value"] > 0.4 and shown < 5:
            print(f"    {s['stm']} stored={s['stored_tb_value']:+.2f} outcome=draw  "
                  f"{s['value_target_no_relabel']} -> {s['value_target_relabel']}")
            shown += 1


def game_make(g, ply, w):
    out = g.make_target(ply, 0, -1, 1.0, A, value_head_type="wdl",
                        history_frames=1, tb_value_weight=w)
    return (out[3][0],)  # values[0] = the (W,D,L) target at this ply


if __name__ == "__main__":
    main()
