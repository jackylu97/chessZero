"""Generate the supervised TB endgame ANCHOR archive: tablebase-optimal
demonstration games from decisive ≤5-man seed FENs, saved as v2 compact shards
(the Stockfish-shard format) for the trainer's --tb-anchor-* injection.

Each game: both sides play TB-optimally from the seed to checkmate; per-ply
targets carry the soft win-preserving DTZ policy, the DTZ-shaped STM value,
and Gaviota |DTM| — the exact recipe the supervised endgame proxy validated
(KQvK 0.91 conversion, endgame_attention_findings_2026_06_28.md), packaged as
buffer games. game_outcome is WHITE-POV (the buffer-wide convention;
make_target's ``_outcome_onehot`` derives STM parity from start_fen since
2026-07-22). Dicts carry ``outcome_pov="white"`` via to_compact_dict; legacy
first-mover-POV archives are normalized at injection (trainer) / load
(from_compact_dict).

Run (CPU-only, safe alongside GPU training):
  PYTHONPATH=. .venv/bin/python scripts/gen_tb_anchor_games.py \
      --seeds data/endgame_seeds_train.txt --out data/tb_anchor \
      --games 25000 --workers 8
"""
import argparse
import pickle
import random
from multiprocessing import Pool
from pathlib import Path

import chess

_PROBER = None
_ARGS = None


def _worker_init(prober_kwargs: dict):
    global _PROBER
    from src.games.syzygy_probe import SyzygyRootProber
    _PROBER = SyzygyRootProber(**prober_kwargs)


def _worker_task(fen: str):
    """FEN → compact GameHistory dict, or None (non-decisive / unplayable)."""
    from src.training.replay_buffer import GameHistory
    from src.training.tb_playout import tb_playout, TBPlayoutError

    board = chess.Board(fen)
    try:
        res = tb_playout(board, _PROBER)
    except TBPlayoutError:
        return None
    h = GameHistory(game_name="chess")
    h.start_fen = fen
    h.actions = res["actions"]
    h.policies = res["policies"]
    h.root_values = res["root_values"]
    h.rewards = res["rewards"]
    h.tablebase_values = res["tablebase_values"]
    h.tablebase_moves_left = res["tablebase_moves_left"]
    h.tablebase_policy = res["tablebase_policy"]
    # White-POV (buffer-wide convention since 2026-07-22).
    h.game_outcome = 1.0 if res["winner"] == chess.WHITE else -1.0
    return h.to_compact_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/endgame_seeds_train.txt")
    ap.add_argument("--out", default="data/tb_anchor")
    ap.add_argument("--games", type=int, default=25000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tb-path", default="data/syzygy")
    ap.add_argument("--tb-gaviota-path", default="data/gaviota")
    # Match the production run's prober config (src/config.py defaults).
    ap.add_argument("--tb-value-dtz-shape", type=float, default=0.5)
    ap.add_argument("--tb-dtz-weight", type=float, default=0.05)
    ap.add_argument("--tb-policy-win-thresh", type=float, default=0.5)
    ap.add_argument("--tb-policy-temp", type=float, default=0.3)
    args = ap.parse_args()

    fens = [ln.strip() for ln in open(args.seeds) if ln.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(fens)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prober_kwargs = dict(
        path=args.tb_path,
        max_pieces=5,
        dtz_weight=args.tb_dtz_weight,
        draw_score=0.0,
        value_dtz_shape=args.tb_value_dtz_shape,
        gaviota_path=args.tb_gaviota_path,
        policy_win_thresh=args.tb_policy_win_thresh,
        policy_temp=args.tb_policy_temp,
    )

    done, shard_idx, shard, lens, outcomes = 0, 0, [], [], [0, 0]
    with Pool(args.workers, initializer=_worker_init,
              initargs=(prober_kwargs,)) as pool:
        for d in pool.imap_unordered(_worker_task, fens, chunksize=64):
            if d is None:
                continue
            shard.append(d)
            lens.append(len(d["actions"]))
            outcomes[0 if d["game_outcome"] > 0 else 1] += 1
            done += 1
            if len(shard) >= args.shard_size:
                p = out_dir / f"tb_anchor_{shard_idx:04d}.pkl"
                _write_shard(p, shard)
                print(f"wrote {p} ({len(shard)} games, {done}/{args.games} total)",
                      flush=True)
                shard, shard_idx = [], shard_idx + 1
            if done >= args.games:
                break
    if shard:
        p = out_dir / f"tb_anchor_{shard_idx:04d}.pkl"
        _write_shard(p, shard)
        print(f"wrote {p} ({len(shard)} games, {done} total)", flush=True)
    if lens:
        import numpy as np
        print(f"DONE: {done} games, mean length {np.mean(lens):.1f} plies "
              f"(max {max(lens)}), ply0-mover wins {outcomes[0]} / losses {outcomes[1]}")


def _write_shard(path: Path, dicts: list):
    """v2 compact shard: header + N compact-dict pickles (see _iter_shard_games)."""
    with open(path, "wb") as f:
        pickle.dump({"version": 2, "n_records": len(dicts)},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
        for d in dicts:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
