"""Deeper integrity probe: NO ply cap, full games, many games, terminal obs,
and a focus on the repetition planes (19,20) on long/repeating games where the
GPU Zobrist ring (depth 160) and CPU full-stack is_repetition could diverge.
"""
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import get_config
from src.games.chess import ChessGame
from src.games.chess_gpu import GpuChessGame
from src.training.replay_buffer import GameHistory


def iter_buf_games(path, game, limit=None):
    with open(path, "rb") as f:
        header = pickle.load(f)
        ver = header["version"]
        n = header["n_records"]
        if limit:
            n = min(n, limit)
        for _ in range(n):
            record, prio = pickle.load(f)
            if ver == 3:
                record = GameHistory.from_compact_dict(record, game)
            yield record


def full_obs_compare(actions, num_planes):
    """Replay actions through CPU + GPU, compare EVERY ply's obs (no cap)."""
    cpu_game = ChessGame()
    gpu = GpuChessGame(); gpu.max_plies = 1024
    cstate = cpu_game.reset()
    gstate = gpu.reset_batch(1, device="cpu")

    plane_max = np.zeros(num_planes)
    plane_plies = np.zeros(num_planes, dtype=np.int64)
    ncmp = 0
    worst = None

    def cmp(ply):
        nonlocal ncmp, worst
        cobs = cpu_game.to_tensor(cstate).numpy()
        gobs = gpu.to_tensor_batch(gstate)[0].numpy()
        d = np.abs(cobs - gobs)
        for p in range(num_planes):
            m = float(d[p].max())
            if m > plane_max[p]:
                plane_max[p] = m
            if m > 1e-5:
                plane_plies[p] += 1
        ncmp += 1

    cmp(0)
    for i, a in enumerate(actions):
        cstate, _, cdone = cpu_game.step(cstate, int(a))
        gstate, _, _, _ = gpu.step_batch_with_legal(gstate, torch.tensor([int(a)], dtype=torch.int64))
        cmp(i + 1)
        if cdone:
            break
    return plane_max, plane_plies, ncmp


def main():
    cfg = get_config("chess_small")
    cfg.device = "cpu"
    game = ChessGame()
    NP = ChessGame.num_planes

    bufdir = "checkpoints/chess/2026_06_19_cold2_pc"
    buf_files = [
        (os.path.join(bufdir, "checkpoint_2000.buf"), 15),   # early big games
        (os.path.join(bufdir, "checkpoint_30000.buf"), 40),  # late repetition-heavy
    ]

    global_plane_max = np.zeros(NP)
    global_plane_plies = np.zeros(NP, dtype=np.int64)
    total_plies = 0
    total_games = 0
    longest = 0
    any_div_games = 0
    div_game_examples = []

    for bf, lim in buf_files:
        if not os.path.exists(bf):
            print(f"SKIP {bf}")
            continue
        print(f"\n===== {os.path.basename(bf)} (up to {lim} games, full length) =====")
        nrep = 0
        for gi, gh in enumerate(iter_buf_games(bf, game, limit=lim)):
            if gh.start_fen is not None:
                continue
            pm, pp, ncmp = full_obs_compare(gh.actions, NP)
            total_plies += ncmp
            total_games += 1
            longest = max(longest, len(gh.actions))
            if gh.draw_by_repetition:
                nrep += 1
            div = [p for p in range(NP) if pm[p] > 1e-5]
            if div:
                any_div_games += 1
                if len(div_game_examples) < 8:
                    div_game_examples.append((os.path.basename(bf), gi, len(gh.actions),
                                              gh.draw_by_repetition, div,
                                              {p: float(pm[p]) for p in div}))
            for p in range(NP):
                global_plane_max[p] = max(global_plane_max[p], pm[p])
                global_plane_plies[p] += pp[p]
            if div:
                print(f"  game{gi} len={len(gh.actions)} rep={gh.draw_by_repetition} "
                      f"DIVERGES planes={div}", flush=True)
            elif gi % 5 == 0:
                print(f"  game{gi} len={len(gh.actions)} ok", flush=True)
        print(f"  rep-draw games in sample: {nrep}", flush=True)

    print(f"\n===== SUMMARY =====")
    print(f"games compared: {total_games}, plies compared: {total_plies}, "
          f"longest game: {longest} plies")
    print(f"games with ANY obs divergence: {any_div_games}/{total_games}")
    print(f"per-plane MAX abs diff across all plies:")
    names = {0:"ownP",1:"ownN",2:"ownB",3:"ownR",4:"ownQ",5:"ownK",
             6:"oppP",7:"oppN",8:"oppB",9:"oppR",10:"oppQ",11:"oppK",
             12:"castOwnKS",13:"castOwnQS",14:"castOppKS",15:"castOppQS",
             16:"ep",17:"turn",18:"movecount",19:"rep2",20:"rep3",21:"noprog"}
    for p in range(NP):
        flag = "  <-- DIVERGES" if global_plane_max[p] > 1e-5 else ""
        print(f"  plane {p:2d} {names.get(p,'?'):>10}: max|d|={global_plane_max[p]:.5f} "
              f"in {global_plane_plies[p]} plies{flag}")
    if div_game_examples:
        print(f"\nDivergence examples:")
        for bf, gi, ln, rep, div, mx in div_game_examples:
            print(f"  {bf} game{gi} len={ln} rep={rep} bad_planes={div} maxd={mx}")


if __name__ == "__main__":
    main()
