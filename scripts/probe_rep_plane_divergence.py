"""Pin down the rep-plane (19/20) divergence between GPU generation and CPU
reconstruction. For a known diverging game, replay both engines step-by-step
and dump the exact ply where they disagree, plus both engines' repetition
verdicts and the underlying board FEN.
"""
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import chess

from src.config import get_config
from src.games.chess import ChessGame
from src.games.chess_gpu import GpuChessGame
from src.training.replay_buffer import GameHistory


def get_game(path, game, idx):
    with open(path, "rb") as f:
        header = pickle.load(f)
        ver = header["version"]
        for i in range(header["n_records"]):
            record, prio = pickle.load(f)
            if i == idx:
                if ver == 3:
                    return GameHistory.from_compact_dict(record, game)
                return record
    return None


def diagnose(actions):
    cpu_game = ChessGame()
    gpu = GpuChessGame(); gpu.max_plies = 1024
    cstate = cpu_game.reset()
    gstate = gpu.reset_batch(1, device="cpu")

    def rep_verdicts(ply):
        cobs = cpu_game.to_tensor(cstate).numpy()
        gobs = gpu.to_tensor_batch(gstate)[0].numpy()
        c19, c20 = float(cobs[19].max()), float(cobs[20].max())
        g19, g20 = float(gobs[19].max()), float(gobs[20].max())
        # python-chess raw verdicts
        b = cstate.board
        pc_rep2 = b.is_repetition(2)
        pc_rep3 = b.is_repetition(3)
        return (c19, c20, g19, g20, pc_rep2, pc_rep3, b.halfmove_clock, b.fen())

    divs = []
    c19, c20, g19, g20, r2, r3, hm, fen = rep_verdicts(0)
    if (c19 != g19) or (c20 != g20):
        divs.append((0, c19, c20, g19, g20, r2, r3, hm, fen))
    for i, a in enumerate(actions):
        cstate, _, cdone = cpu_game.step(cstate, int(a))
        gstate, _, _, _ = gpu.step_batch_with_legal(gstate, torch.tensor([int(a)], dtype=torch.int64))
        c19, c20, g19, g20, r2, r3, hm, fen = rep_verdicts(i + 1)
        if (c19 != g19) or (c20 != g20):
            divs.append((i + 1, c19, c20, g19, g20, r2, r3, hm, fen))
        if cdone:
            break
    return divs


def main():
    cfg = get_config("chess_small")
    cfg.device = "cpu"
    game = ChessGame()
    bf = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf"
    for gidx in (27, 35):
        gh = get_game(bf, game, gidx)
        print(f"\n=== game {gidx} len={len(gh.actions)} rep={gh.draw_by_repetition} ===")
        divs = diagnose(gh.actions)
        print(f"  {len(divs)} diverging plies (CPU rep-plane != GPU rep-plane)")
        for (ply, c19, c20, g19, g20, r2, r3, hm, fen) in divs:
            print(f"  ply {ply}: CPU(rep2={c19},rep3={c20}) GPU(rep2={g19},rep3={g20}) "
                  f"| python-chess is_repetition(2)={r2} (3)={r3} halfmove={hm}")
            print(f"      fen: {fen}")


if __name__ == "__main__":
    main()
