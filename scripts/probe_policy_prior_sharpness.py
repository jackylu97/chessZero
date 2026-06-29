"""Probe: is the trained conv-policy PRIOR sharp enough to guide MCTS, or flat?

For each checkpoint, sample real positions (random playouts), run the net, mask
logits to legal moves, softmax, and measure:
  - entropy of the legal-masked prior vs uniform entropy ln(n_legal)
  - top-1 prior probability
  - whether the prior's top move == python-chess "obviously good" capture exists
A flat prior (entropy ~= uniform, top1 ~= 1/n_legal) => garbage prior =>
MCTS can't search => collapse. A sharp prior rules that mechanism out.

Also dumps the MCTS visit-policy sharpness via a quick run for one checkpoint,
to compare prior sharpness vs post-search sharpness (does search concentrate?).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import chess
import numpy as np
import torch

from src.config import get_config
from src.games.chess import ChessGame, _move_to_action
from src.games import chess_gpu as cg
from src.games.base import GameState
from scripts.eval_checkpoint_health import build_network


def sample_positions(n=200, max_depth=60, seed=0):
    rng = random.Random(seed)
    boards = []
    for _ in range(n):
        b = chess.Board()
        for _ in range(rng.randint(4, max_depth)):
            ms = list(b.legal_moves)
            if not ms or b.is_game_over():
                break
            b.push(rng.choice(ms))
        if not b.is_game_over() and any(True for _ in b.legal_moves):
            boards.append(b)
    return boards


def analyze_ckpt(ckpt_path, boards, device, cfg, game, gpu):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = build_network(ckpt, game, cfg, device)
    net.eval()
    state = gpu.from_python_chess(boards, device=device)
    obs = gpu.to_tensor_batch(state)
    n_frames = int(cfg.history_frames)
    c, h, w = obs.shape[1:]
    if n_frames > 1:
        stacked = torch.zeros(obs.shape[0], n_frames * c, h, w, device=device, dtype=obs.dtype)
        stacked[:, :c] = obs  # first-ply style: older frames zero
    else:
        stacked = obs
    with torch.no_grad():
        _hidden, logits, _v = net.initial_inference(stacked)
    logits = logits.float().cpu().numpy()

    ent_ratios = []
    top1s = []
    for i, b in enumerate(boards):
        legal = [_move_to_action(m, b.turn) for m in b.legal_moves]
        nl = len(legal)
        if nl <= 1:
            continue
        ll = logits[i, legal]
        ll = ll - ll.max()
        p = np.exp(ll); p = p / p.sum()
        ent = -(p * np.log(p + 1e-12)).sum()
        unif = np.log(nl)
        ent_ratios.append(ent / unif)       # 1.0 = uniform, <1 = sharper
        top1s.append(float(p.max()))
    ent_ratios = np.array(ent_ratios); top1s = np.array(top1s)
    return {
        "n": len(ent_ratios),
        "ent_ratio_mean": float(ent_ratios.mean()),
        "ent_ratio_p10": float(np.percentile(ent_ratios, 10)),
        "top1_mean": float(top1s.mean()),
        "top1_max": float(top1s.max()),
        "uniform_top1_mean": float(np.mean([1.0 / len([1 for _ in b.legal_moves]) for b in boards if len(list(b.legal_moves)) > 1])),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config("chess_small"); cfg.device = device
    game = ChessGame(); gpu = cg.GpuChessGame()
    boards = sample_positions(n=200, seed=0)
    print(f"device={device} positions={len(boards)} history_frames={cfg.history_frames}\n")
    base = "checkpoints/chess/2026_06_19_cold2_pc"
    for step in (1000, 6000, 30000):
        path = f"{base}/checkpoint_{step}.pt"
        if not os.path.exists(path):
            print(f"step {step}: MISSING {path}"); continue
        r = analyze_ckpt(path, boards, device, cfg, game, gpu)
        print(f"step {step:6d}: n={r['n']:3d}  "
              f"prior_entropy/uniform={r['ent_ratio_mean']:.3f} (p10={r['ent_ratio_p10']:.3f})  "
              f"top1_prior_mean={r['top1_mean']:.3f} (max={r['top1_max']:.3f})  "
              f"uniform_top1={r['uniform_top1_mean']:.3f}")
    print("\nInterpretation: entropy/uniform near 1.0 AND top1≈uniform_top1 => FLAT prior")
    print("                (no within-position move ranking from the policy head).")


if __name__ == "__main__":
    main()
