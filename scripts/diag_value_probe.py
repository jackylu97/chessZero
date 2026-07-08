"""Probe the value head's response to:
  (a) known winning/drawing/losing positions (calibration, no Stockfish)
  (b) the SAME position with rep planes toggled (does rep encoding shift V toward draw?)
  (c) the no-progress plane swept 0->1.5
This isolates: does the rep/np ENCODING itself bias the value head toward draw?
"""
import sys
import numpy as np
import torch
import chess

sys.path.insert(0, "/workspace/chessZero")
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, GameState
from scripts.eval_checkpoint_health import build_network

CKPT = "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"

CALIB = [
    ("white up a queen (WIN)",   "4k3/8/8/8/8/8/8/3QK3 w - - 0 1",  +1),
    ("white up a rook (WIN)",    "4k3/8/8/8/8/8/8/3RK3 w - - 0 1",  +1),
    ("dead K vs K (DRAW)",       "4k3/8/8/8/8/8/8/4K3 w - - 0 1",    0),
    ("white DOWN a queen (LOSS)","3qk3/8/8/8/8/8/8/4K3 w - - 0 1",  -1),
    ("white DOWN a rook (LOSS)", "3rk3/8/8/8/8/8/8/4K3 w - - 0 1",  -1),
    ("startpos (~draw)",         chess.STARTING_FEN,                 0),
]


def main():
    torch.serialization.add_safe_globals([MuZeroConfig])
    cfg = get_config("chess_small"); cfg.device = "cpu"
    game = ChessGame()
    HF = cfg.history_frames
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    net = build_network(ck, game, cfg, "cpu")
    C = game.num_planes

    def obs_from_fen(fen):
        board = chess.Board(fen)
        st = GameState(board=board, current_player=1 if board.turn == chess.WHITE else -1)
        frame = game.to_tensor(st)  # (C,8,8)
        # zero-pad history to HF frames (newest first, rest zero) — matches early-game stack
        stack = torch.cat([frame] + [torch.zeros_like(frame)] * (HF - 1), dim=0)
        return stack.unsqueeze(0), frame

    @torch.no_grad()
    def value_and_wdl(obs):
        hidden, _pl, value = net.initial_inference(obs)
        v = float(value.reshape(-1)[0])
        logits = net.prediction.value_head(hidden)
        probs = torch.softmax(logits, dim=-1).reshape(-1).tolist()
        return v, probs

    print("=== (a) calibration on constructed positions ===")
    print(f"{'desc':<28} {'truth':>5} {'net_V':>8} {'P(W)':>6} {'P(D)':>6} {'P(L)':>6}")
    for desc, fen, truth in CALIB:
        obs, _ = obs_from_fen(fen)
        v, p = value_and_wdl(obs)
        print(f"{desc:<28} {truth:>5} {v:>8.3f} {p[0]:>6.3f} {p[1]:>6.3f} {p[2]:>6.3f}")

    print("\n=== (b) rep-plane toggle on a WINNING position (white up a queen) ===")
    print("Set rep2/rep3 planes ON, keeping the board identical. Does V drop toward draw?")
    base_obs, frame = obs_from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    variants = [
        ("rep2=0 rep3=0 (no rep)", 0.0, 0.0, 0.0),
        ("rep2=1 rep3=0 (2-fold)", 1.0, 0.0, 0.0),
        ("rep2=1 rep3=1 (3-fold)", 1.0, 1.0, 0.0),
        ("np=0.5 (50 halfmoves)",  0.0, 0.0, 0.5),
        ("np=1.0 (100 halfmoves)", 0.0, 0.0, 1.0),
        ("rep3=1 + np=1.0",        1.0, 1.0, 1.0),
    ]
    print(f"{'variant':<26} {'net_V':>8} {'P(W)':>6} {'P(D)':>6} {'P(L)':>6}")
    for desc, r2, r3, npv in variants:
        obs = base_obs.clone()
        # newest frame is channels [0:C]; set planes 19,20,21
        obs[0, 19, :, :] = r2
        obs[0, 20, :, :] = r3
        obs[0, 21, :, :] = npv
        v, p = value_and_wdl(obs)
        print(f"{desc:<26} {v:>8.3f} {p[0]:>6.3f} {p[1]:>6.3f} {p[2]:>6.3f}")

    print("\n=== (c) same toggle on the START position (objectively ~draw) ===")
    base_obs2, _ = obs_from_fen(chess.STARTING_FEN)
    for desc, r2, r3, npv in variants:
        obs = base_obs2.clone()
        obs[0, 19, :, :] = r2
        obs[0, 20, :, :] = r3
        obs[0, 21, :, :] = npv
        v, p = value_and_wdl(obs)
        print(f"{desc:<26} {v:>8.3f} {p[0]:>6.3f} {p[1]:>6.3f} {p[2]:>6.3f}")


if __name__ == "__main__":
    main()
