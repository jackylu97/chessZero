"""Does the POLICY collapse onto repetition/shuffle moves over training?

Two angles:
 A) On positions sampled from the step-N buffer, compute the policy head's top
    move. Check what fraction of top moves are "shuffle" moves: a move that, if
    played, repeats a position already seen in that game (creates/extends a
    repetition) OR a non-capture non-pawn king/piece back-and-forth. We test the
    concrete property: does pushing the policy's top move yield a board that is a
    repetition (board.is_repetition(2)) given the game history up to that ply?
 B) Policy-target (visit dist) agreement with the head, and entropy trend, to
    confirm the target itself is sharpening onto these moves (not just the head).

Run: .venv/bin/python scripts/probe_policy_rep_collapse.py
"""
import os, sys, pickle, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move, _move_to_action
from src.training.replay_buffer import GameHistory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from eval_checkpoint_health import build_network

CKDIR = "checkpoints/chess/2026_06_19_cold2_pc"
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


def main():
    torch.serialization.add_safe_globals([MuZeroConfig])
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = DEV
    hf = cfg.history_frames

    for step in [1000, 6000, 30000]:
        games = load_games(f"{CKDIR}/checkpoint_{step}.buf", game, limit=200)
        sp = [g for g in games if not g.external_values]
        rng = random.Random(step)
        # reconstruct boards WITH move stack so is_repetition works
        samples = []  # (board_with_stack, gh, ply)
        tries = 0
        while len(samples) < 80 and tries < 20000:
            tries += 1
            g = rng.choice(sp)
            if len(g.actions) <= 12:
                continue
            # Cap reconstruction depth (full-game replay through python-chess is the
            # bottleneck on step-1000's ~385-ply games). Sample plies in [10,60].
            hi = min(len(g.actions) - 1, 60)
            if hi <= 10:
                continue
            ply = rng.randint(10, hi)
            board = chess.Board(); ok = True
            for a in g.actions[:ply]:
                mv = _action_to_move(int(a), board)
                if mv is None or mv not in board.legal_moves:
                    ok = False; break
                board.push(mv)
            if not ok or board.is_game_over():
                continue
            samples.append((board, g, ply))

        ck = torch.load(f"{CKDIR}/checkpoint_{step}.pt", map_location=DEV, weights_only=True)
        net = build_network(ck, game, cfg, DEV)
        obs = [g._stack_history(ply, hf) for (_, g, ply) in samples]
        with torch.no_grad():
            pls = []
            for i in range(0, len(obs), 64):
                xs = torch.stack(obs[i:i+64])
                pl, _ = net.prediction(net.representation(xs))
                pls.append(pl.float().cpu().numpy())
            P = np.concatenate(pls)

        head_rep, tgt_rep = 0, 0
        head_n, tgt_n = 0, 0
        played_rep = 0
        for i, (board, g, ply) in enumerate(samples):
            legal_moves = list(board.legal_moves)
            legal_actions = [_move_to_action(m, board.turn) for m in legal_moves]
            # head top move
            z = P[i][legal_actions]
            top_mv = legal_moves[int(np.argmax(z))]
            b2 = board.copy(); b2.push(top_mv)
            if b2.is_repetition(2):
                head_rep += 1
            head_n += 1
            # policy-target top move
            pol = g.policies[ply] if ply < len(g.policies) else None
            if pol is not None and pol.sum() > 0:
                tgt_top_a = int(np.argmax(pol))
                tgt_mv = _action_to_move(tgt_top_a, board)
                if tgt_mv is not None and tgt_mv in board.legal_moves:
                    b3 = board.copy(); b3.push(tgt_mv)
                    if b3.is_repetition(2):
                        tgt_rep += 1
                    tgt_n += 1
            # actually-played move
            if ply < len(g.actions):
                pm = _action_to_move(int(g.actions[ply]), board)
                if pm is not None and pm in board.legal_moves:
                    b4 = board.copy(); b4.push(pm)
                    if b4.is_repetition(2):
                        played_rep += 1

        print(f"step {step:5d}: head top-move creates repetition {head_rep}/{head_n} ({head_rep/max(1,head_n):.0%}) | "
              f"target top-move {tgt_rep}/{tgt_n} ({tgt_rep/max(1,tgt_n):.0%}) | "
              f"played-move {played_rep}/{len(samples)} ({played_rep/len(samples):.0%})")


if __name__ == "__main__":
    main()
