"""Fast version: does the POLICY collapse onto repetition moves over training?

Reads compact buffer records directly (no observation replay). Reconstructs each
sampled board by replaying actions through python-chess, builds the 8-frame
stacked obs from to_tensor of the last 8 boards, runs the policy head, and checks
whether the head's TOP move (and the stored policy-target top move, and the
actually-played move) creates a repetition (board.is_repetition(2)).
"""
import os, sys, pickle, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess

from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move, _move_to_action
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from eval_checkpoint_health import build_network

CKDIR = "checkpoints/chess/2026_06_19_cold2_pc"
DEV = "cpu"


def read_records(path, limit=None):
    recs = []
    with open(path, "rb") as f:
        header = pickle.load(f)
        for _ in range(header["n_records"]):
            d, _prio = pickle.load(f)
            recs.append(d)
            if limit and len(recs) >= limit:
                break
    return recs


def stacked_obs_from_boards(game, boards, hf):
    """boards: list of chess.Board snapshots, oldest..newest at the target ply.
    Build (num_planes*hf, 8, 8): newest-first, zero-pad missing earlier frames."""
    frames = []
    for t in range(hf):
        j = len(boards) - 1 - t
        if j >= 0:
            st = game.reset(); st.board = boards[j]
            frames.append(game.to_tensor(st))
        else:
            frames.append(torch.zeros_like(frames[-1]))
    return torch.cat(frames, dim=0)


def main():
    torch.serialization.add_safe_globals([MuZeroConfig])
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = DEV
    hf = cfg.history_frames

    for step in [6000, 30000]:
        recs = read_records(f"{CKDIR}/checkpoint_{step}.buf", limit=120)
        sp = [d for d in recs if len(d.get("external_values", [])) == 0]
        rng = random.Random(step)
        samples = []  # (boards_to_ply, d, ply)
        tries = 0
        while len(samples) < 100 and tries < 5000:
            tries += 1
            d = rng.choice(sp)
            acts = d["actions"]
            if len(acts) <= 12:
                continue
            hi = min(len(acts) - 1, 55)
            if hi <= 12:
                continue
            ply = rng.randint(12, hi)
            board = chess.Board(); boards = [board.copy()]; ok = True
            for a in acts[:ply]:
                mv = _action_to_move(int(a), board)
                if mv is None or mv not in board.legal_moves:
                    ok = False; break
                board.push(mv); boards.append(board.copy())
            if not ok or board.is_game_over():
                continue
            samples.append((boards, d, ply))

        ck = torch.load(f"{CKDIR}/checkpoint_{step}.pt", map_location=DEV, weights_only=True)
        net = build_network(ck, game, cfg, DEV)
        obs = [stacked_obs_from_boards(game, b, hf) for (b, _, _) in samples]
        with torch.no_grad():
            P = []
            for i in range(0, len(obs), 64):
                xs = torch.stack(obs[i:i+64])
                pl, _ = net.prediction(net.representation(xs))
                P.append(pl.float().cpu().numpy())
            P = np.concatenate(P)

        head_rep = head_n = tgt_rep = tgt_n = played_rep = 0
        # also: head argmax concentration (mode collapse)
        argmaxes = []
        for i, (boards, d, ply) in enumerate(samples):
            board = boards[-1]
            legal_moves = list(board.legal_moves)
            legal_actions = [_move_to_action(m, board.turn) for m in legal_moves]
            z = P[i][legal_actions]
            top_mv = legal_moves[int(np.argmax(z))]
            argmaxes.append(int(legal_actions[int(np.argmax(z))]))
            b2 = board.copy(); b2.push(top_mv)
            head_rep += int(b2.is_repetition(2)); head_n += 1
            # policy-target (reconstruct from compact: onehot or sparse)
            mode = d["policies_mode"]
            tgt_a = None
            if mode == "onehot":
                tgt_a = int(d["policies_data"][ply]) if ply < len(d["policies_data"]) else None
            else:
                if ply < len(d["policies_data"]):
                    idxs, vals = d["policies_data"][ply]
                    if len(vals):
                        tgt_a = int(idxs[int(np.argmax(vals))])
            if tgt_a is not None:
                tm = _action_to_move(tgt_a, board)
                if tm is not None and tm in board.legal_moves:
                    b3 = board.copy(); b3.push(tm)
                    tgt_rep += int(b3.is_repetition(2)); tgt_n += 1
            if ply < len(d["actions"]):
                pm = _action_to_move(int(d["actions"][ply]), board)
                if pm is not None and pm in board.legal_moves:
                    b4 = board.copy(); b4.push(pm)
                    played_rep += int(b4.is_repetition(2))
        from collections import Counter
        c = Counter(argmaxes)
        print(f"step {step:5d}: head-top->rep {head_rep}/{head_n} ({head_rep/max(1,head_n):.0%}) | "
              f"target-top->rep {tgt_rep}/{tgt_n} ({tgt_rep/max(1,tgt_n):.0%}) | "
              f"played->rep {played_rep}/{len(samples)} ({played_rep/len(samples):.0%}) | "
              f"head argmax: {len(c)} unique / {len(samples)}")


if __name__ == "__main__":
    main()
