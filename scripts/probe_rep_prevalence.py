"""Quantify rep-plane divergence prevalence across many games and both shards.
For each game replay actions through GPU + python-chess, count plies where the
GPU rep2/rep3 verdict != python-chess rep2/rep3 verdict. python-chess is the
ground truth that from_compact_dict reproduces, so any divergence = a ply whose
rep plane differed between SELF-PLAY (GPU) and TRAINING (reconstructed CPU).
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, chess
from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.games.chess_gpu import GpuChessGame, _zobrist_hash, REP_HISTORY_K
from src.training.replay_buffer import GameHistory


def iter_games(path, game, limit):
    with open(path, "rb") as f:
        header = pickle.load(f); ver = header["version"]
        n = min(limit, header["n_records"])
        for _ in range(n):
            record, prio = pickle.load(f)
            yield GameHistory.from_compact_dict(record, game) if ver == 3 else record


def gpu_rep_verdict(st):
    cur = _zobrist_hash(st)[0].item()
    rc = int(st.rep_count[0].item())
    ring = st.rep_hashes[0, :rc].tolist()
    c = sum(1 for h in ring if h == cur)
    return (c >= 2), (c >= 3)


def main():
    cfg = get_config("chess_small"); cfg.device = "cpu"
    game = ChessGame()
    bufdir = "checkpoints/chess/2026_06_19_cold2_pc"
    for bf, lim in [(os.path.join(bufdir, "checkpoint_30000.buf"), 120)]:
        tot_games = 0; rep_games = 0
        games_with_div = 0; total_div_plies = 0; total_plies = 0
        gpu_under = 0  # plies where GPU said "no rep" but python-chess said "rep"
        gpu_over = 0
        for gh in iter_games(bf, game, lim):
            if gh.start_fen is not None:
                continue
            tot_games += 1
            if gh.draw_by_repetition:
                rep_games += 1
            gpu = GpuChessGame(); gpu.max_plies = 1024
            st = gpu.reset_batch(1, device="cpu")
            board = chess.Board()
            gdiv = 0
            for a in gh.actions:
                st, _, _, _ = gpu.step_batch_with_legal(st, torch.tensor([int(a)], dtype=torch.int64))
                board.push(_action_to_move(int(a), board))
                total_plies += 1
                g2, g3 = gpu_rep_verdict(st)
                p2, p3 = board.is_repetition(2), board.is_repetition(3)
                if (g2 != p2) or (g3 != p3):
                    gdiv += 1
                    total_div_plies += 1
                    # undercount: python-chess sees rep, GPU doesn't
                    if (p2 and not g2) or (p3 and not g3):
                        gpu_under += 1
                    if (g2 and not p2) or (g3 and not p3):
                        gpu_over += 1
            if gdiv:
                games_with_div += 1
        print(f"{os.path.basename(bf)}: games={tot_games} rep_draw_games={rep_games}")
        print(f"  games with >=1 rep-plane divergence: {games_with_div}/{tot_games} "
              f"({100*games_with_div/max(1,tot_games):.0f}%)")
        print(f"  diverging plies: {total_div_plies}/{total_plies} "
              f"({100*total_div_plies/max(1,total_plies):.2f}%)")
        print(f"  of which GPU-UNDERcount (GPU missed a rep that python-chess saw): {gpu_under}")
        print(f"          GPU-OVERcount: {gpu_over}")


if __name__ == "__main__":
    main()
