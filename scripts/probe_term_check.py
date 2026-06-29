"""Check what actually terminated game 27 on the GPU and whether the GPU's
internal threefold check agrees with the obs-time ring count.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.games.chess_gpu import GpuChessGame, _zobrist_hash, REP_HISTORY_K
from src.training.replay_buffer import GameHistory


def get_game(path, game, idx):
    with open(path, "rb") as f:
        header = pickle.load(f); ver = header["version"]
        for i in range(header["n_records"]):
            record, prio = pickle.load(f)
            if i == idx:
                return GameHistory.from_compact_dict(record, game) if ver == 3 else record


def main():
    cfg = get_config("chess_small"); cfg.device = "cpu"
    game = ChessGame()
    bf = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf"
    gh = get_game(bf, game, 27)
    print(f"game 27: len={len(gh.actions)} stored draw_by_repetition={gh.draw_by_repetition} "
          f"draw_by_no_progress={gh.draw_by_no_progress} outcome={gh.game_outcome}")
    gpu = GpuChessGame(); gpu.max_plies = 1024
    st = gpu.reset_batch(1, device="cpu")
    for i, a in enumerate(gh.actions):
        st, rew, done, _ = gpu.step_batch_with_legal(st, torch.tensor([int(a)], dtype=torch.int64))
        d = bool(st.done[0].item())
        if d:
            print(f"GPU terminates at ply {i+1}: done={d} "
                  f"threefold={bool(st.terminal_threefold[0].item())} "
                  f"no_progress={bool(st.terminal_no_progress[0].item())} "
                  f"winner={int(st.winner[0].item())} halfmove={int(st.halfmove[0].item())} "
                  f"rep_count={int(st.rep_count[0].item())}")
            # ring threefold count at termination
            cur = int(_zobrist_hash(st)[0].item())
            rc = int(st.rep_count[0].item())
            ring = st.rep_hashes[0, :rc].tolist()
            print(f"   obs-time matches_of_cur={sum(1 for h in ring if h==cur)}")
            break
    else:
        print("GPU never terminated in stored action list (len exhausted)")


if __name__ == "__main__":
    main()
