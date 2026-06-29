"""Probe the reward head: in chess, reward is 0 for every non-terminal transition
and +/-1/0 only at the terminal step. The reward head should output ~0 for all
moves from a non-terminal position. If it outputs large values, MCTS Q
(= reward - discount*child_value) is being driven by a hallucinated reward.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F, chess
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _move_to_action, _action_to_move
from src.training.replay_buffer import stack_with_history
from scripts.eval_checkpoint_health import build_network
from scripts.probe_draw_smoking_gun import build_obs_with_history

torch.serialization.add_safe_globals([MuZeroConfig])
CKPT = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"


@torch.no_grad()
def main():
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = "cpu"
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    net = build_network(ckpt, game, cfg, "cpu")
    from src.model.utils import wdl_to_scalar

    positions = [
        ("KQvK mate-in-1 (7k/5Q2/6K1/...)", chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")),
        ("KQvK shuffle (in-distribution)", None),  # filled below with history
    ]
    # Build shuffle board with history
    b5 = chess.Board("7k/8/6K1/8/8/8/8/3Q4 w - - 0 1")
    for s in ["Qd4","Kg8","Qd1","Kh8","Qd4","Kg8","Qd1","Kh8"]:
        b5.push_san(s)
    positions[1] = ("KQvK shuffle (in-distribution, full history)", b5)

    for name, board in positions:
        obs, legal, _ = build_obs_with_history(game, board, cfg.history_frames)
        h, _, _ = net.initial_inference(obs.unsqueeze(0))
        acts = torch.tensor(legal, dtype=torch.long)
        hn = h.expand(len(legal), *h.shape[1:]).contiguous()
        next_h, reward, pol, val = net.recurrent_inference(hn, acts)
        reward = reward.view(-1).numpy()
        valsc = wdl_to_scalar(net.prediction(next_h)[1].float(), draw_score=cfg.draw_score).numpy()
        print(f"\n{name}")
        print(f"  reward head over {len(legal)} legal moves: "
              f"min={reward.min():+.4f} max={reward.max():+.4f} mean={reward.mean():+.4f} std={reward.std():.4f}")
        # Top-5 by |reward|
        order = np.argsort(-np.abs(reward))
        print(f"  largest |reward| moves (these dominate MCTS Q on the 1st sim):")
        for j in order[:6]:
            a = int(legal[j]); mv = _action_to_move(a, board)
            san = board.san(mv) if (mv and mv in board.legal_moves) else f"a{a}"
            # mover-POV Q on first visit = reward - discount*child_value
            q1 = reward[j] - cfg.discount * valsc[j]
            print(f"    {san:8s} reward={reward[j]:+.4f}  child_v={valsc[j]:+.4f}  Q1={q1:+.4f}")
        # What is the TRUE reward target for each? 0 unless the move is terminal.
        n_terminal = 0
        for a in legal:
            mv = _action_to_move(int(a), board)
            if mv is None or mv not in board.legal_moves:
                continue
            nb = board.copy(); nb.push(mv)
            if nb.is_game_over(claim_draw=False) or nb.is_repetition(3):
                n_terminal += 1
        print(f"  TRUE reward target: 0 for all non-terminal moves; {n_terminal} legal moves are terminal (reward !=0 allowed)")


if __name__ == "__main__":
    main()
