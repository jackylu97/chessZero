"""IN-DISTRIBUTION decisive test: on REAL buffer positions, does the reward head's
hallucinated nonzero reward on non-terminal moves drive MCTS selection?

For each sampled real position:
  - run numpy MCTS (200 sims), record the picked (argmax-visit) move
  - compute, for the ROOT's children, raw_q WITH reward and WITHOUT reward
    (value-only Q = -discount*child_value), and see if the argmax changes
  - record reward-head stats over the legal moves (should be ~0; true reward
    is 0 for every non-terminal move)
Distinguishes: 'value head can't tell' (value-only argmax also picks a bad/random
move) vs 'reward hallucination flips the pick' (value-only argmax differs and is
more sensible) vs both.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F, chess
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import ReplayBuffer
from src.mcts.mcts import MCTS
torch.serialization.add_safe_globals([MuZeroConfig])
from scripts.eval_checkpoint_health import build_network

CKPT = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"
BUF = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf"


@torch.no_grad()
def main():
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = "cpu"
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    net = build_network(ckpt, game, cfg, "cpu")
    rb = ReplayBuffer(cfg); rb.load(BUF, game=game)
    games = [g for g in rb.buffer if not g.external_values and len(g) >= 8]
    print(f"Loaded {len(games)} self-play games")

    rng = np.random.default_rng(1)
    mcts = MCTS(net, game, cfg, "cpu")

    n_reward_drives = 0   # value-only argmax != reward-incl argmax
    n_total = 0
    reward_max_acc = []
    reward_mean_acc = []
    root_value_oob = 0   # |root.value| > 1 (impossible for pure WDL)
    picked_reward_acc = []  # reward of the actually-picked child
    n_picked_is_max_reward = 0

    sample = rng.choice(len(games), size=40, replace=False)
    for gi in sample:
        gh = games[int(gi)]
        si = int(rng.integers(2, len(gh) - 1))
        obs = gh._stack_history(si, cfg.history_frames)
        # Need legal actions at this ply. Reconstruct the board by replaying.
        board = chess.Board()
        for a in gh.actions[:si]:
            mv = _action_to_move(int(a), board)
            if mv is None or mv not in board.legal_moves:
                board = None; break
            board.push(mv)
        if board is None or board.is_game_over():
            continue
        legal = game.legal_actions(_state(game, board))
        root = mcts.run(obs, legal, add_noise=False)
        visits = root.child_visits; rewards = root.child_rewards
        vsums = root.child_value_sums
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = np.where(visits > 0, vsums / np.maximum(visits, 1.0), 0.0)
        raw_q_incl = rewards - cfg.discount * cv     # actual Q
        raw_q_val = -cfg.discount * cv               # value-only Q (reward dropped)

        # Argmax by VISITS = what MCTS actually picks
        picked = int(np.argmax(visits))
        # Best-by-Q among visited (incl reward) vs value-only
        visited = visits > 0
        if visited.sum() < 2:
            continue
        qincl = np.where(visited, raw_q_incl, -np.inf)
        qval = np.where(visited, raw_q_val, -np.inf)
        best_incl = int(np.argmax(qincl))
        best_val = int(np.argmax(qval))
        n_total += 1
        if best_incl != best_val:
            n_reward_drives += 1
        reward_max_acc.append(float(np.abs(rewards[visited]).max()))
        reward_mean_acc.append(float(np.abs(rewards[visited]).mean()))
        picked_reward_acc.append(float(abs(rewards[picked])))
        if abs(rewards[picked]) >= np.abs(rewards[visited]).max() - 1e-9:
            n_picked_is_max_reward += 1
        if abs(root.value) > 1.0:
            root_value_oob += 1

    print(f"\nIN-DISTRIBUTION REWARD-HALLUCINATION TEST (n={n_total} positions, 200 sims each)")
    print(f"  reward head |reward| over visited children: "
          f"mean-of-max={np.mean(reward_max_acc):.4f}  mean-of-mean={np.mean(reward_mean_acc):.4f}")
    print(f"    (TRUE reward = 0 for every non-terminal move; any nonzero is hallucination)")
    print(f"  Q-argmax changes when reward dropped (reward flips the pick): "
          f"{n_reward_drives}/{n_total} = {n_reward_drives/max(n_total,1):.1%}")
    print(f"  MCTS-picked (max-visit) child also has the MAX hallucinated reward: "
          f"{n_picked_is_max_reward}/{n_total} = {n_picked_is_max_reward/max(n_total,1):.1%}")
    print(f"  |root.value|>1 (impossible for pure WDL => reward inflated): "
          f"{root_value_oob}/{n_total}")
    print(f"  mean |reward| of the PICKED child: {np.mean(picked_reward_acc):.4f}")


def _state(game, board):
    st = game.reset(); st.board = board.copy()
    st.current_player = 1 if board.turn == chess.WHITE else -1
    return st


if __name__ == "__main__":
    main()
