"""Ground-truth target-alignment trace.

Independently replays a real buffer game's action sequence through python-chess
to get the TRUE board / side-to-move / legal-move set at every ply index, then
calls make_target and verifies index-by-index that:

  obs[i]     <-> board at ply (state_index+i)   (newest frame identity)
  action[i]  <-> the move played from that board (legal & matches stored)
  reward[i]  <-> self.rewards[idx-1]             (DeepMind into-state convention)
  value[i]   <-> STM-at-idx POV                  (W,D,L sign correctness)
  policy[i]  <-> argmax is LEGAL at board[idx] AND matches the played move

Runs over ALL game classes (rep-draw = 99% of self-play, decisive, warmstart)
and aggregates pass/fail counts across MANY random state_index samples.
CPU only.
"""
import sys
from collections import Counter, defaultdict
import numpy as np
import torch
import chess

sys.path.insert(0, "/workspace/chessZero")
from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import ReplayBuffer

np.random.seed(0)

BUF = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf"
N_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 300
SAMPLES_PER_GAME = 4

cfg = get_config("chess_small")
cfg.device = "cpu"
A = ChessGame().action_space_size
K = cfg.num_unroll_steps
HF = cfg.history_frames
NP = ChessGame().num_planes
print(f"buf={BUF}")
print(f"K={K} td_steps={cfg.td_steps} discount={cfg.discount} vhead={cfg.value_head_type} "
      f"HF={HF} A={A} num_planes={NP}")
print(f"selfplay_q_ratio={cfg.selfplay_q_ratio} rep_penalty={cfg.repetition_penalty} "
      f"rep_decay={cfg.repetition_penalty_decay}")


def classify(gh):
    if gh.external_values:
        return "warmstart"
    if abs(gh.game_outcome) >= 0.5:
        return "decisive"
    if gh.draw_by_repetition:
        return "rep-draw"
    if gh.draw_by_no_progress:
        return "noprog-draw"
    return "other-draw"


_buf = ReplayBuffer(max_size=10**9)
_buf.load(BUF, game=ChessGame())
games = _buf.buffer[:N_GAMES]
print(f"\nloaded {len(games)} games; classes: {Counter(classify(g) for g in games)}")


def ground_truth(gh):
    """Return (fens, stms, legal_action_sets, played_actions, replay_ok)."""
    board = chess.Board() if not gh.start_fen else chess.Board(gh.start_fen)
    fens = [board.fen()]
    stms = [board.turn]  # True=white
    legal_sets = [set(_safe_action(m, board) for m in board.legal_moves)]
    played = []
    ok = True
    for a in gh.actions:
        mv = _action_to_move(a, board)
        if mv is None or mv not in board.legal_moves:
            ok = False
            break
        played.append(a)
        board.push(mv)
        fens.append(board.fen())
        stms.append(board.turn)
        legal_sets.append(set(_safe_action(m, board) for m in board.legal_moves))
    return fens, stms, legal_sets, played, ok


def _safe_action(mv, board):
    from src.games.chess import _move_to_action
    try:
        return _move_to_action(mv, board.turn)
    except Exception:
        return -1


# Aggregate counters.
agg = defaultdict(lambda: Counter())
fail_examples = defaultdict(list)
replay_fail = 0


def check_game(gh, cls):
    global replay_fail
    fens, stms, legal_sets, played, ok = ground_truth(gh)
    if not ok:
        replay_fail += 1
        return
    N = len(gh.actions)
    if N < 2:
        return
    for _ in range(SAMPLES_PER_GAME):
        si = np.random.randint(0, N)
        obs_list, obs_mask, actions, values, rewards, policies = gh.make_target(
            si, K, cfg.td_steps, cfg.discount, A,
            value_head_type=cfg.value_head_type, history_frames=HF,
            eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
            q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
            selfplay_q_ratio=cfg.selfplay_q_ratio,
            repetition_penalty=cfg.repetition_penalty,
            repetition_penalty_window=int(cfg.repetition_penalty_window),
            repetition_penalty_decay=float(cfg.repetition_penalty_decay),
        )
        for i in range(K + 1):
            idx = si + i
            in_game = idx < N
            c = agg[cls]

            # ---- ACTION alignment (only K entries; trainer applies actions[k]) ----
            if i < K:
                want = gh.actions[idx] if idx < len(gh.actions) else 0
                if actions[i] == want:
                    c["action_ok"] += 1
                else:
                    c["action_FAIL"] += 1
                    if len(fail_examples["action"]) < 5:
                        fail_examples["action"].append((cls, si, i, actions[i], want))

            # ---- REWARD into-state convention: reward[i] == self.rewards[idx-1] ----
            prev = idx - 1
            exp_rew = gh.rewards[prev] if 0 <= prev < len(gh.rewards) else 0.0
            if abs(float(rewards[i]) - exp_rew) < 1e-6:
                c["reward_ok"] += 1
            else:
                c["reward_FAIL"] += 1
                if len(fail_examples["reward"]) < 5:
                    fail_examples["reward"].append((cls, si, i, float(rewards[i]), exp_rew))

            # ---- OBS newest-frame identity vs ground-truth board index ----
            cap = min(idx, len(gh) - 1)
            newest = obs_list[i][:NP]
            ref = gh.observations[cap][:NP]
            if torch.allclose(newest, ref):
                c["obs_ok"] += 1
            else:
                c["obs_FAIL"] += 1

            # ---- VALUE STM parity (decisive only — sign is meaningful) ----
            if in_game and cls == "decisive":
                stm_is_white = stms[idx]
                stm_won = (gh.game_outcome > 0) == stm_is_white
                v = values[i]
                sval = float(v[0] - v[2]) if isinstance(v, np.ndarray) else float(v)
                if abs(sval) > 1e-3:
                    if (sval > 0) == stm_won:
                        c["value_parity_ok"] += 1
                    else:
                        c["value_parity_FAIL"] += 1
                        if len(fail_examples["value"]) < 5:
                            fail_examples["value"].append(
                                (cls, si, i, sval, stm_won, stm_is_white))

            # ---- POLICY: argmax legal at board[idx] AND matches played move ----
            if in_game:
                pol = np.asarray(policies[i])
                if pol.sum() > 0:
                    amax = int(np.argmax(pol))
                    legal = amax in legal_sets[idx]
                    if legal:
                        c["policy_legal_ok"] += 1
                    else:
                        c["policy_legal_FAIL"] += 1
                        if len(fail_examples["policy_legal"]) < 5:
                            fail_examples["policy_legal"].append((cls, si, i, amax))
                    # Stored policy argmax vs played action at this ply.
                    if idx < len(played):
                        if amax == played[idx]:
                            c["policy_argmax_eq_played"] += 1
                        else:
                            c["policy_argmax_ne_played"] += 1
                else:
                    # zero policy should only happen at terminal idx == N
                    if idx == N:
                        c["policy_zero_terminal_ok"] += 1
                    else:
                        c["policy_zero_INGAME_FAIL"] += 1


for gh in games:
    check_game(gh, classify(gh))

print(f"\nreplay_fail games (action decode mismatch vs python-chess): {replay_fail}")
print("\n=== AGGREGATE per class ===")
for cls in sorted(agg):
    print(f"\n[{cls}]")
    for k in sorted(agg[cls]):
        print(f"   {k:32s} {agg[cls][k]}")

if any(fail_examples.values()):
    print("\n=== FAIL EXAMPLES ===")
    for k, v in fail_examples.items():
        if v:
            print(f"  {k}: {v}")
else:
    print("\nNo alignment failures recorded.")
