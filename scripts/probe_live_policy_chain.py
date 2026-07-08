"""Live end-to-end probe of the GPU-resident self-play policy chain.

Runs a REAL self-play batch with a trained checkpoint, then for every stored
ply asserts the played move, the policy-target index, the legal mask, and the
conv-policy logits are mutually consistent.

Failures here = a mechanistic bug that the static encoding tests can't catch
(obs/policy desync, wrong-frame stacking, logit/target index mismatch).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import get_config
from src.games.chess import ChessGame, _action_to_move, _move_to_action
from src.training.self_play import play_games_parallel_gpu_resident
from src.training.replay_buffer import stack_with_history
from scripts.eval_checkpoint_health import build_network


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else \
        "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config("chess_small")
    cfg.device = device
    cfg.use_gpu_chess = True
    cfg.use_tensor_mcts = True
    cfg.use_gpu_resident_self_play = True
    cfg.num_simulations = int(getattr(cfg, "num_simulations", 50))
    # Keep the batch small + games short for a fast probe.
    cfg.max_plies = 40
    n_games = 8

    print(f"ckpt={ckpt_path} device={device} sims={cfg.num_simulations} "
          f"history_frames={cfg.history_frames} policy_head={cfg.policy_head_type}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    game = ChessGame()
    net = build_network(ckpt, game, cfg, device)
    net.eval()

    torch.manual_seed(0)
    histories = play_games_parallel_gpu_resident(
        net, cfg, n_games, device=device, training_step=0
    )
    print(f"got {len(histories)} histories, lengths={[len(h) for h in histories]}")

    A = game.action_space_size
    n_frames = int(cfg.history_frames)

    # Counters
    n_ply = 0
    played_illegal = 0          # played action not in stored legal set
    played_not_in_policy = 0    # played action has ~0 policy-target mass
    policy_mass_on_illegal = 0  # policy target puts mass on a non-legal action
    policy_argmax_illegal = 0   # policy-target argmax not legal
    legal_mask_desync = 0       # legal set recomputed from obs != stored legal set
    decode_fail = 0             # played action id won't decode to a legal move on a fresh board
    logit_illegal_beats = 0     # net logit on best illegal > masked-legal max (info only)

    # Rebuild the python-chess board per game by replaying decoded actions, so we
    # can independently recompute the legal set from the ground-truth position and
    # confirm the stored legal mask + played action agree with python-chess.
    for h in histories:
        board = game.reset().board  # fresh start position
        prior_obs = []
        for t in range(len(h.actions)):
            n_ply += 1
            played = int(h.actions[t])
            policy = np.asarray(h.policies[t], dtype=np.float64)
            legal_stored = set(h.legal_actions_list[t])
            obs_t = h.observations[t]  # single-frame (C,H,W)

            # --- independent ground truth from python-chess ---
            legal_truth = set(_move_to_action(m, board.turn) for m in board.legal_moves)

            # 1. stored legal mask vs python-chess truth
            if legal_stored != legal_truth:
                legal_mask_desync += 1

            # 2. played action legal?
            if played not in legal_stored:
                played_illegal += 1

            # 3. played action has policy mass?
            if policy[played] <= 1e-6:
                played_not_in_policy += 1

            # 4. policy mass only on legal actions?
            nz = set(np.nonzero(policy > 1e-6)[0].tolist())
            if not nz.issubset(legal_stored):
                policy_mass_on_illegal += 1

            # 5. policy argmax legal?
            if policy.sum() > 0 and int(policy.argmax()) not in legal_stored:
                policy_argmax_illegal += 1

            # 6. played action decodes to the SAME move python-chess will accept
            mv = _action_to_move(played, board)
            if mv is None or mv not in board.legal_moves:
                decode_fail += 1
                board = None
                break

            # 7. conv-policy logits: re-run the net on the reconstructed stacked
            #    obs and check legal-masked logits are finite and that the played
            #    move's logit participates (not -inf after masking).
            stacked = stack_with_history(obs_t, prior_obs, n_frames).unsqueeze(0).to(device)
            with torch.no_grad():
                hidden, policy_logits, _v = net.initial_inference(stacked)
            pl = policy_logits[0].float().cpu().numpy()
            legal_idx = np.fromiter(legal_stored, dtype=np.int64)
            masked = np.full(A, -1e30); masked[legal_idx] = pl[legal_idx]
            if not np.isfinite(pl[played]):
                logit_illegal_beats += 1  # reuse counter name loosely; logit nan/inf
            best_illegal = pl.copy(); best_illegal[legal_idx] = -1e30
            # info only: does the unmasked argmax land on an illegal move?
            # (expected sometimes; just track magnitude)

            # advance ground-truth board
            board.push(mv)
            prior_obs.append(obs_t)

    print("\n=== LIVE CHAIN CONSISTENCY (n_ply=%d) ===" % n_ply)
    print(f"  legal_mask_desync (stored != python-chess) : {legal_mask_desync}")
    print(f"  played_illegal (not in legal set)          : {played_illegal}")
    print(f"  played_not_in_policy (0 target mass)       : {played_not_in_policy}")
    print(f"  policy_mass_on_illegal                     : {policy_mass_on_illegal}")
    print(f"  policy_argmax_illegal                      : {policy_argmax_illegal}")
    print(f"  decode_fail (action->move not legal)       : {decode_fail}")
    print(f"  played-move logit nan/inf after net rerun  : {logit_illegal_beats}")


if __name__ == "__main__":
    main()
