"""Compare dynamics-predicted next-state vs representation-of-actual-next-state.

For each (position, move) pair:
  h_root      = representation(obs_root)
  h_dyn_next  = dynamics(h_root, action)         # what the model PREDICTS comes next
  h_repr_next = representation(obs_after_move)   # what the model SEES at the actual next state

Then compare downstream policy / value evaluations on h_dyn_next vs h_repr_next.
A healthy world model has these close. A collapsed world model has h_dyn_next stuck
near h_root (or at a centroid) regardless of action.

Run: .venv/bin/python scripts/probe_dynamics_vs_repr.py --checkpoint <path>
"""
import argparse
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
import torch
import torch.nn.functional as F

from src.config import get_config
from src.games.chess import ChessGame, _move_to_action
from src.games.base import GameState
from src.model.muzero_net import MuZeroNetwork


# Mid-game positions — chosen for variety in material/structure/tension.
MIDGAME_FENS = [
    # tactical Italian middlegame, white to move
    "r1bq1rk1/ppp2ppp/2n1pn2/2bp4/2BP4/2N1PN2/PPP1QPPP/R1B2RK1 w - - 0 8",
    # KID-ish, sharp pawn tension
    "r1bq1rk1/pp2nppp/2nppb2/8/2PNP3/2N1B3/PP2BPPP/R2Q1RK1 w - - 4 9",
    # Caro-Kann endgame approach, more locked
    "r2q1rk1/pp1bppbp/2np1np1/8/3NPP2/2N1B3/PPPQB1PP/R4RK1 w - - 0 11",
    # Sicilian dragon-like, opposite-side castling
    "r2q1rk1/pp1bppbp/2np1np1/8/3NP3/2N1BP2/PPPQB1PP/2KR3R w - - 0 11",
    # heavy material imbalance: white up an exchange
    "r3k2r/ppp2ppp/2n5/3p1b2/3P4/2N1B3/PPP2PPP/2KR3R w kq - 0 12",
    # endgame transition: queens off, rooks + minors
    "r4rk1/pp3ppp/2n1pn2/3p4/3P4/2N1PN2/PP3PPP/R4RK1 w - - 0 14",
    # pawn endgame study
    "8/p1p2pkp/1p1p2p1/3P4/2P5/1P3KP1/P4P1P/8 w - - 0 30",
    # opposite-coloured bishops
    "8/4kpp1/4p2p/4P3/3PB3/5b2/r5KP/8 w - - 0 35",
    # white pinned, complex middlegame
    "r2q1rk1/pp2bppp/2nppn2/8/3NP3/2N1BP2/PPPQB1PP/R4RK1 b - - 1 11",
    # closed center, manoeuvring
    "r1bq1rk1/pp1nbppp/2p1pn2/3p4/2PPP3/2N1BN2/PP3PPP/R2QKB1R w KQ - 0 8",
]


def _build_obs_with_history(game: ChessGame, prior_states: list, current_state: GameState, n_frames: int) -> torch.Tensor:
    """Stack the last n_frames observations, newest first (matches GameHistory._stack_history).
    Zero-pads missing earlier frames."""
    cur_obs = game.to_tensor(current_state)         # (C, H, W)
    if n_frames <= 1:
        return cur_obs
    frames = [cur_obs]
    for s in reversed(prior_states):
        if len(frames) >= n_frames:
            break
        frames.append(game.to_tensor(s))
    while len(frames) < n_frames:
        frames.append(torch.zeros_like(cur_obs))
    # newest first along channel dim
    return torch.cat(frames, dim=0)


def _wdl_softmax(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)


def _value_from_wdl(wdl: torch.Tensor) -> float:
    """V = P(W) - P(L) (ignores draw_score for clean comparison)."""
    return float(wdl[0] - wdl[2])


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """KL(p || q) on flat distributions."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return float((p * (p.log() - q.log())).sum())


def _cos_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--moves-per-position", type=int, default=4,
                    help="How many candidate moves to evaluate per position. Top-policy + a couple random.")
    args = ap.parse_args()

    config = get_config("chess")
    game = ChessGame()
    n_frames = config.history_frames
    print(f"Using history_frames={n_frames}, hidden_planes={config.hidden_planes}, "
          f"action_embed_dim={config.action_embed_dim}, value_head_type={config.value_head_type}")

    network = MuZeroNetwork(
        observation_channels=game.num_planes * n_frames,
        action_space_size=game.action_space_size,
        hidden_planes=config.hidden_planes,
        num_blocks=config.num_residual_blocks,
        latent_h=config.latent_h, latent_w=config.latent_w,
        input_h=game.board_size[0], input_w=game.board_size[1],
        fc_hidden=config.fc_hidden,
        value_support_size=config.value_support_size,
        reward_support_size=config.reward_support_size,
        action_embed_dim=config.action_embed_dim,
        use_consistency_loss=config.use_consistency_loss,
        proj_hid=config.proj_hid, proj_out=config.proj_out,
        pred_hid=config.pred_hid, pred_out=config.pred_out,
        use_scalar_transform=config.use_scalar_transform,
        value_target_scale=config.value_target_scale,
        value_head_type=config.value_head_type,
        draw_score=config.draw_score,
    ).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    network.load_state_dict(ckpt["model_state_dict"])
    network.eval()
    print(f"Loaded {args.checkpoint} (step {ckpt.get('step', '?')})")

    # Aggregators
    rows = []  # one row per (position, move)
    per_pos_dyn_spread = []  # how varied is h_dyn_next across moves at the same pos

    rng = np.random.RandomState(7)

    with torch.no_grad():
        for fi, fen in enumerate(MIDGAME_FENS):
            board = chess.Board(fen)
            stm = chess.WHITE if board.turn == chess.WHITE else chess.BLACK
            state = GameState(board=board, current_player=1 if stm == chess.WHITE else -1)

            # Root obs (no real game history → zero-padded for n_frames-1 slots)
            obs_root = _build_obs_with_history(game, prior_states=[], current_state=state, n_frames=n_frames).to(args.device)
            h_root, root_policy_logits, root_v_scalar = network.initial_inference(obs_root.unsqueeze(0))
            # Get root WDL too
            _, _, root_value_logits = network.initial_inference_logits(obs_root.unsqueeze(0)) \
                if hasattr(network, "initial_inference_logits") else (None, None, None)
            root_wdl = _wdl_softmax(root_value_logits[0]) if root_value_logits is not None else None

            legal = list(board.legal_moves)
            if not legal:
                continue
            # rank moves by policy prior; pick top-N + a couple of random ones
            policy_probs = F.softmax(root_policy_logits[0], dim=-1).cpu()
            legal_actions = [_move_to_action(m, stm) for m in legal]
            scored = sorted(zip(legal_actions, legal), key=lambda x: -float(policy_probs[x[0]]))
            chosen = scored[: max(1, args.moves_per_position - 1)]
            if len(scored) > len(chosen):
                # add one random non-top move for variety
                pool = [t for t in scored[len(chosen):]]
                chosen.append(pool[rng.randint(0, len(pool))])

            dyn_latents_for_this_pos = []
            print(f"\n=== Pos {fi}: {fen}")
            print(f"  root V={float(root_v_scalar.item()):+.3f}  "
                  f"WDL=({root_wdl[0]:.2f}, {root_wdl[1]:.2f}, {root_wdl[2]:.2f})  "
                  f"legal={len(legal)}")

            for action, move in chosen:
                # 1) Dynamics-predicted next latent
                action_t = torch.tensor([action], dtype=torch.int64, device=args.device)
                h_dyn_next, reward_pred, dyn_policy_logits, dyn_v_scalar = network.recurrent_inference(h_root, action_t)
                _, dyn_reward_logits = network.dynamics(h_root, action_t)  # for completeness
                # 2) Apply the move on the actual board, get representation of resulting obs
                next_state, _, _ = game.step(state, action)
                obs_next = _build_obs_with_history(game, prior_states=[state], current_state=next_state, n_frames=n_frames).to(args.device)
                h_repr_next, repr_policy_logits, repr_v_scalar = network.initial_inference(obs_next.unsqueeze(0))

                # 3) Latent comparison
                cos_dyn_repr = _cos_flat(h_dyn_next, h_repr_next)
                cos_root_repr = _cos_flat(h_root, h_repr_next)        # baseline: how different is next from current
                cos_root_dyn = _cos_flat(h_root, h_dyn_next)          # how different is dynamics output from input
                # 4) Magnitudes / spread
                dyn_norm = float(h_dyn_next.norm().item())
                repr_norm = float(h_repr_next.norm().item())

                # 5) Downstream evaluations: run prediction head on each
                _, dyn_policy_logits_p, dyn_v_logits = network.initial_inference_logits(obs_root.unsqueeze(0))  # unused
                # call prediction directly on h_dyn_next and h_repr_next
                dyn_pol_logits, dyn_val_logits = network.prediction(h_dyn_next)
                rep_pol_logits, rep_val_logits = network.prediction(h_repr_next)

                dyn_pol = F.softmax(dyn_pol_logits[0], dim=-1)
                rep_pol = F.softmax(rep_pol_logits[0], dim=-1)
                dyn_wdl = _wdl_softmax(dyn_val_logits[0])
                rep_wdl = _wdl_softmax(rep_val_logits[0])

                # restrict policy comparison to moves legal in the next position (the
                # network never sees the full 4672-dim simplex meaningfully; KL on the
                # legal subset is more interpretable).
                legal_next = [_move_to_action(m, next_state.board.turn) for m in next_state.board.legal_moves] \
                    if not next_state.done else []

                if legal_next:
                    legal_idx = torch.tensor(legal_next, device=args.device)
                    dyn_pol_legal = dyn_pol[legal_idx]
                    rep_pol_legal = rep_pol[legal_idx]
                    dyn_pol_legal = dyn_pol_legal / (dyn_pol_legal.sum() + 1e-12)
                    rep_pol_legal = rep_pol_legal / (rep_pol_legal.sum() + 1e-12)
                    kl_dyn_rep = _kl(dyn_pol_legal, rep_pol_legal)
                    pol_top1_match = int(dyn_pol_legal.argmax().item() == rep_pol_legal.argmax().item())
                else:
                    kl_dyn_rep = float("nan")
                    pol_top1_match = -1

                v_dyn = _value_from_wdl(dyn_wdl)
                v_rep = _value_from_wdl(rep_wdl)
                pw_diff = float(dyn_wdl[0] - rep_wdl[0])
                pd_diff = float(dyn_wdl[1] - rep_wdl[1])
                pl_diff = float(dyn_wdl[2] - rep_wdl[2])

                rows.append({
                    "fen": fen,
                    "move": board.san(move),
                    "cos_dyn_repr": cos_dyn_repr,
                    "cos_root_repr": cos_root_repr,
                    "cos_root_dyn": cos_root_dyn,
                    "dyn_norm": dyn_norm, "repr_norm": repr_norm,
                    "v_dyn": v_dyn, "v_rep": v_rep,
                    "v_diff": v_dyn - v_rep,
                    "pw_diff": pw_diff, "pd_diff": pd_diff, "pl_diff": pl_diff,
                    "kl_legal": kl_dyn_rep,
                    "pol_top1_match": pol_top1_match,
                })
                dyn_latents_for_this_pos.append(h_dyn_next.flatten())

                print(f"  {board.san(move):>6}  "
                      f"cos(dyn,repr)={cos_dyn_repr:+.4f}  "
                      f"cos(root,repr)={cos_root_repr:+.4f}  "
                      f"cos(root,dyn)={cos_root_dyn:+.4f}  "
                      f"V_dyn={v_dyn:+.3f} V_rep={v_rep:+.3f} (Δ={v_dyn - v_rep:+.3f})  "
                      f"KL_legal={kl_dyn_rep:.4f}")

            # how varied is the dynamics output across different moves at the same position?
            if len(dyn_latents_for_this_pos) >= 2:
                dyn_stack = torch.stack(dyn_latents_for_this_pos)
                cos_mat = F.cosine_similarity(dyn_stack.unsqueeze(0), dyn_stack.unsqueeze(1), dim=-1)
                off = cos_mat[~torch.eye(len(dyn_stack), dtype=torch.bool, device=cos_mat.device)]
                per_pos_dyn_spread.append(float(off.mean().item()))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (across all (position, move) pairs)")
    print("=" * 70)
    if rows:
        def col(name): return [r[name] for r in rows]
        def stat(name, fmt="+.4f"):
            xs = np.array(col(name), dtype=np.float64)
            xs = xs[np.isfinite(xs)]
            if len(xs) == 0:
                print(f"  {name}: (no finite values)")
                return
            print(f"  {name:>20}  mean={xs.mean():{fmt}}  std={xs.std():{fmt}}  "
                  f"min={xs.min():{fmt}}  max={xs.max():{fmt}}  n={len(xs)}")
        stat("cos_dyn_repr")
        stat("cos_root_repr")
        stat("cos_root_dyn")
        stat("v_diff")
        stat("pw_diff")
        stat("pd_diff")
        stat("pl_diff")
        stat("kl_legal")
        if per_pos_dyn_spread:
            spread = np.array(per_pos_dyn_spread)
            print(f"\n  per-position dyn-output cos sim across different moves:")
            print(f"    mean across positions = {spread.mean():+.4f}  "
                  f"(higher = dynamics output ignores the action; ~1.0 = action-blind)")

        top1 = np.array([r["pol_top1_match"] for r in rows])
        valid = top1 >= 0
        if valid.any():
            print(f"  policy top-1 agreement (dyn vs repr, on legal moves of next pos): "
                  f"{top1[valid].mean()*100:.1f}% ({int(top1[valid].sum())}/{int(valid.sum())})")

    print("\nInterpretation guide:")
    print("  cos(dyn, repr)  — agreement between predicted and actual next-state latent.")
    print("                    Healthy: > ~0.95 (close prediction)")
    print("                    Collapsed-but-trivial: very high (both vectors are constants)")
    print("                    Action-aware but wrong: moderate (~0.6-0.9)")
    print("  cos(root, repr) — natural drift between consecutive states.")
    print("                    Should be < cos(dyn, repr) if dynamics is doing useful work.")
    print("  cos(root, dyn)  — dynamics output similarity to its INPUT. If ≈ 1, dynamics ≈ identity.")
    print("  per-pos dyn spread — across different moves from same root, how varied are predictions?")
    print("                    ~1.0 means action-blind (output ignores action).")
    print("  v_diff / pw_diff — dyn-evaluated value vs repr-evaluated value of the same next state.")
    print("                    If small (≈0), dynamics agrees with what the network would say.")
    print("  KL_legal — KL between policies dyn vs repr produces, restricted to legal next moves.")


if __name__ == "__main__":
    main()
