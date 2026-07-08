#!/usr/bin/env python3
"""Custom probe: representation + dynamics degeneracy across cold2_pc checkpoints.

Uses chess_small config (matches the checkpoints). Reports:

REPRESENTATION (on in-distribution Stockfish-pool positions, real history):
  - cross-position cosine, effective rank (entropy + participation ratio)
  - per-dim latent std, fraction of dead dims
  - linear-probe R2(eval), R2(outcome), decisive sign-acc

DYNAMICS (recurrent rollout vs representation-of-true-next):
  - cross-action cosine (action-blind detector)
  - per-position dyn-output spread across moves
  - cos(dyn_next, repr_next): consistency
  - cos(root, dyn_next) vs cos(root, repr_next): is dyn ~ identity?
  - inverse-action recovery accuracy
  - K-step rollout: does latent drift to a fixed point? (consecutive-step cosine,
    norm of step delta) — fixed-point collapse detector
  - value spread across actions (within-position Q signal at the latent level)

Run:
  .venv/bin/python scripts/probe_repr_dyn_degeneracy.py \
      --checkpoint checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import chess

from src.config import MuZeroConfig, get_config
from src.games.chess import ChessGame, _move_to_action, _action_to_move
from src.model.muzero_net import MuZeroNetwork
from src.training.replay_buffer import _iter_shard_games, stack_with_history


def build_network(ckpt, game, cfg, device):
    sd = ckpt["model_state_dict"]
    has_conv_policy = any(".policy_head.mix." in k or ".policy_head.proj." in k for k in sd)
    has_moves_left = any(k.startswith("moves_left_head.") for k in sd)
    has_consistency = any(k.startswith("projection.") for k in sd)
    has_inverse = any(k.startswith("inverse_dynamics_head.") for k in sd)
    net = MuZeroNetwork(
        observation_channels=game.num_planes * cfg.history_frames,
        action_space_size=game.action_space_size, hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w,
        input_h=8, input_w=8, fc_hidden=cfg.fc_hidden, value_support_size=cfg.value_support_size,
        reward_support_size=cfg.reward_support_size, action_embed_dim=cfg.action_embed_dim,
        use_consistency_loss=has_consistency, proj_hid=cfg.proj_hid, proj_out=cfg.proj_out,
        pred_hid=cfg.pred_hid, pred_out=cfg.pred_out, use_scalar_transform=cfg.use_scalar_transform,
        value_target_scale=cfg.value_target_scale, value_head_type=cfg.value_head_type,
        draw_score=cfg.draw_score, value_head_init_std=getattr(cfg, "value_head_init_std", 0.0),
        use_inverse_dynamics_loss=has_inverse,
        inverse_dynamics_hidden=getattr(cfg, "inverse_dynamics_hidden", 256),
        policy_head_type="conv" if has_conv_policy else "flat",
        use_moves_left=has_moves_left,
        moves_left_support_size=getattr(cfg, "moves_left_support_size", 10),
    ).to(device)
    net.load_state_dict(sd)
    net.eval()
    return net, dict(conv_policy=has_conv_policy, moves_left=has_moves_left,
                     consistency=has_consistency, inverse=has_inverse)


def effective_rank(H):
    """Entropy effective rank + participation ratio of latent covariance."""
    Hc = H - H.mean(0, keepdims=True)
    # SVD-based singular values
    s = np.linalg.svd(Hc, compute_uv=False)
    ev = s ** 2
    ev = ev[ev > 0]
    if len(ev) == 0:
        return 0.0, 0.0
    p = ev / ev.sum()
    ent_rank = float(np.exp(-(p * np.log(p)).sum()))
    part_ratio = float((ev.sum() ** 2) / (ev ** 2).sum())
    return ent_rank, part_ratio


def mean_crosspos_cosine(H, seed=0, max_pairs=20000):
    rng = np.random.default_rng(seed)
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    n = len(Hn)
    npair = min(max_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=npair)
    j = rng.integers(0, n, size=npair)
    m = i != j
    i, j = i[m], j[m]
    return float((Hn[i] * Hn[j]).sum(1).mean())


def ridge_r2(H, y, seed=0, alphas=(100.0, 1000.0, 10000.0)):
    """Ridge probe with alpha selected by best held-out R2 (avoids the
    under-regularized 4096-dim overfit that produced spurious negative R2)."""
    rng = np.random.default_rng(seed)
    n = len(H)
    idx = rng.permutation(n)
    ntr = int(n * 0.8)
    tr, te = idx[:ntr], idx[ntr:]
    Xtr, Xte = H[tr], H[te]
    ytr, yte = y[tr], y[te]
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ym = ytr.mean()
    best_r2, best = -1e9, None
    for alpha in alphas:
        A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ (ytr - ym))
        pred = Xte @ w + ym
        ss_res = ((yte - pred) ** 2).sum()
        ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-12
        r2 = float(1 - ss_res / ss_tot)
        if r2 > best_r2:
            best_r2, best = r2, (pred, yte)
    return best_r2, best


def sample_positions(shards_dir, game, hf, n_games, n_per_game, seed):
    rng = np.random.default_rng(seed)
    paths = sorted(glob.glob(os.path.join(shards_dir, "**", "*.pkl"), recursive=True))
    rng.shuffle(paths)
    obs_list, ev_list, out_list = [], [], []
    games_used = 0
    for p in paths:
        if games_used >= n_games:
            break
        try:
            games = list(_iter_shard_games(p, game=game))
        except Exception:
            continue
        rng.shuffle(games)
        for gh in games:
            if games_used >= n_games:
                break
            n_ev = len(gh.external_values)
            if n_ev == 0:
                continue
            games_used += 1
            k = min(n_per_game, n_ev)
            plies = rng.choice(n_ev, size=k, replace=False)
            for ply in plies:
                stack = gh._stack_history(int(ply), hf)
                obs_list.append(stack.numpy())
                ev = float(gh.external_values[int(ply)])
                stm_white = (int(ply) % 2 == 0)
                out = float(gh.game_outcome) * (1.0 if stm_white else -1.0)
                ev_list.append(ev)
                out_list.append(out)
    return (np.asarray(obs_list, dtype=np.float32),
            np.asarray(ev_list, dtype=np.float32),
            np.asarray(out_list, dtype=np.float32))


# Diverse FENs for dynamics probing (real legal moves, varied phase)
DYN_FENS = [
    "r1bq1rk1/ppp2ppp/2n1pn2/2bp4/2BP4/2N1PN2/PPP1QPPP/R1B2RK1 w - - 0 8",
    "r1bq1rk1/pp2nppp/2nppb2/8/2PNP3/2N1B3/PP2BPPP/R2Q1RK1 w - - 4 9",
    "r3k2r/ppp2ppp/2n5/3p1b2/3P4/2N1B3/PPP2PPP/2KR3R w kq - 0 12",
    "r4rk1/pp3ppp/2n1pn2/3p4/3P4/2N1PN2/PP3PPP/R4RK1 w - - 0 14",
    "8/p1p2pkp/1p1p2p1/3P4/2P5/1P3KP1/P4P1P/8 w - - 0 30",
    "r2q1rk1/pp2bppp/2nppn2/8/3NP3/2N1BP2/PPPQB1PP/R4RK1 b - - 1 11",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="chess_small")
    ap.add_argument("--shards-dir", default="data/stockfish_injection")
    ap.add_argument("--n-games", type=int, default=300)
    ap.add_argument("--n-per-game", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-k", type=int, default=8)
    args = ap.parse_args()

    dev = args.device
    game = ChessGame()
    cfg = get_config(args.config); cfg.device = dev
    hf = cfg.history_frames
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    step = ckpt.get("step", "?")
    net, heads = build_network(ckpt, game, cfg, dev)

    print(f"\n{'='*72}\nREPR+DYN DEGENERACY — step {step}  cfg={args.config}\n  {args.checkpoint}\n  heads={heads}\n{'='*72}")

    # ---------- REPRESENTATION ----------
    obs, ev, out = sample_positions(args.shards_dir, game, hf, args.n_games, args.n_per_game, args.seed)
    M = len(obs)
    H = []
    with torch.no_grad():
        for s in range(0, M, 256):
            x = torch.from_numpy(obs[s:s+256]).to(dev)
            h = net.representation(x).reshape(x.shape[0], -1).float().cpu().numpy()
            H.append(h)
    H = np.concatenate(H, 0)
    D = H.shape[1]
    cpc = mean_crosspos_cosine(H, seed=args.seed)
    ent_rank, part_ratio = effective_rank(H)
    perdim_std = H.std(0)
    dead_frac = float((perdim_std < 1e-4).mean())
    r2_eval, _ = ridge_r2(H, ev, seed=args.seed)
    r2_out, op = ridge_r2(H, out, seed=args.seed)
    pred, yv = op
    dec = np.abs(yv) > 0.5
    sign_acc = float((np.sign(pred[dec]) == np.sign(yv[dec])).mean()) if dec.sum() else float("nan")

    print(f"\n[REPRESENTATION] M={M} positions, eval_std={ev.std():.3f}, D={D}")
    print(f"  cross-position cosine     {cpc:+.4f}   (->1.0 collapse; health <0.98)")
    print(f"  effective rank (entropy)  {ent_rank:8.1f}   (of {D})")
    print(f"  participation ratio       {part_ratio:8.1f}")
    print(f"  per-dim std mean/median   {perdim_std.mean():.4f}/{np.median(perdim_std):.4f}  dead_frac={dead_frac:.3f}")
    print(f"  linear-probe R2(eval)     {r2_eval:+.4f}")
    print(f"  linear-probe R2(outcome)  {r2_out:+.4f}")
    print(f"  decisive sign-accuracy    {sign_acc:.3f}")

    # ---------- DYNAMICS ----------
    print(f"\n[DYNAMICS] (cross-action / consistency / inverse / rollout fixed-point)")
    cross_cos_list, perpos_spread = [], []
    cos_dyn_repr_list, cos_root_dyn_list, cos_root_repr_list = [], [], []
    inv_acc_list, dyn_vstd_list = [], []
    from src.model.utils import wdl_to_scalar

    with torch.no_grad():
        for fen in DYN_FENS:
            board = chess.Board(fen)
            stm = chess.WHITE if board.turn == chess.WHITE else chess.BLACK
            from src.games.base import GameState
            state = GameState(board=board, current_player=1 if stm == chess.WHITE else -1)
            cur = game.to_tensor(state)
            obs_root = stack_with_history(cur, [], hf).unsqueeze(0).to(dev)
            h_root = net.representation(obs_root)

            legal = list(board.legal_moves)
            if len(legal) < 2:
                continue
            legal_actions = [_move_to_action(m, stm) for m in legal]
            probe = legal_actions[:16]
            acts = torch.tensor(probe, device=dev)
            hn = h_root.expand(len(probe), *h_root.shape[1:]).contiguous()
            dyn, _ = net.dynamics(hn, acts)

            dv = F.normalize(dyn.reshape(len(probe), -1), dim=-1)
            cc_mat = dv @ dv.T
            off = cc_mat[~torch.eye(len(probe), dtype=bool, device=dev)]
            cross_cos_list.append(off.mean().item())
            perpos_spread.append(off.mean().item())

            # value spread across actions
            _, vl = net.prediction(dyn)
            avs = wdl_to_scalar(vl.float(), draw_score=cfg.draw_score)
            dyn_vstd_list.append(avs.std().item())

            # inverse recovery
            if net.inverse_dynamics_head is not None:
                logits = net.predict_inverse_action(hn, dyn)
                inv_acc_list.append((logits.argmax(-1) == acts).float().mean().item())

            # consistency: dyn_next vs repr(actual next obs), and identity check
            for a, mv in list(zip(probe, legal))[:6]:
                if mv not in board.legal_moves:
                    continue
                nb = board.copy(); nb.push(mv)
                nst = GameState(board=nb, current_player=-state.current_player)
                ncur = game.to_tensor(nst)
                nobs = stack_with_history(ncur, [cur], hf).unsqueeze(0).to(dev)
                h_repr_next = net.representation(nobs)
                h_dyn_next, _ = net.dynamics(h_root, torch.tensor([int(a)], device=dev))
                cos_dyn_repr_list.append(F.cosine_similarity(h_dyn_next.flatten(), h_repr_next.flatten(), dim=0).item())
                cos_root_dyn_list.append(F.cosine_similarity(h_root.flatten(), h_dyn_next.flatten(), dim=0).item())
                cos_root_repr_list.append(F.cosine_similarity(h_root.flatten(), h_repr_next.flatten(), dim=0).item())

    def ms(x):
        x = np.array(x, dtype=np.float64); x = x[np.isfinite(x)]
        return (float(x.mean()), float(x.std())) if len(x) else (float("nan"), float("nan"))

    cc_m, cc_s = ms(cross_cos_list)
    cdr_m, cdr_s = ms(cos_dyn_repr_list)
    crd_m, _ = ms(cos_root_dyn_list)
    crr_m, _ = ms(cos_root_repr_list)
    inv_m, _ = ms(inv_acc_list)
    dvs_m, _ = ms(dyn_vstd_list)
    print(f"  cross-action cosine       {cc_m:+.4f} +-{cc_s:.4f}   (->1.0 action-blind; health <0.90)")
    print(f"  cos(dyn_next, repr_next)  {cdr_m:+.4f} +-{cdr_s:.4f}   (consistency; higher=better world model)")
    print(f"  cos(root, dyn_next)       {crd_m:+.4f}              (->1.0 dynamics~identity)")
    print(f"  cos(root, repr_next)      {crr_m:+.4f}              (natural drift baseline)")
    print(f"  inverse-recovery acc      {inv_m:.3f}              (action recoverable from (h,h'); health >0.30)")
    print(f"  dyn value-spread/action   {dvs_m:.4f}             (within-position Q signal; health >0.02)")

    # ---------- K-STEP ROLLOUT FIXED-POINT ----------
    print(f"\n[ROLLOUT] K={args.rollout_k}-step latent rollout (random legal actions)")
    with torch.no_grad():
        # use 1st fen
        board = chess.Board(DYN_FENS[0])
        from src.games.base import GameState
        state = GameState(board=board, current_player=1)
        cur = game.to_tensor(state)
        obs_root = stack_with_history(cur, [], hf).unsqueeze(0).to(dev)
        h = net.representation(obs_root)
        h0 = h.clone()
        prev = h.clone()
        step_cos, to_centroid_cos, step_delta = [], [], []
        rng = np.random.default_rng(args.seed)
        legal_actions = [_move_to_action(m, chess.WHITE) for m in board.legal_moves]
        for k in range(args.rollout_k):
            a = int(rng.choice(legal_actions))
            h_next, _ = net.dynamics(h, torch.tensor([a], device=dev))
            step_cos.append(F.cosine_similarity(prev.flatten(), h_next.flatten(), dim=0).item())
            to_centroid_cos.append(F.cosine_similarity(h0.flatten(), h_next.flatten(), dim=0).item())
            step_delta.append((h_next - prev).norm().item() / (prev.norm().item() + 1e-12))
            prev = h_next
            h = h_next
        print(f"  consecutive-step cosine : {[f'{c:.3f}' for c in step_cos]}")
        print(f"  cos to h0 (drift)       : {[f'{c:.3f}' for c in to_centroid_cos]}")
        print(f"  rel step delta norm     : {[f'{d:.3f}' for d in step_delta]}")
        print(f"  (rising step-cos + falling delta -> fixed-point collapse)")

    print(f"\nSUMMARY step={step} cpc={cpc:.4f} eff_rank={ent_rank:.1f} r2_eval={r2_eval:.4f} "
          f"r2_out={r2_out:.4f} sign_acc={sign_acc:.3f} cross_action={cc_m:.4f} "
          f"cos_dyn_repr={cdr_m:.4f} inv_acc={inv_m:.3f} dyn_vstd={dvs_m:.4f}")


if __name__ == "__main__":
    main()
