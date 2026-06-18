"""Canonical checkpoint health evaluation (cross-run).

Loads a checkpoint (CPU by default — safe to run alongside a live GPU training
run) and reports a structured health verdict across six dimensions:

  1. Training-loss trajectories (from TensorBoard) — reasonable / not collapsed?
  2. Representation: latent health + are distinct positions distinctly encoded?
  3. Value head: calibration on clearly-won/drawn/lost positions + spread + draw saturation.
  4. Dynamics + inverse head: cross-action separation, dyn-vs-repr-next alignment,
     and inverse-action recovery accuracy (is the dynamics output action-conditioned?).
  5. Reasonable moves: model top move vs Stockfish top move (overlap).
  6. MCTS trees: root value + top children (visits, Q, prior) — are values sane?

Ends with a DECISION gate (parameterized): HOLD if the inverse head isn't working,
PROCEED-with-scheduled-changes if the value head is the next bottleneck, else AWAIT.

Usage:
  .venv/bin/python scripts/eval_checkpoint_health.py --checkpoint <path.pt> [--device cpu]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import chess
import chess.engine

from src.config import get_config
from src.games.chess import ChessGame, _move_to_action, _action_to_move
from src.model.muzero_net import MuZeroNetwork, _min_max_normalize
from src.training.replay_buffer import stack_with_history, _iter_shard_games

# --- thresholds (tune once; shared across runs) ----------------------------
TH = dict(
    latent_std_min=0.01, crosspos_cos_max=0.98,
    value_calib_win=0.25, value_calib_draw=0.35, value_spread_min=0.05, draw_sat_max=0.85,
    value_eval_corr_min=0.50,  # corr(head V, Stockfish eval) on in-distribution pool positions
    inv_recovery_min=0.30, crossaction_cos_max=0.90, dyn_value_std_min=0.02,
    sf_overlap_min=0.30,
)


def pool_calib_positions(game, hf, pool_dir, seed, n_per_bucket=40, max_games=400):
    """In-distribution calibration set: real Stockfish-pool positions WITH full
    8-frame history (no zero-padding), binned by their side-to-move eval into
    clear-win / balanced-draw / clear-loss. Returns
    (buckets: {label: [(obs, eval)]}, flat: [(obs, eval)]).
    Replaces the old constructed-FEN-with-zero-pad calibration, which was
    material- AND history-OOD (warmstart data is dense middlegames with history).
    """
    import glob
    rng = np.random.default_rng(seed)
    paths = sorted(glob.glob(os.path.join(pool_dir, "**", "*.pkl"), recursive=True))
    rng.shuffle(paths)
    buckets = {"win (eval>+0.5)": [], "draw (|eval|<0.1)": [], "loss (eval<-0.5)": []}
    flat = []
    used = 0
    for p in paths:
        if used >= max_games or all(len(v) >= n_per_bucket for v in buckets.values()):
            break
        try:
            games = list(_iter_shard_games(p, game=game))
        except Exception:
            continue
        rng.shuffle(games)
        for gh in games:
            if used >= max_games:
                break
            n_ev = len(gh.external_values)
            if n_ev == 0:
                continue
            used += 1
            for ply in rng.choice(n_ev, size=min(8, n_ev), replace=False):
                ply = int(ply)
                ev = float(gh.external_values[ply])
                obs = gh._stack_history(ply, hf)  # real history, STM-relative
                flat.append((obs, ev))
                if ev > 0.5 and len(buckets["win (eval>+0.5)"]) < n_per_bucket:
                    buckets["win (eval>+0.5)"].append((obs, ev))
                elif abs(ev) < 0.1 and len(buckets["draw (|eval|<0.1)"]) < n_per_bucket:
                    buckets["draw (|eval|<0.1)"].append((obs, ev))
                elif ev < -0.5 and len(buckets["loss (eval<-0.5)"]) < n_per_bucket:
                    buckets["loss (eval<-0.5)"].append((obs, ev))
    return buckets, flat
# Below this step, a lagging value head is read as "early/still-developing", not as
# "the persistent bottleneck" — so the gate says KEEP-RUNNING rather than PROCEED.
EARLY_STEPS = 6000


def build_network(ckpt, game, cfg, device):
    # Detect heads from the checkpoint so old and new architectures both load
    # (matches play_web.load_network). Lets the probe compare e.g. a conv+moves-left
    # checkpoint against a conv-only one.
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
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net


def random_playout(game, n_plies, seed):
    rng = np.random.default_rng(seed)
    s = game.reset(); frames = []
    for _ in range(n_plies):
        if s.done:
            break
        frames.append(game.to_tensor(s))
        s, _, _ = game.step(s, int(rng.choice(game.legal_actions(s))))
    return s, frames  # state at the position, prior frames (history)


def obs_from_fen(game, fen):
    board = chess.Board(fen)
    state = game.reset()
    state.board = board  # ChessGame state wraps a python-chess board
    return state, board


# Clear value-calibration positions (zero-padded history — flagged).
CALIB = [
    ("white up a queen (win)",  "4k3/8/8/8/8/8/8/3QK3 w - - 0 1",  +1),
    ("dead K vs K (draw)",      "4k3/8/8/8/8/8/8/4K3 w - - 0 1",    0),
    ("white down a queen (loss)","3qk3/8/8/8/8/8/8/4K3 w - - 0 1", -1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mcts-sims", type=int, default=40)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--pool-dir", default="data/stockfish_injection",
                    help="Stockfish shard dir for in-distribution value calibration "
                         "(real positions + full history). Falls back to constructed "
                         "FENs if the pool is absent.")
    ap.add_argument("--logdir", default=None)
    args = ap.parse_args()

    dev = args.device
    game = ChessGame(); HF = get_config("chess").history_frames
    cfg = get_config("chess"); cfg.device = dev
    from src.config import MuZeroConfig
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    step = ckpt.get("step", "?")
    net = build_network(ckpt, game, cfg, dev)
    print(f"\n{'='*70}\nCHECKPOINT HEALTH EVAL — step {step}  ({args.checkpoint})\n{'='*70}")
    verdict = {}

    # ---- 1. Loss trajectories from TB --------------------------------------
    logdir = args.logdir or os.path.join("runs", "chess", os.path.basename(os.path.dirname(args.checkpoint)))
    print(f"\n[1] LOSS TRAJECTORIES  (logdir={logdir})")
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(logdir); ea.Reload()
        def last(tag):
            try: return ea.Scalars(tag)[-1].value
            except Exception: return float("nan")
        def first(tag):
            try: return ea.Scalars(tag)[0].value
            except Exception: return float("nan")
        for t in ["loss/total_loss","loss/value_loss","loss/policy_loss","loss/inverse_loss",
                  "loss/consistency_loss","loss/reward_loss","value/mae","value/target_std",
                  "dynamics/cross_action_cos","train/grad_norm"]:
            print(f"    {t:28s} {first(t):.4f} -> {last(t):.4f}")
        losses_ok = (np.isfinite(last("loss/total_loss"))
                     and last("loss/inverse_loss") < first("loss/inverse_loss")  # inverse decreasing
                     and last("value/mae") < last("value/target_std") * 1.05)     # value beats mean-predict
        verdict["losses_ok"] = bool(losses_ok)
    except Exception as e:
        print(f"    (could not read TB: {e})")
        verdict["losses_ok"] = None

    # ---- build probe positions: midgame (real history) + calib (zero-pad) ---
    mids = []
    for i, n in enumerate([6, 14, 22, 30]):
        s, frames = random_playout(game, n, seed=100 + i)
        cur = game.to_tensor(s)
        obs = stack_with_history(cur, frames, HF)
        mids.append((f"midgame@{n}ply", s, s.board, obs, game.legal_actions(s)))

    @torch.no_grad()
    def eval_pos(obs):
        h = net.representation(obs.unsqueeze(0).to(dev))
        pl, vl = net.prediction(h)
        from src.model.utils import wdl_to_scalar
        wdl = F.softmax(vl.float(), -1)[0]
        v = wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).item()
        return h, pl[0], wdl, v

    # ---- 2. Representation health ------------------------------------------
    print(f"\n[2] REPRESENTATION")
    hs = []
    for name, s, board, obs, legal in mids:
        h, _, _, _ = eval_pos(obs); hs.append(h.flatten())
    H = torch.stack(hs)
    latent_std = H.std(0).mean().item()
    Hn = F.normalize(H, dim=-1); cpc = (Hn @ Hn.T)[~torch.eye(len(H), dtype=bool)].mean().item()
    print(f"    latent range [{H.min():.3f},{H.max():.3f}] per-dim std {latent_std:.3f} | cross-position cos {cpc:.3f}")
    verdict["repr_ok"] = bool(latent_std > TH["latent_std_min"] and cpc < TH["crosspos_cos_max"]
                              and torch.isfinite(H).all())

    # ---- 3. Value head: calibration + spread + draw saturation -------------
    print(f"\n[3] VALUE HEAD")
    # In-distribution calibration: real Stockfish-pool positions WITH full history,
    # binned by side-to-move eval. (The old constructed-FEN check used zero-padded
    # history on material-OOD toy endgames; kept below as an informational readout
    # only — it no longer gates the verdict.)
    calibrated = None
    veval_corr = float("nan")
    buckets, flat = (({}, []) if not os.path.isdir(args.pool_dir)
                     else pool_calib_positions(game, HF, args.pool_dir, seed=7))
    if flat:
        @torch.no_grad()
        def batch_V(obs_list):
            xs = torch.stack(obs_list).to(dev)
            _, vl = net.prediction(net.representation(xs))
            from src.model.utils import wdl_to_scalar
            return wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).cpu().numpy()
        bucket_means = {}
        for label, items in buckets.items():
            if items:
                vs = batch_V([o for o, _ in items])
                bucket_means[label] = (float(np.mean(vs)), float(np.std(vs)), len(items))
                print(f"    {label:20s} mean V={bucket_means[label][0]:+.3f} "
                      f"±{bucket_means[label][1]:.3f}  (n={len(items)})  [real history]")
        Vall = batch_V([o for o, _ in flat]); Eall = np.array([e for _, e in flat])
        if Vall.std() > 1e-6 and Eall.std() > 1e-6:
            veval_corr = float(np.corrcoef(Vall, Eall)[0, 1])
        slope = float(np.polyfit(Eall, Vall, 1)[0]) if Eall.std() > 1e-6 else float("nan")
        print(f"    corr(V, Stockfish eval)={veval_corr:+.3f}  slope={slope:+.3f}  (n={len(flat)}, in-distribution)")
        vw = bucket_means.get("win (eval>+0.5)", (None,))[0]
        vd = bucket_means.get("draw (|eval|<0.1)", (None,))[0]
        vl_ = bucket_means.get("loss (eval<-0.5)", (None,))[0]
        calibrated = bool(
            veval_corr > TH["value_eval_corr_min"]
            and (vw is None or vw > TH["value_calib_win"])
            and (vl_ is None or vl_ < -TH["value_calib_win"])
            and (vd is None or abs(vd) < TH["value_calib_draw"])
        )
    # Informational: constructed FENs (zero-pad history, material-OOD) — NOT gating.
    for name, fen, sign in CALIB:
        st, board = obs_from_fen(game, fen)
        cur = game.to_tensor(st); obs = stack_with_history(cur, [], HF)
        _, _, wdl, v = eval_pos(obs)
        print(f"    [ref] {name:24s} V={v:+.3f}  WDL=({wdl[0]:.2f},{wdl[1]:.2f},{wdl[2]:.2f})  [constructed, zero-pad — OOD]")
    if calibrated is None:  # no pool — fall back to the constructed-FEN gate
        cv = {}
        for name, fen, sign in CALIB:
            st, board = obs_from_fen(game, fen)
            obs = stack_with_history(game.to_tensor(st), [], HF)
            _, _, _, v = eval_pos(obs); cv[sign] = v
        calibrated = (cv.get(+1, 0) > TH["value_calib_win"] and cv.get(-1, 0) < -TH["value_calib_win"]
                      and abs(cv.get(0, 1)) < TH["value_calib_draw"])
        print("    (no pool — calibration gated on constructed FENs, zero-pad)")
    mid_vs, mid_pd = [], []
    for name, s, board, obs, legal in mids:
        _, _, wdl, v = eval_pos(obs); mid_vs.append(v); mid_pd.append(wdl[1].item())
    vspread = float(np.std(mid_vs)); draw_sat = float(np.mean(mid_pd))
    print(f"    midgame V spread (std) {vspread:.3f} | mean P(draw) {draw_sat:.3f}")
    verdict["value_calibrated"] = bool(calibrated)
    verdict["value_spread_ok"] = bool(vspread > TH["value_spread_min"])
    verdict["value_not_draw_saturated"] = bool(draw_sat < TH["draw_sat_max"])
    verdict["value_health"] = bool(calibrated and vspread > TH["value_spread_min"] and draw_sat < TH["draw_sat_max"])

    # ---- 4. Dynamics + inverse head ----------------------------------------
    print(f"\n[4] DYNAMICS + INVERSE HEAD")
    cross_cos, inv_acc, dyn_vstd, dynrepr_cos = [], [], [], []
    for name, s, board, obs, legal in mids:
        with torch.no_grad():
            h = net.representation(obs.unsqueeze(0).to(dev))
            probe = legal[:16] if len(legal) >= 2 else legal
            acts = torch.tensor(probe, device=dev)
            hn = h.expand(len(probe), *h.shape[1:]).contiguous()
            dyn, _ = net.dynamics(hn, acts)
            dv = F.normalize(dyn.reshape(len(probe), -1), dim=-1)
            cc = (dv @ dv.T)[~torch.eye(len(probe), dtype=bool)].mean().item() if len(probe) > 1 else float("nan")
            cross_cos.append(cc)
            # value spread across actions (does prediction see action-distinct states?)
            _, vl = net.prediction(dyn)
            from src.model.utils import wdl_to_scalar
            avs = wdl_to_scalar(vl.float(), draw_score=cfg.draw_score)
            dyn_vstd.append(avs.std().item())
            # inverse-action recovery
            if net.inverse_dynamics_head is not None:
                logits = net.predict_inverse_action(hn, dyn)
                acc = (logits.argmax(-1) == acts).float().mean().item()
                inv_acc.append(acc)
            # dyn vs repr(actual next obs) alignment for first few legal moves
            algs = []
            for a in probe[:4]:
                mv = _action_to_move(int(a), board)
                if mv is None or mv not in board.legal_moves:
                    continue
                nb = board.copy(); nb.push(mv)
                nst = game.reset(); nst.board = nb
                nobs = stack_with_history(game.to_tensor(nst), [], HF)
                hnext = net.representation(nobs.unsqueeze(0).to(dev))
                hd, _ = net.dynamics(h, torch.tensor([int(a)], device=dev))
                algs.append(F.cosine_similarity(hd.flatten(), hnext.flatten(), dim=0).item())
            if algs:
                dynrepr_cos.append(float(np.mean(algs)))
    cc_m = float(np.nanmean(cross_cos)); inv_m = float(np.mean(inv_acc)) if inv_acc else float("nan")
    dvstd_m = float(np.mean(dyn_vstd)); dr_m = float(np.mean(dynrepr_cos)) if dynrepr_cos else float("nan")
    print(f"    cross-action cos {cc_m:.3f} (1=blind) | inverse-recovery acc {inv_m:.3f} | "
          f"dyn value-spread/action {dvstd_m:.3f} | cos(dyn, repr_next) {dr_m:.3f}")
    # Inverse head working == the DYNAMICS is action-conditioned: action recoverable
    # from (h, h_next) AND outputs separated across actions. The dyn-value-spread is a
    # VALUE-head property (does the value head map action-distinct states to distinct
    # values) — a SEPARATE coupling diagnostic, NOT an inverse-head criterion. Including
    # it spuriously fails inverse_working when the dynamics is action-aware but the value
    # head is still near-constant/under-trained (observed at step 2000: recovery 0.95,
    # cross-action cos 0.78 — clearly working — but dyn_value_std 0.001).
    verdict["inverse_working"] = bool(
        (np.isnan(inv_m) or inv_m > TH["inv_recovery_min"]) and cc_m < TH["crossaction_cos_max"]
    )
    verdict["dyn_value_coupling"] = bool(dvstd_m > TH["dyn_value_std_min"])  # value-side: uses the action-distinct states

    # ---- 5. Reasonable moves vs Stockfish ----------------------------------
    print(f"\n[5] REASONABLE MOVES (model top-3 vs Stockfish top move)")
    sf_hits = 0; sf_total = 0; eng = None
    try:
        eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        for name, s, board, obs, legal in mids:
            _, pl, _, _ = eval_pos(obs)
            order = pl.argsort(descending=True).tolist()
            top, seen = [], 0
            for ai in order:
                mv = _action_to_move(ai, board)
                if mv is not None and mv in board.legal_moves:
                    top.append(mv); seen += 1
                if seen >= 3: break
            sf_best = eng.play(board, chess.engine.Limit(depth=10)).move
            hit = sf_best in top; sf_hits += int(hit); sf_total += 1
            print(f"    {name:14s} model_top3={[board.san(m) for m in top]} SF={board.san(sf_best)} {'HIT' if hit else 'miss'}")
    except Exception as e:
        print(f"    (stockfish unavailable / error: {e})")
    finally:
        if eng: eng.quit()
    sf_overlap = (sf_hits / sf_total) if sf_total else float("nan")
    print(f"    SF top move in model top-3: {sf_hits}/{sf_total} = {sf_overlap:.2f}")
    verdict["moves_reasonable"] = bool(np.isnan(sf_overlap) or sf_overlap >= TH["sf_overlap_min"])

    # ---- 6. MCTS tree inspection -------------------------------------------
    print(f"\n[6] MCTS TREES ({args.mcts_sims} sims)")
    from src.mcts.mcts import MCTS, select_action
    cfg.num_simulations = args.mcts_sims
    mcts = MCTS(net, game, cfg, dev)
    mcts_ok = []
    for name, s, board, obs, legal in mids[:3]:
        try:
            root = mcts.run(obs, legal, add_noise=False)
            order = np.argsort(-(root.child_visits if root.child_visits is not None else np.array([0.])))
            print(f"    {name}: root.value={root.value:+.3f}  legal={len(legal)}")
            qs = []
            for j in order[:4]:
                a = int(root.child_actions[j]); vis = float(root.child_visits[j])
                q = (root.child_value_sums[j] / vis) if vis > 0 else 0.0
                pr = float(root.child_priors[j]); qs.append(q)
                mv = _action_to_move(a, board); san = board.san(mv) if (mv and mv in board.legal_moves) else f"a{a}"
                print(f"        {san:6s} visits={vis:4.0f} Q={q:+.3f} prior={pr:.3f}")
            mcts_ok.append(float(np.std(qs)) > 0.01)  # Q's not all identical (flat-Q = draw-basin symptom)
        except Exception as e:
            print(f"    {name}: MCTS error {e}"); mcts_ok.append(False)
    verdict["mcts_values_reasonable"] = bool(mcts_ok and any(mcts_ok))

    # ---- VERDICT + DECISION ------------------------------------------------
    print(f"\n{'='*70}\nVERDICT")
    for k, v in verdict.items():
        print(f"    [{'PASS' if v else ('????' if v is None else 'FAIL')}] {k}")
    inv = verdict["inverse_working"]; valh = verdict["value_health"]
    early = isinstance(step, int) and step < EARLY_STEPS
    print(f"\nDECISION:")
    if not inv:
        print("  >>> HOLD — inverse action head NOT working as intended "
              "(cross-action cos high AND recovery low).\n"
              "  >>> Wait for user intervention before the scheduled changes.")
    elif valh:
        print("  >>> AWAIT — inverse head works AND value head looks healthy; value not yet the\n"
              "  >>> bottleneck. Report and let the run continue.")
    elif early:
        print(f"  >>> KEEP-RUNNING — inverse head works; value head lagging but step {step} < {EARLY_STEPS}\n"
              "  >>> (too early to call value THE bottleneck; it may still develop spread).\n"
              "  >>> Re-check at the next +2000 checkpoint.")
    else:
        print("  >>> PROCEED — inverse head works; value head is the PERSISTENT bottleneck\n"
              "  >>> (under-spread / draw-saturated / mis-calibrated past the early window).\n"
              "  >>> Implement scheduled changes (q_ratio blend + analysis bundle), then surface before relaunch.")
    print('='*70)


if __name__ == "__main__":
    main()
