"""Trace the ACTUAL self-play buffer at checkpoint_6000.buf:
 - game outcomes + draw types + game lengths (is the collapse here?)
 - per-ply root_values distribution (what MCTS banks)
 - the WDL value TARGET that training sees (make_target), incl. q_ratio blend
 - what the value head PREDICTS on real buffer positions vs the target
 - reward head on real positions (non-terminal => should be ~0)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer
from src.model.utils import wdl_to_scalar
from scripts.eval_checkpoint_health import build_network

torch.serialization.add_safe_globals([MuZeroConfig])
CKPT = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"
BUF = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf"


@torch.no_grad()
def main():
    game = ChessGame(); cfg = get_config("chess_small"); cfg.device = "cpu"
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    net = build_network(ckpt, game, cfg, "cpu")

    rb = ReplayBuffer(cfg)
    rb.load(BUF, game=game)
    games = rb.buffer
    print(f"Loaded {len(games)} games from {BUF}")

    # ---- 1. Outcome / draw-type / length distribution ----
    outs = np.array([g.game_outcome for g in games], dtype=np.float64)
    lens = np.array([len(g.actions) for g in games], dtype=np.float64)
    n_draw = int(np.sum(outs == 0))
    n_3fold = sum(1 for g in games if getattr(g, "draw_by_repetition", False))
    n_noprog = sum(1 for g in games if getattr(g, "draw_by_no_progress", False))
    n_decisive = int(np.sum(np.abs(outs) >= 0.5))
    has_ext = sum(1 for g in games if g.external_values)
    print(f"\n[1] OUTCOMES: draws={n_draw}/{len(games)} ({n_draw/len(games):.1%})  "
          f"decisive={n_decisive} ({n_decisive/len(games):.1%})")
    print(f"    3fold-draws={n_3fold}  no-progress-draws={n_noprog}  warmstart(ext)={has_ext}")
    print(f"    game length: mean={lens.mean():.1f} median={np.median(lens):.0f} "
          f"min={lens.min():.0f} max={lens.max():.0f}")
    # length percentiles
    print(f"    length pctiles 10/25/50/75/90: "
          f"{np.percentile(lens,[10,25,50,75,90]).round(0)}")

    # ---- 2. root_values distribution (what MCTS banks per ply) ----
    all_rv = np.concatenate([np.array(g.root_values, dtype=np.float64) for g in games if g.root_values])
    print(f"\n[2] root_values across all plies (n={len(all_rv)}): "
          f"mean={all_rv.mean():+.4f} std={all_rv.std():.4f} "
          f"min={all_rv.min():+.4f} max={all_rv.max():+.4f}")
    print(f"    |root_value|>0.3 fraction: {np.mean(np.abs(all_rv)>0.3):.3f}  "
          f">0.5: {np.mean(np.abs(all_rv)>0.5):.3f}")

    # ---- 3. The actual WDL TARGET training sees (make_target with cfg blends) ----
    # Sample plies across draw games and check the target distribution.
    rng = np.random.default_rng(0)
    draw_games = [g for g in games if g.game_outcome == 0 and not g.external_values]
    print(f"\n[3] WDL TARGETS on self-play DRAW games (n_games={len(draw_games)}), "
          f"repetition_penalty={cfg.repetition_penalty} selfplay_q_ratio={cfg.selfplay_q_ratio}")
    tgt_W, tgt_D, tgt_L = [], [], []
    for g in rng.choice(len(draw_games), size=min(60, len(draw_games)), replace=False):
        gh = draw_games[int(g)]
        if len(gh) < 2:
            continue
        si = int(rng.integers(0, len(gh) - 1))
        out = gh.make_target(
            si, num_unroll_steps=0, td_steps=cfg.td_steps, discount=cfg.discount,
            action_space_size=game.action_space_size, history_frames=cfg.history_frames, value_head_type=cfg.value_head_type,
            q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
            selfplay_q_ratio=cfg.selfplay_q_ratio,
            repetition_penalty=cfg.repetition_penalty,
            repetition_penalty_decay=cfg.repetition_penalty_decay,
        )
        tv = out[3][0]  # target_values[0], shape (3,)
        tgt_W.append(float(tv[0])); tgt_D.append(float(tv[1])); tgt_L.append(float(tv[2]))
    tgt_W, tgt_D, tgt_L = map(np.array, (tgt_W, tgt_D, tgt_L))
    print(f"    target on draw plies: W={tgt_W.mean():.3f} D={tgt_D.mean():.3f} L={tgt_L.mean():.3f} (n={len(tgt_W)})")
    print(f"    => scalarized target V = W-L+draw_score*D = "
          f"{(tgt_W - tgt_L + cfg.draw_score*tgt_D).mean():+.4f}")

    # Same for DECISIVE games (winner side plies)
    dec_games = [g for g in games if abs(g.game_outcome) >= 0.5 and not g.external_values]
    if dec_games:
        dW, dD, dL = [], [], []
        for g in rng.choice(len(dec_games), size=min(60, len(dec_games)), replace=False):
            gh = dec_games[int(g)]
            if len(gh) < 2: continue
            si = int(rng.integers(0, len(gh)-1))
            out = gh.make_target(si, 0, cfg.td_steps, cfg.discount, game.action_space_size,
                                 history_frames=cfg.history_frames,
                                 value_head_type=cfg.value_head_type, q_ratio=cfg.q_ratio,
                                 warmstart_q_ratio=cfg.warmstart_q_ratio,
                                 selfplay_q_ratio=cfg.selfplay_q_ratio,
                                 repetition_penalty=cfg.repetition_penalty,
                                 repetition_penalty_decay=cfg.repetition_penalty_decay)
            tv = out[3][0]; dW.append(float(tv[0])); dD.append(float(tv[1])); dL.append(float(tv[2]))
        dW,dD,dL = map(np.array,(dW,dD,dL))
        print(f"    target on DECISIVE plies: W={dW.mean():.3f} D={dD.mean():.3f} L={dL.mean():.3f} (n={len(dW)})")

    # ---- 4. value head PREDICTION vs TARGET on real positions ----
    # Pull a batch of real positions (with history) and compare predicted WDL to target.
    print(f"\n[4] VALUE HEAD PRED vs TARGET (real in-distribution positions)")
    pred_V, tgt_V, pred_D = [], [], []
    sample_games = rng.choice(len(games), size=min(80, len(games)), replace=False)
    for gi in sample_games:
        gh = games[int(gi)]
        if gh.external_values or len(gh) < 4:
            continue
        si = int(rng.integers(0, len(gh)-1))
        obs = gh._stack_history(si, cfg.history_frames)
        _, vl = net.prediction(net.representation(obs.unsqueeze(0)))
        wdl = F.softmax(vl.float(), -1)[0].numpy()
        pV = wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).item()
        out = gh.make_target(si, 0, cfg.td_steps, cfg.discount, game.action_space_size,
                             history_frames=cfg.history_frames,
                             value_head_type=cfg.value_head_type, q_ratio=cfg.q_ratio,
                             warmstart_q_ratio=cfg.warmstart_q_ratio,
                             selfplay_q_ratio=cfg.selfplay_q_ratio,
                             repetition_penalty=cfg.repetition_penalty,
                             repetition_penalty_decay=cfg.repetition_penalty_decay)
        tv = out[3][0]
        tV = float(tv[0] - tv[2] + cfg.draw_score*tv[1])
        pred_V.append(pV); tgt_V.append(tV); pred_D.append(float(wdl[1]))
    pred_V, tgt_V, pred_D = map(np.array, (pred_V, tgt_V, pred_D))
    corr = np.corrcoef(pred_V, tgt_V)[0,1] if pred_V.std()>1e-6 and tgt_V.std()>1e-6 else float('nan')
    print(f"    pred V: mean={pred_V.mean():+.4f} std={pred_V.std():.4f} | "
          f"target V: mean={tgt_V.mean():+.4f} std={tgt_V.std():.4f}")
    print(f"    corr(pred V, target V)={corr:+.3f}   mean pred P(draw)={pred_D.mean():.3f}  (n={len(pred_V)})")


if __name__ == "__main__":
    main()
