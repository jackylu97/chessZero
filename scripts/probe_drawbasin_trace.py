"""Trace actual per-ply value TARGETS (make_target) for real drawn games and
compare to the network's predicted WDL. CPU-only.

Goal: distinguish "value head genuinely cannot tell" (fundamental) from a BUG
where the target is right but something downstream still draws.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F

from src.config import MuZeroConfig, get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer
from src.model.utils import wdl_to_scalar, eval_to_wdl
from scripts.eval_checkpoint_health import build_network

torch.serialization.add_safe_globals([MuZeroConfig])

CKPT = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"
BUF  = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf"

def main():
    dev = "cpu"
    game = ChessGame()
    cfg = get_config("chess_small"); cfg.device = dev
    HF = cfg.history_frames
    print("CONFIG (the knobs that shape value targets):")
    for k in ["value_head_type","draw_score","td_steps","discount","q_ratio",
              "warmstart_q_ratio","selfplay_q_ratio","repetition_penalty",
              "repetition_penalty_decay","repetition_penalty_window",
              "eval_to_wdl_alpha","eval_to_wdl_beta","num_unroll_steps",
              "value_loss_weight_selfplay","decisive_sample_frac"]:
        print(f"  {k} = {getattr(cfg,k,'<MISSING>')}")

    ck = torch.load(CKPT, map_location=dev, weights_only=True)
    net = build_network(ck, game, cfg, dev)
    print(f"\nloaded net step={ck.get('step')}")

    rb = ReplayBuffer(max_size=100000)
    rb.load(BUF, game=game)

    # pick a repetition-drawn self-play game of moderate length
    cand = [i for i,g in enumerate(rb.buffer)
            if g.draw_by_repetition and 40 <= len(g) <= 80 and not g.external_values]
    gi = cand[0]
    g = rb.buffer[gi]
    print(f"\n=== TRACE GAME {gi}: len={len(g)} outcome={g.game_outcome} "
          f"rep={g.draw_by_repetition} nop={g.draw_by_no_progress} "
          f"reanalyze={g.reanalyze_count} ===")

    nus = cfg.num_unroll_steps
    rep_end = len(g) - 1

    @torch.no_grad()
    def net_wdl_and_v(idx):
        obs = g._stack_history(idx, HF).unsqueeze(0)
        h = net.representation(obs)
        _, vl = net.prediction(h)
        wdl = F.softmax(vl.float(), -1)[0].numpy()
        v = wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).item()
        return wdl, v

    plies = [0, 1, len(g)//4, len(g)//2, 3*len(g)//4, len(g)-3, len(g)-2, len(g)-1]
    plies = sorted(set(p for p in plies if 0 <= p < len(g)))
    print(f"\n{'ply':>4} {'STM':>5} {'rootV':>7} | {'TARGET(W,D,L)':>22} {'tgtV':>6} {'rampW':>6}"
          f" | {'NET(W,D,L)':>22} {'netV':>6}")
    for p in plies:
        obs_list, mask, acts, values, rewards, pols = g.make_target(
            p, nus, cfg.td_steps, cfg.discount, game.action_space_size,
            value_head_type=cfg.value_head_type, history_frames=HF,
            eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
            q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
            selfplay_q_ratio=cfg.selfplay_q_ratio,
            repetition_penalty=cfg.repetition_penalty,
            repetition_penalty_window=int(getattr(cfg,"repetition_penalty_window",0)),
            repetition_penalty_decay=float(getattr(cfg,"repetition_penalty_decay",0.0)),
        )
        tgt = np.asarray(values[0], dtype=np.float64)  # (3,) W,D,L
        tgtv = tgt[0] - tgt[2] + cfg.draw_score*tgt[1]
        # what the ramp weight is at this ply
        plies_to_end = rep_end - p
        ramp = cfg.repetition_penalty * (cfg.repetition_penalty_decay ** plies_to_end)
        wdl, netv = net_wdl_and_v(p)
        stm = "W" if p%2==0 else "B"
        rv = g.root_values[p] if p < len(g.root_values) else float('nan')
        print(f"{p:>4} {stm:>5} {rv:>7.3f} | ({tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f})  {tgtv:>+6.3f} {ramp:>6.3f}"
              f" | ({wdl[0]:.3f},{wdl[1]:.3f},{wdl[2]:.3f})  {netv:>+6.3f}")

    # Aggregate over ALL repetition-drawn games: what fraction of target mass
    # is W vs D vs L, and what is mean target scalar V?
    print("\n=== AGGREGATE over self-play games (target side) ===")
    agg = {"rep":[], "dec":[]}
    rng = np.random.default_rng(0)
    sample = rng.choice(len(rb.buffer), size=min(400,len(rb.buffer)), replace=False)
    tgtW=tgtD=tgtL=0.0; ntgt=0; tgtvs=[]
    for i in sample:
        gg = rb.buffer[int(i)]
        if gg.external_values:
            continue
        # sample a few plies
        for _ in range(4):
            p = int(rng.integers(0, len(gg)))
            _,_,_,values,_,_ = gg.make_target(
                p, nus, cfg.td_steps, cfg.discount, game.action_space_size,
                value_head_type=cfg.value_head_type, history_frames=HF,
                eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
                q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
                selfplay_q_ratio=cfg.selfplay_q_ratio,
                repetition_penalty=cfg.repetition_penalty,
                repetition_penalty_window=int(getattr(cfg,"repetition_penalty_window",0)),
                repetition_penalty_decay=float(getattr(cfg,"repetition_penalty_decay",0.0)),
            )
            t = np.asarray(values[0],dtype=np.float64)
            tgtW+=t[0];tgtD+=t[1];tgtL+=t[2];ntgt+=1
            tgtvs.append(t[0]-t[2]+cfg.draw_score*t[1])
    print(f"  n_target_samples={ntgt}  mean target WDL=({tgtW/ntgt:.4f},{tgtD/ntgt:.4f},{tgtL/ntgt:.4f})")
    print(f"  mean target scalar V (incl draw_score)= {np.mean(tgtvs):+.4f}  std={np.std(tgtvs):.4f}")

if __name__ == "__main__":
    main()
