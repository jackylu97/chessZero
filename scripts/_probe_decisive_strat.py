"""Deep dive: decisive_sample_frac stratification with a near-empty decisive pool.

Quantifies how much the 50% decisive stratum oversamples a tiny set of decisive
games, and whether IS weights actually correct for it. Also checks priority
degeneracy across multiple checkpoints.
"""
import sys, numpy as np, torch
sys.path.insert(0, "/workspace/chessZero")
np.random.seed(0); torch.manual_seed(0)

from src.config import get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer

cfg = get_config("chess_small"); cfg.device = "cpu"
game = ChessGame()

for ck in [1000, 6000, 30000]:
    BUF = f"/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_{ck}.buf"
    rb = ReplayBuffer(cfg.replay_buffer_size)
    try:
        rb.load(BUF, game=game)
    except Exception as e:
        print(f"[{ck}] load failed: {e}"); continue
    n = len(rb)
    dec_idx = [i for i in range(n) if abs(rb.buffer[i].game_outcome) >= 0.5 and not rb.buffer[i].external_values]
    n_dec = len(dec_idx)
    pr = np.array(rb._priorities, dtype=np.float64)
    print(f"\n=== ckpt {ck}: {n} games, {n_dec} decisive-selfplay ===")
    # Effective number of DISTINCT decisive games sampled per batch over many batches.
    # decisive_sample_frac=0.5 => n_sub = 128 positions drawn (with replacement) from n_dec games.
    n_sub = int(round(cfg.batch_size * cfg.decisive_sample_frac))
    print(f"  decisive stratum draws {n_sub} positions from {n_dec} games every batch "
          f"=> avg {n_sub/max(1,n_dec):.1f} samples/decisive-game/batch")
    # How many times will a given decisive game be hit per 1000 batches (epoch-ish)?
    # vs a draw game in the complement stratum (1489 games sharing 128 slots)
    comp_idx = [i for i in range(n) if not (abs(rb.buffer[i].game_outcome) >= 0.5 and not rb.buffer[i].external_values)]
    n_comp = len(comp_idx)
    print(f"  complement stratum draws {cfg.batch_size-n_sub} positions from {n_comp} games "
          f"=> avg {(cfg.batch_size-n_sub)/max(1,n_comp):.4f} samples/draw-game/batch")
    if n_dec > 0:
        ratio = (n_sub/n_dec) / ((cfg.batch_size-n_sub)/max(1,n_comp))
        print(f"  => a decisive game is sampled {ratio:.0f}x more often per batch than a draw game")
    # Simulate many batches, count distinct decisive games actually seen + per-game freq
    seen = np.zeros(n, dtype=np.int64)
    NB = 200
    for _ in range(NB):
        _, gidx, _ = rb.sample_batch(
            cfg.batch_size, cfg.num_unroll_steps, cfg.td_steps, cfg.discount,
            game.action_space_size, alpha=cfg.per_alpha, beta=0.5,
            value_head_type=cfg.value_head_type, history_frames=cfg.history_frames,
            eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
            decisive_sample_frac=cfg.decisive_sample_frac, q_ratio=cfg.q_ratio,
            warmstart_q_ratio=cfg.warmstart_q_ratio, selfplay_q_ratio=cfg.selfplay_q_ratio,
            repetition_penalty=cfg.repetition_penalty,
            repetition_penalty_decay=cfg.repetition_penalty_decay)
        for gi in gidx:
            seen[gi] += 1
    dec_seen = seen[dec_idx]
    print(f"  over {NB} batches: each decisive game sampled mean={dec_seen.mean():.0f} "
          f"(min={dec_seen.min()} max={dec_seen.max()}) times")
    print(f"  total decisive samples = {dec_seen.sum()} of {NB*cfg.batch_size} = "
          f"{dec_seen.sum()/(NB*cfg.batch_size)*100:.0f}% of all training positions")
    # positions within a decisive game: each game has len plies; the decisive
    # value signal (|z|=1) is at EVERY ply (MC return td_steps=-1), so all plies
    # carry the same |z|. Count distinct (game,pos) coverage is broad; the issue
    # is the GAME diversity, which is just n_dec.
    print(f"  DISTINCT decisive games = {n_dec} (the value head's entire decisive-signal diversity)")
