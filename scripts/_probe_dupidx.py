"""Duplicate game_indices in a stratified batch -> priority update behavior.

Under decisive_sample_frac, the same decisive game index appears MANY times in
one batch (128 samples from ~11-71 games). update_priorities iterates
zip(game_indices, td_errors) and writes self._priorities[idx]=err each time, so
for a duplicated idx the LAST write wins. We measure how many distinct vs total
indices a batch has, and demonstrate last-write-wins discards most TD info.
"""
import sys, numpy as np
sys.path.insert(0, "/workspace/chessZero")
np.random.seed(2)
from src.config import get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer

cfg = get_config("chess_small"); cfg.device="cpu"
game = ChessGame()
rb = ReplayBuffer(cfg.replay_buffer_size)
rb.load("/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf", game=game)

batch, gidx, isw = rb.sample_batch(
    cfg.batch_size, cfg.num_unroll_steps, cfg.td_steps, cfg.discount, game.action_space_size,
    alpha=cfg.per_alpha, beta=0.52, value_head_type=cfg.value_head_type,
    history_frames=cfg.history_frames, eval_to_wdl_alpha=cfg.eval_to_wdl_alpha,
    eval_to_wdl_beta=cfg.eval_to_wdl_beta, decisive_sample_frac=cfg.decisive_sample_frac,
    q_ratio=cfg.q_ratio, warmstart_q_ratio=cfg.warmstart_q_ratio,
    selfplay_q_ratio=cfg.selfplay_q_ratio, repetition_penalty=cfg.repetition_penalty,
    repetition_penalty_decay=cfg.repetition_penalty_decay)

uniq, counts = np.unique(gidx, return_counts=True)
print(f"batch has {len(gidx)} samples, {len(uniq)} DISTINCT game indices")
print(f"  max duplication of a single game in one batch: {counts.max()}x")
print(f"  games appearing >1x: {(counts>1).sum()}  (their TD errors collide on update)")
dup = (counts>1).sum()
print(f"  => update_priorities does last-write-wins: for {dup} duplicated games,")
print(f"     all but 1 of their { (counts[counts>1].sum()) } sampled positions' TD errors are DISCARDED")
print(f"     ({counts[counts>1].sum()-dup} of {len(gidx)} = "
      f"{(counts[counts>1].sum()-dup)/len(gidx)*100:.0f}% of the batch's priority updates lost)")
