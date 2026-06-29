"""IS-weight correctness under decisive stratification.

The PER IS weight is supposed to make the *expected weighted gradient* equal to
the uniform-sampling gradient: E_sample[w_i * g_i] ∝ (1/N) Σ_i g_i.
That requires w_i ∝ 1 / P_actual(i), where P_actual is the TRUE marginal
sampling prob. Under stratification P_actual(i) = (n_stratum/B) * P_stratum(i).

The code uses w_i = (N_stratum * P_stratum(i))^(-beta), which omits the
(n_stratum/B) allocation factor and uses N_stratum (not N). We measure, for a
decisive game vs a draw game of EQUAL priority, the ratio of
  (actual sampling frequency)  vs  (IS weight)
If IS correctly compensates, freq_dec/freq_draw should ~= w_draw/w_dec.
"""
import sys, numpy as np
sys.path.insert(0, "/workspace/chessZero")
np.random.seed(1)
from src.config import get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer

cfg = get_config("chess_small"); cfg.device="cpu"
game = ChessGame()
BUF = "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf"
rb = ReplayBuffer(cfg.replay_buffer_size); rb.load(BUF, game=game)
n = len(rb)
dec = np.array([i for i in range(n) if abs(rb.buffer[i].game_outcome)>=0.5 and not rb.buffer[i].external_values])
draw = np.array([i for i in range(n) if not (abs(rb.buffer[i].game_outcome)>=0.5 and not rb.buffer[i].external_values)])
print(f"buffer {n}: decisive={len(dec)} draw/other={len(draw)}")

# Accumulate, per game: sampling frequency and SUM of IS weights it received.
freq = np.zeros(n); wsum = np.zeros(n)
NB = 60
beta = 0.52
for _ in range(NB):
    _, gidx, isw = rb.sample_batch(
        cfg.batch_size, cfg.num_unroll_steps, cfg.td_steps, cfg.discount,
        game.action_space_size, alpha=cfg.per_alpha, beta=beta,
        value_head_type=cfg.value_head_type, history_frames=cfg.history_frames,
        eval_to_wdl_alpha=cfg.eval_to_wdl_alpha, eval_to_wdl_beta=cfg.eval_to_wdl_beta,
        decisive_sample_frac=cfg.decisive_sample_frac, q_ratio=cfg.q_ratio,
        warmstart_q_ratio=cfg.warmstart_q_ratio, selfplay_q_ratio=cfg.selfplay_q_ratio,
        repetition_penalty=cfg.repetition_penalty, repetition_penalty_decay=cfg.repetition_penalty_decay)
    for gi, w in zip(gidx, isw):
        freq[gi]+=1; wsum[gi]+=w

# Per-game: effective weighted-sample mass = sum of IS weights it contributed.
# If IS perfectly equalized to uniform, every game should contribute the SAME
# total weighted mass per epoch (proportional to nothing but its own count of
# positions). Compare decisive vs draw classes.
dec_freq = freq[dec].sum()/len(dec)
draw_freq = freq[draw].sum()/len(draw)
dec_wmass = wsum[dec].sum()/len(dec)
draw_wmass = wsum[draw].sum()/len(draw)
print(f"\nPer-GAME averages over {NB} batches:")
print(f"  decisive: sampled {dec_freq:.1f}x   total IS-weight mass {dec_wmass:.1f}")
print(f"  draw    : sampled {draw_freq:.2f}x  total IS-weight mass {draw_wmass:.2f}")
print(f"  sampling ratio decisive/draw = {dec_freq/draw_freq:.1f}x")
print(f"  IS-weighted-mass ratio decisive/draw = {dec_wmass/draw_wmass:.1f}x")
print(f"  => If IS fully corrected oversampling, the weighted-mass ratio would be ~1.0.")
print(f"     A ratio of {dec_wmass/draw_wmass:.1f} means decisive games still dominate the")
print(f"     EFFECTIVE (post-IS) gradient by that factor.")

# Also report what a CORRECT IS weight would have been for one example pair.
priorities = np.where(np.isfinite(rb._priorities)&(np.array(rb._priorities)>0), rb._priorities, 1.0)
p = priorities**cfg.per_alpha; p/=p.sum()
n_sub = int(round(cfg.batch_size*cfg.decisive_sample_frac)); B=cfg.batch_size
# pick a median-priority game in each class
di = dec[np.argsort(p[dec])[len(dec)//2]]
di2 = draw[np.argsort(p[draw])[len(draw)//2]]
sub_p = p[dec]/p[dec].sum(); comp_p=p[draw]/p[draw].sum()
P_actual_dec = (n_sub/B) * sub_p[np.where(dec==di)[0][0]]
P_actual_draw = ((B-n_sub)/B) * comp_p[np.where(draw==di2)[0][0]]
print(f"\nExample median-priority pair:")
print(f"  decisive game: P_actual={P_actual_dec:.5f}  draw game: P_actual={P_actual_draw:.6f}")
print(f"  TRUE prob ratio decisive/draw = {P_actual_dec/P_actual_draw:.1f}x oversampled")
w_correct_dec = (n*P_actual_dec)**(-beta)
w_correct_draw = (n*P_actual_draw)**(-beta)
print(f"  CORRECT IS w_dec/w_draw = {w_correct_dec/w_correct_draw:.3f} (should be <1, down-weighting decisive)")
# code's IS:
w_code_dec = (len(dec)*sub_p[np.where(dec==di)[0][0]])**(-beta)
w_code_draw = (len(draw)*comp_p[np.where(draw==di2)[0][0]])**(-beta)
print(f"  CODE'S  IS w_dec/w_draw = {w_code_dec/w_code_draw:.3f}")
