# Consolidated Mechanistic Verdict — Draw Collapse (2026-06-19)

Four multi-agent workflows (14 investigators + 4 prosecutors, ~3.5M subagent tokens) traced ACTUAL
values through every layer on real `cold2_pc` checkpoints/buffers. Both final prosecutors agree.

## Verdict: NO hidden code bug causes the collapse. It is the target distribution. Scaling won't fix it.

### Cleared with hard evidence (every layer traced on real data)
MCTS math / PUCT consts / negamax backup / MinMax normalization · WDL targets + POV/parity · scalar
transforms · K-unroll alignment · reward convention · **gradient flow** (value head healthy, grad reaches
every param, fits to floor) · loss/weighting · PER priorities · **serialization** (byte-exact, except the
ep-hash bug below) · **obs/history-stacking** (GPU↔CPU byte-identical) · **fp16** (inference-only; training
is fp32-master AMP) · **GPU self-play** path · **conv-policy action encoding** · reference-impl diff
(LightZero/muzero-general/EfficientZero/Lc0). The strongest prior candidate (repr-path-retains-signal vs
dyn-path-loses) was **empirically FALSIFIED** — an OOD artifact of extreme constructed FENs; on real
positions both paths are at chance from step 1000.

### The hard truth (sharper than before)
The value head is at **CHANCE on real positions — within AND across**: GT-rank acc 0.53 (chance),
corr-to-Stockfish +0.16 with **sign_acc 0.31 (below chance)**, from step 1000. The old "0.85 across-position
calibration" was on extreme/constructed positions; on real self-play positions the head **cannot tell who is
winning.** This is a LABEL-RESOLUTION limit, not capacity or convergence.

### Scale-up verdict
**It hits the same wall.** Value head at-chance at 1k/6k/30k (label limit) · more sims HURT (childV sibling-std
collapses ~25× from 25→400 sims) · training LONGER makes it worse (value/mae rises after 25k; calibration
+0.07→−0.16; eff_rank 240→135; decisive games 99→5). More compute reproduces the basin at larger scale.

## Real defects found (your bug instinct partly vindicated — real, but NOT the cause)
1. **`decisive_sample_frac=0.5` overfit** *(smoking gun, both workflows)* — decisive pool collapses 99→5-13
   games, but 50% of every batch is drawn from them; stratified IS weights (1.0/1.0) don't correct the ~150×
   oversampling → value head overfits ~11 trajectories (pred-V std 0.57 on decisive batch vs 0.16 random;
   value/mae *rises* late). Compounded by `decisive_retention_multiplier=1.0`. `replay_buffer.py:711-743`.
2. **`repetition_penalty` ply-ramp imprint** *(smoking gun)* — at 0.35 it's the SOLE target variance in 99.3%
   draws, a pure function of plies-to-end (sibling-blind), and the head provably learns it (corr V↔plies 0.40,
   corr V↔material −0.1). The anti-draw lever teaches "how soon the draw comes," not board value.
3. **ep-hash rep undercount** *(real code bug)* — GPU Zobrist XORs the en-passant file whenever `state.ep≥0`,
   but FIDE/python-chess count ep only when *legal* → rep planes 19/20 + threefold detection off by one ply.
4. **Unmasked leaf expansion** *(reference deviation)* — leaves expand the full 4672-action softmax → illegal
   moves enter the tree, dynamics queried on illegal transitions (99% illegal for chess). `mcts.py:176`, `tensor_mcts.py:~1276`.
5. *(already fixed `ba9a97c`: `selfplay_q_ratio→0`, `repetition_penalty` STM-guard.)*

## The one missing thing
A decisive, **position-accurate value signal in the targets.** The self-play value target is the MC outcome
under a policy that can't convert, so won positions are played to draws and labeled ≈draw. The fix is
**external decisive supervision**: (a) win-adjudication *(previously vetoed)* or (b) persistent Stockfish
per-position value targets (the `external_values` path / warm anchor). NOT more compute, NOT a code fix.
