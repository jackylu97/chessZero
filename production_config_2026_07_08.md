# Production Run Configuration — 2026-07-08

Launch: `bash scripts/launch_production.sh` (knobs: `PAR_GAMES`, `N_GAMES`, `BUFFER`, `STEPS`, `RUN`).
Instruments: `bash scripts/prod_probes.sh <run-id>` in a second tmux (reward-precision probe + conversion diag at every 10k checkpoint, appended to `production_probes.log`).

Every setting below carries its evidence. Nothing is inherited by accident — that
principle cost us two runs to learn (strategy doc §11–13).

## Architecture — `chess_hybrid_xl` + from→to policy head (23.9M+ inference params)

| component | setting | evidence |
|---|---|---|
| body | d288, 6-layer attention (8 heads) in rep+dyn, 3 conv-SE stem blocks, 3-layer shared pred body | XL scale test: eval + SF-agreement records, no buffer overfit at 60k (§16). Wide-shallow (d384×4L) **lost** to this depth in the arch sweep — do not trade depth for width |
| **policy head** | **`from_to`** (relational bilinear from/to-square scores + underpromo branch) | arch sweep arm C: **+48% proxy MCTS conversion** (0.568 vs 0.383), +5pts policy top-1, reproducible (±0.02); codec-parity tested (`tests/test_arch_heads.py`) |
| reward head | 8 planes (`--reward-head-planes 8`) | the reward head is the search's in-tree mate detector and THE conversion-critical component (muted-reward diag: 0.20→0.048 on identical weights, §11); 1-plane was an afterthought |
| scalar heads | conv squeeze (default) | attention pooling (arm D) was a wash (0.378 vs 0.383) — not adopted |
| value head | WDL, 8 planes; moves-left 8 planes | prior campaign results |
| training memory | `--grad-checkpoint-attention` (exact math, ~25–30% slower steps) | fits batch-512 XL on 32GB; **drop it on ≥48GB** for free speed |

## Search & targets

| setting | value | evidence |
|---|---|---|
| algorithm | Plain Gumbel MuZero, m=16, root noise on in self-play | improvement guarantee; deterministic evals; validated across all arms |
| **sims** | **400 — hard floor. Do not lower.** | 200-sim training regressed conversion 0.107→0.047 while general play soared; sweep equalized LOW (the net lost the skill, §16). **Standing rule: too little search fails compoundingly; too much costs linearly.** 800 is a legitimate upgrade if wall-clock affords it AFTER games/step is maxed |
| policy targets | π′ (completed-Q), zero-target opening plies | Gumbel paper; opening ε-mixture below |
| value targets | game z + fill + TB relabel, `--tb-value-hard` (one-hot at TB certainty) | hard-z exonerated by the near-mate gradient probe (§12); fixes the ±0.88/±1 inversion |
| sampling | uniform (`--per-alpha 0`) | 3-way precedent consensus + protects demo channels. NOTE: PER in the old production stack was accidentally load-bearing (decisive concentration) — its jobs are now done deliberately by fill/anchor/mixture (§16 add.2) |

## Data channels (the conversion loop's fuel — all load-bearing, §15)

| channel | setting | role |
|---|---|---|
| Stockfish warmstart | 30k steps, rolling 300-game pool, injection 300/256 | foundation; XL's doubled warmstart produced the strongest imitation phase measured |
| TB anchor | 64 games / 256 steps, `--anchor-max-size 1024` | **feeds the reward head's mate-example density** — the minimal arm proved that without it the beacon channel stays dark and conversion sits at the value-only floor (0.048) |
| rollout fill | on | truth-restoring z + on-distribution demos; the measured draw-basin fix |
| TB relabels | value 1.0 (hard), DTM 1.0, **policy 0.5→0.2 (DAgger)** | oracle labels at learner-visited states (O(T²)→O(T)) |
| endgame seeds | frac 0.30, resign-exempt, **DTM curriculum** (8→100 over 50%) | practice channel + the honest in-run conversion metric |
| batch mixture | declarative schedule: warmup 0.70/0.30/0 → 0.40/0.15/0.45 (10%) → 0.20/0.10/0.70 (40%) → 0.10/0.10/0.80 (60%) | composition is now CHOSEN (task #19); SF-dominant warmup per 2026-07-06 decision (was an emergent 40/60 anchor-heavy accident) |
| opening ε-mixture | mean 6 plies, 15% uniform floor, policy-sampled @T1.5 | diversity without training toward random moves |

## Scale & schedule

- `STEPS=600000` default (adjust to the week at observed it/s), LR milestones fractional (auto-scale), warmup 30k
- **Games/step is the underrated lever**: 1024 parallel games on ≥48GB is a true replay-ratio reduction (passes-per-position ~5.7 @512/5120 → target ~3; scale `BUFFER` with games/round)
- NOT included (deliberately): symmetry augmentation (parity bug fixed a0fd5af but shelved per user), attention-pooled heads (wash), material search utility (parked, task #15), sibling-zero reward contrast targets (parked; re-open if reward precision stalls — see tripwires)

## Tripwires (check `production_probes.log` + TensorBoard daily)

1. **Reward precision** (per 10k): false-fire should mature toward ≤0.12 (control's path 0.38→0.12). Rising or stuck >0.25 by mid-run → re-open sibling-zero contrast targets (§12 design, built-to-spec pending)
2. **Conversion diag** (per 10k, Gumbel@400): compare against control 0.152 / v3 0.160-long / XL-at-60k 0.047 (the sims casualty). Should climb steadily once past ~40k
3. **SF-agreement** (`loss/policy_loss_warmstart`): the only cross-run-comparable net-quality metric and the big-model-overfit tripwire. Records: 0.814 (XL). Sustained rise = buffer memorization → raise BUFFER
4. **`batch/frac_*` scalars**: must match the mixture schedule — divergence means channel starvation
5. **eval-vs-random saturates ~70%+**: draws = unconverted wins at that point. KNOWN GAP: no fixed-engine eval ladder is built; treat vs-random as a weak signal late in the run and lean on the diag
6. **grad_norm spikes / AMP staircase**: isolated = fine (clipped); repeated = investigate batch composition

## Known gaps accepted at launch

- Eval ladder (fixed Stockfish-limited opponent) not built — top backlog item
- A/B/A sims-recovery leg unfinished (XL paused at 68k when the GPU was ceded to the arch sweep); the 400-floor stands on §16 evidence regardless
- from→to head has proxy + smoke validation but no full self-play run yet — it is the boldest single element of this config; its tripwire is the conversion diag at 40–50k vs the v3 trajectory
