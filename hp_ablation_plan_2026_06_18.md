# Self-Play Improvement — Hyperparameter Ablation Plan

**Started 2026-06-18 · revised 2026-06-19 for the cold-start substrate.**
**Status:** EXECUTING — cold-start positive control launched.
**Question:** *Why does the model not improve during self-play, and which single hyperparameter
makes it start?* We isolate the **self-play mechanism** on a **cold-start `chess_small`** substrate
(no Stockfish distillation, no anchor) so a positive result is confound-free and generalizes to warm-start.

Grounded against the actual `src/config.py`, `replay_buffer.py`, `trainer.py`, `mcts.py`/`tensor_mcts.py`,
`scripts/train.py`; stress-tested by an adversarial design critic. Corrections to earlier assumptions are flagged inline.

---

## 0. Decision log (2026-06-19)

| Decision | Rationale |
|---|---|
| **Substrate = cold-start `chess_small`** (not warm-start). | Tests the self-play loop in isolation: no distillation residue, no anchor, no "S was distilled under old HPs" confound. A lever that makes cold self-play climb is confound-free proof and generalizes to the easier warm regime. `warm_cs` keeps running and is **banked as the warm reference** — conclusions drawn later. |
| **Primary metric = `probe_sibling_ranking` trajectory** (gauntlet for survivors). | Lower variance, directly reads the broken quantity. See §4. |
| **Positive control FIRST, before any ladder.** | Cold start has an *asymmetric null* — a flat arm could be "lever failed" or "compute-starved." Must first prove the loop climbs at all in budget. *Launched:* `2026_06_19_cold_posctrl`. |
| **Reorder the ladder for the cold regime.** | The draw-basin pathology (reaches won position → can't convert → shuffles to draw) needs a policy good enough to *reach* won positions. A near-random cold net fails everywhere, so `repetition_penalty` is a **late-regime** lever — deferred. Lead with target-**shape** levers that act from step 1. |
| **`value_head_init_std = 0.01`** in the cold baseline. | Default 0.0 zero-inits the value-head last linear → blocks body gradient through the head at cold start. 0.01 lets gradient flow without dominating. Itself becomes an ablatable lever. |
| **`draw_score` in scope**, with eval-config pinning. | Top non-self-referential lever; behavioral, so pin `draw_score`/`selfplay_q_ratio` identical across arm & reference at eval time. |
| **Add a dedicated LR/horizon arm.** | At resume/short horizons LR sits at the 1e-3 peak (milestones at 0.5/0.75·`training_steps`). Rule out "too few peak-LR steps" before calling any HP inert. |
| **Added 5 CLI flags** (were config-only): `--draw-score`, `--eval-to-wdl-alpha`, `--eval-to-wdl-beta`, `--decisive-sample-frac`, `--reanalyze-interval`. | Every arm is now a clean one-flag change. *(committed)* |

---

## 1. What is already settled — do NOT re-test

| Finding | Evidence | Implication |
|---|---|---|
| **Root cause = V^π ≈ draw.** Self-play value target is ~90% the game *outcome* under a policy that can't convert. | `probe_value_target_audit.py`: SF>+3 positions get z=+0.18, 67% draws; root_value corr-to-SF 0.47 vs outcome z 0.18. | Fix must inject decisive value signal. |
| **Conv policy head is a real win** (+58/+68 Elo). | Head-to-head gauntlets. | Keep `policy_head_type=conv` (default). Not an ablation target. |
| **moveshead PLATEAUED by ~30k** (−2 Elo / 42k steps, 900-game gauntlet). | `match_checkpoints.py`. | More steps in the same regime do nothing. |
| **Capacity is NOT the bottleneck.** | Bigger value head can't fix a target with no resolution. | `chess_small` (5.5M) is a valid, representative substrate. |
| ⚠️ **"Search levers falsified" was measured at WARM self-play checkpoints** (miscalibrated value): more sims HURT, dirichlet nothing, temperature scatters. | `probe_sim_scaling.py` + sweeps. | **Do NOT assume these hold at cold start.** Cold dynamics differ — `num_simulations`/temperature are *real Tier-2 arms* on this substrate, not foregone negative controls. |

**Vetoed (do not propose):** win-adjudication, leaf-legal-masking. **Constraint:** only threefold + 75-move
draws may be penalized — never stalemate / insufficient-material / computer draws.

---

## 2. Verified mechanics

**Self-play value target** (`replay_buffer._wdl_target_at`, WDL head — the only path on cold start since no `external_values`):
```
legacy = outcome_onehot(ply)                    # STM-relative [W,D,L]
# threefold/no-progress draws only: legacy = [0, 1-d, d], d = repetition_penalty · decay**plies_to_end
target = (1 - selfplay_q_ratio)·legacy + selfplay_q_ratio·eval_to_wdl(root_value, alpha, beta)
```
At `selfplay_q_ratio=0.1` the target is 90% the outcome. `eval_to_wdl(·, alpha=4, beta=2)` is the
scalar→WDL logistic (sharper alpha / smaller beta = narrower draw zone = more decisive targets).
The WDL→scalar value MCTS optimizes is `V = P(W) − P(L) + draw_score·P(D)` (`draw_score=−0.05`).

**Cold-start gating** (`trainer.py:212-230`): `chess_small` default `self_play_warmup_steps=0` + no
injection shards → `selfplay_on = not bool([]) = True` from step 0. The buffer seeds via self-play to
`min_buffer_size=500` before training. ✅ Confirmed in cold_pc log ("falling back to self-play to seed the buffer").
*(If you ever add injection to a cold run, you MUST also set `--self-play-warmup-steps>0` or self-play silently turns off — trainer.py:230.)*

**Checkpoints / paths — footgun:** the launch header prints `args.game` (`chess_small`) but the real dir is
`config.game` = **`"chess"`**. So files land in **`checkpoints/chess/<run-id>/`** and TB in **`runs/chess/<run-id>/`**
regardless of the header. `checkpoint_interval=1000`. `.buf` is skipped until self-play games exist.
`match_checkpoints.py` needs `--game chess_small` for sizing but the **path** is under `checkpoints/chess/`.

**Flag reality** (now): real flags — `--selfplay-q-ratio`, `--draw-score`, `--eval-to-wdl-alpha/-beta`,
`--decisive-sample-frac`, `--reanalyze-interval`, `--repetition-penalty[-decay/-window]`,
`--num-simulations`, `--dirichlet-alpha`, `--temperature-drop-step`, `--value-head-init-std`,
`--decisive-retention-multiplier`, `--no-moves-left`, `--no-consistency-loss`. Still config-only:
`temperature_init`/`schedule`, `lr`/`lr_warmup_steps`, `num_unroll_steps`, `batch_size`, `per_*`,
`replay_buffer_size`, proj/pred dims, `value_loss_weight_*`. Note `value_loss_weight` is **effectively 1.0**
(phase splits both 1.0), not the 0.25 I'd assumed.

---

## 3. The cold-start HP ladder (reordered)

Tiers by expected leverage on the cold self-play loop. Each arm is a **fresh cold run** with one HP changed
from the positive control (the HP shapes learning from step 0 — the correct cold-start design).

### Tier 1 — target-shape & gradient-flow levers that act from step 1

| HP | Arm | Mechanism | Flag |
|---|---|---|---|
| **value_head_init_std** | `0.01 → 0.0` (and `→ 0.05`) | Does zero-init actually block the value head's gradient at cold start? Brackets the baseline choice and tests a known cold-start failure mode directly. | `--value-head-init-std` |
| **selfplay_q_ratio** | `0.1 → 0.5` | Blends 50% of the MCTS root_value (the search verdict / TD bootstrap) into the target vs 10%. Standard MuZero bootstrapping; tests whether more search-verdict signal vs raw outcome accelerates the loop. *Self-referential caveat is weaker at cold start (no pre-baked draw bias to amplify) but still watch for basin-deepening as it learns.* | `--selfplay-q-ratio` |
| **draw_score** | `−0.05 → −0.20` (pin eval value) | Subtracts the draw mass squashing positions to V≈0 so a decisive line out-ranks a shuffle in PUCT from the start. | `--draw-score` |
| **eval_to_wdl_alpha / beta** | `alpha 4→8` (and/or `beta 2→1`) | Sharpens scalar→WDL so the blended-in target component is decisive, not draw-flat. | `--eval-to-wdl-alpha` / `-beta` |

### Tier 2 — exploration, data balance, optimization horizon (NOT pre-falsified on cold start)

| HP | Arm | Mechanism | Flag |
|---|---|---|---|
| **decisive_sample_frac** | `0.5 → 0.8` | Cold games are decisive-rich; up-weight them so the value head learns W/L structure early. | `--decisive-sample-frac` |
| **num_simulations** | `200 → 400` | At cold start more search may *help* (unlike the warm checkpoint). Real arm, not a control. | `--num-simulations` |
| **LR / horizon** | `--steps 10000` (milestones fire at 5k/7.5k → real anneal) **or** lower `lr` | Rules out "constant peak LR for the whole window" as the reason nothing moves. | `--steps` (+ config `lr`) |
| **temperature window** | `--temperature-drop-step 60` | Wider exploratory window → more diverse positions feeding the buffer. | `--temperature-drop-step` |

### Tier 3 — late-regime / self-referential (defer until the net reaches convertible positions)

`repetition_penalty 0.35→0.7` (only fires on won-but-shuffled threefold/no-progress draws — inert until
the policy can reach won endgames) · `reanalyze_interval 1024→512` (self-referential; revisit once value recalibrates) ·
`--no-consistency-loss` / `--no-moves-left` (free backbone capacity for the value head — diagnostic).

### Removed for cold start (belong to the banked warm_cs reference)
`warmstart_sample_frac`, `stockfish_injection_*`, `warmstart_q_ratio` — no Stockfish anchor exists in a cold run.

---

## 4. Measurement

**Primary — `probe_sibling_ranking.py --checkpoint` trajectory.** Within-position Spearman(value, SF)
over a node's legal moves — *the* broken quantity (calibrated across positions ~0.85, near-zero within).
Run on checkpoints at 2k/4k/…/Nk; the signal is *is the curve rising, and faster than the positive control's?*
Low variance, reads the mechanism directly, sensitive at short horizons (value calibration moves long before strength).
**Caveat:** it's a proxy — validate it (control C1) by confirming it shows the known +58-Elo conv-vs-flat gap.

**Secondary:** built-in repr probe (`r2_eval` across-position calibration, logged each `eval_interval`) trajectory;
`probe_value_target_audit.py --buf` for target quality.

**Strength (cold start makes this usable):** cold games are decisive-rich → lower draw rate → tighter Elo CI
than the 88%-draw warm regime. Use `match_checkpoints.py --game chess_small`:
(a) **beat-past-self** (arm@Nk vs arm@earlier) — the AlphaZero yardstick; (b) **arm@Nk vs posctrl@Nk** matched-step.
**900-game gauntlet** reserved for the 2–3 arms that clear the trajectory screen.

**Gate:** the positive control must show a **rising** sibling-corr / decisive-rate / beat-past-self trajectory
before the ladder launches. If flat at the first horizon, extend before concluding anything — never run the ladder blind.

---

## 5. Run order & command template

Each arm = a supervised cold run, one HP changed. Checkpoints → `checkpoints/chess/<run-id>/`
(config.game=="chess"; the header misprints "chess_small"). Self-play is CPU-single-core
launch-bound → ~3–4 arms run concurrently, but training contends on the GPU so concurrency gives
~**2× aggregate throughput**, not 4× (measured: 4 runs → SM ~86%, ~3 steps/s each vs ~6 solo).

**Baseline flags — EVERY run incl. all arms:**
`--game chess_small --ckpt-game chess --device cuda --steps 30000 --eval-interval 2000`
`--value-head-init-std 0.01 --tensor-mcts-compile-net --mask-illegal-policy`

```
tmux new-session -d -s <ARM> -c /workspace/chessZero \
  "scripts/supervise_train.sh <baseline flags> --run-id 2026_06_19_<ARM> \
   --train-log logs/2026_06_19_<ARM>.log  <ONE HP OVERRIDE>"
```

> The FIRST batch (cold_pc/qr05/draw020/wdl8, launched 02:58) lacks `--mask-illegal-policy`
> (re-added after — it was the prior baseline: illegal-mass penalty + makes the metric real,
> which is otherwise hardcoded 0 when masking is off). Batch 1 is internally consistent
> (arm-vs-PC valid); all FUTURE batches include it. `--tensor-mcts-compile-net` = the 1.4×
> net-compile + engages the Triton MLH kernel.

| # | Arm | Override | Status |
|---|---|---|---|
| **PC** | **positive control / baseline** | *(none)* — `--value-head-init-std 0.01` only | ✅ launched `2026_06_19_cold_posctrl` |
| C1 | proxy-validation | measure conv-vs-flat known +58 Elo through `probe_sibling_ranking` | todo |
| A1 | `vhinit0` / `vhinit05` | `--value-head-init-std 0.0` / `0.05` | gated on PC climbing |
| A2 | `qratio05` | `--selfplay-q-ratio 0.5` | " |
| A3 | `drawscore020` | `--draw-score -0.20` (pin eval) | " |
| A4 | `wdlsharp8` | `--eval-to-wdl-alpha 8` | " |
| B1 | `decisive08` | `--decisive-sample-frac 0.8` | Tier 2 |
| B2 | `sims400` | `--num-simulations 400` | Tier 2 |
| B3 | `lr10k` | `--steps 10000` (anneal) | Tier 2 |
| — | (Tier 3 deferred) | `repetition_penalty`, `reanalyze_interval`, aux-loss | after net reaches convertible positions |

Per-arm wall-clock ≈ a full cold run to the horizon (self-play ~600 s/round dominates — measured on warm_cs;
the positive control pins the real cold figure). Run Tier-1 (A1–A4) concurrently first; decide from trajectories.

---

## 6. Open items / next checkpoints

1. **Positive control trajectory** — first read at the first few checkpoints (sibling-corr, decisive-rate, loss). Decide horizon + whether the loop climbs.
2. **Build/confirm the trajectory harness** — wrap `probe_sibling_ranking` over a checkpoint series + the matched-step `match_checkpoints` call.
3. **(deferred)** fix the misleading checkpoint-dir header (`args.game` → `config.game`).
4. Revisit Tier-3 (`repetition_penalty`) once the positive control shows the policy reaching/converting advantages.
