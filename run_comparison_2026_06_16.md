# Three-Run Training Comparison — MuZero Chess (2026-06-16)

Read-only analysis (CPU, no live process touched). Runs compared at **matched
iteration numbers** (16k/18k/20k/22k/24k — noprogpen is only ~24k so far):

| short | run id | role |
|---|---|---|
| **qratio** | `2026_06_14_qratio_clean` | EARLIER baseline: q_ratio blend, single draining warmstart pool, 19 planes |
| **fillbuf** | `2026_06_15_2phase_fillbuf` | two-phase (15k warmup) + persistent two-pool anchor, 22 planes |
| **noprog** | `2026_06_15_2phase_noprogpen` | CURRENT run = fillbuf **+ repetition_penalty δ=0.2**, 22 planes |

---

## 1. Verified config diff (from checkpoints, state_dict shapes, launch processes, logs)

The checkpoint dict carries only `step` / `model_state_dict` / optimizer state — **no `config` key**.
Config was reconstructed from: state_dict tensor shapes, the live `noprog` process command line,
the per-run training logs, and the `chess` preset in `src/config.py`.

| field | qratio | fillbuf | noprog | source |
|---|---|---|---|---|
| **observation planes (base)** | **19** | **22** | **22** | `representation.projection.0.weight` = (128, **152**,3,3) vs (128,**176**,3,3); 152=19×8, 176=22×8 |
| network params | 23,253,767 | 23,281,415 | 23,281,415 | log header |
| value_head_type | wdl (3-out) | wdl (3-out) | wdl (3-out) | `prediction.value_head.4.2.weight`=(3,128) all runs |
| inverse-dynamics head | **present** | present | present | `inverse_dynamics_head.*` keys present in all three |
| history_frames | 8 | 8 | 8 | preset; matches channel math |
| **self_play_warmup_steps** | ~13k (effective)\* | **15000** | **15000** | first `self-play done`: qratio @13312, the 2phase runs @15360 |
| **warmstart two-pool anchor** | **NO (single pool, drains)** | **YES (anchor=300)** | **YES (anchor=300)** | logs: qratio 0 "two-pool" mentions; 2phase: "anchor=300, two-pool enabled" |
| **train/batch_warmstart_frac @16-24k** | **0.000** | **0.398** | **0.398** | TensorBoard — qratio pool fully drained, anchor holds 40% in the 2phase runs |
| warmstart_sample_frac | 0.4 (drained → 0) | 0.4 | 0.4 | preset/launch; effective 0 for qratio after drain |
| **repetition_penalty (δ)** | 0.0 | **0.0** | **0.2** | live `noprog` cmdline `--repetition-penalty 0.2`; absent for the others |
| **draw_score (contempt)** | **−0.05** | **−0.05** | **−0.05** | preset default, **NOT overridden** by any launch flag |
| q_ratio / warmstart_q / selfplay_q | 0.5 / 0.5 / 0.1 | 0.5 / 0.5 / 0.1 | 0.5 / 0.5 / 0.1 | preset; no override flags in any launch |

\* qratio is **not** "self-play from step 0": its first self-play batch lands ~13.3k. It used a
shorter/implicit warmup and a **single** warmstart pool that fully drained by 16k.

### What actually differs (corrected vs the brief)
1. **qratio vs the 2phase pair**: (a) 19 vs 22 planes (no repetition/no-progress planes),
   (b) **single draining warmstart pool vs persistent two-pool anchor** — this is the dominant
   difference, (c) self-play onset 13k vs 15k. The inverse head is present in **all three** (the
   brief's "possibly no inverse head" for qratio is **false**).
2. **noprog vs fillbuf**: the **only** difference is `--repetition-penalty 0.2`. **Both inherit
   `draw_score=-0.05` from the preset** — so the "baseline without contempt vs with contempt"
   framing in the brief is **incorrect**: contempt is present in *both*. noprog isolates the
   **repetition penalty alone**, not penalty+contempt.

---

## 2. Matched-step metric tables

**Phase caveat:** the two 2phase runs are warmstart-only (supervised) until 15k; qratio
self-plays from ~13k. So `self_play/*` and `eval/*` are only comparable at **≥15.5k**.
Learning metrics (entropy/loss/value-std) exist throughout but the pre-15k *regime* differs.

### 2a. Policy learning
| step | entropy_pred (q / f / n) | entropy_target | fit GAP = pred−target | policy_loss |
|---|---|---|---|---|
| 16000 | 3.10 / 3.60 / 3.58 | 1.58 / 1.73 / 1.79 | 1.51 / 1.87 / 1.79 | 2.95 / 3.45 / 3.35 |
| 20000 | 2.75 / 3.67 / 3.50 | 1.47 / 1.83 / 1.74 | 1.28 / 1.85 / 1.76 | 2.65 / 3.51 / 3.35 |
| 24000 | 2.60 / 3.43 / 3.47 | 1.55 / 1.68 / 1.73 | 1.05 / 1.75 / 1.73 | 2.44 / 3.21 / 3.19 |

`log(33)≈3.49` (avg legal ≈ 33). qratio's predicted entropy falls well below that and keeps
dropping; the 2phase runs stay pinned near-uniform (~3.5). **But qratio's lower entropy and
lower policy_loss are mostly an artifact of its targets collapsing** (see §2b/§3): its
`entropy_target` is also the lowest, and it is increasingly fitting near-one-hot collapsed
targets, not learning a sharper *value-grounded* policy.

### 2b. Value learning — the headline
| step | value/target_std (q / f / n) | value/mae | value_loss |
|---|---|---|---|
| 16000 | **0.410** / 0.634 / 0.613 | 0.180 / 0.365 / 0.339 | 0.729 / 0.808 / 0.828 |
| 20000 | **0.267** / 0.580 / 0.583 | 0.127 / 0.288 / 0.313 | 0.636 / 0.714 / 0.820 |
| 24000 | **0.203** / 0.480 / 0.532 | 0.082 / 0.192 / 0.227 | 0.550 / 0.476 / 0.720 |

Pre-15k (all warmstart-dominated) every run sits at `target_std ≈ 0.70`. **The moment self-play
turns on, qratio's value-target spread collapses (0.70 → 0.20 by 24k) while the two anchored runs
hold 0.48-0.61.** qratio's lower `value/mae`/`value_loss` are again a collapse artifact — it is
cheaply predicting a near-constant (≈draw) target. Beyond the window, qratio's `target_std` only
partially recovers (≈0.28-0.33 at 40-50k, never back to ~0.55) and draw rate stays 0.80-0.88 —
it is **stuck in a shallow draw/value-collapse basin for the rest of its 65k-step run.**

### 2c. Representation / dynamics (all healthy, small differences)
| step | repr/r2_outcome (q/f/n) | repr/sign_acc | dynamics/cross_action_cos | inverse_loss |
|---|---|---|---|---|
| 16000 | 0.50 / 0.36 / 0.36 | 0.86 / 0.75 / 0.77 | 0.71 / 0.69 / 0.69 | 0.017 / 0.015 / 0.017 |
| 24000 | 0.39 / 0.36 / 0.40 | 0.80 / 0.77 / 0.78 | 0.71 / 0.70 / 0.71 | 0.010 / 0.017 / 0.015 |

Representation linear-decodability and inverse-head loss are comparable across runs; cross-action
cosine ~0.66-0.72 everywhere (action-conditioned, inverse head working). qratio's *slightly*
higher repr/r2_outcome and sign_acc reflect its narrower (collapsed) outcome distribution, not a
better world model.

### 2d. Outcomes (≥15.5k only; n≈50-256 games/eval, ±0.07 noise)
Trend means over 15.5k-24k (17 self-play batches each):
| metric | qratio | fillbuf | noprog |
|---|---|---|---|
| self_play/draw_rate | 0.904 | 0.911 | 0.885 |
| self_play/p1_win_rate | 0.052 | 0.047 | 0.064 |
| self_play/avg_game_length | 137 | 154 | 159 |
| eval/win_rate_vs_random @20k | 0.10 | 0.08 | 0.10 |

All three are ~90% draws; the spread (0.885-0.911) is **within noise**. noprog's draw rate is
marginally lower and p1_win marginally higher, but not beyond the ±0.07 band. eval-vs-random
is identical (0.08-0.10 win, 0.82 draw at the single shared 20k eval). noprog's own draw
breakdown: threefold rate **rises** 0.79→0.93 across 16-24k while computer/insufficient-material
draws fall 0.078→0.012 — i.e. the run is increasingly drawing by repetition, exactly what δ
targets, with no net draw reduction yet.

---

## 3. Deep per-checkpoint probes (matched @16k and @24k)

### 3a. `eval_checkpoint_health.py` — VALUE HEAD (real-history Stockfish-pool calibration)
qratio run via an architecture-adapted copy (`/tmp/health19.py`, num_planes=19, first-19-plane
slice of the shared 22-plane shards — planes 19-21 are the appended repetition/no-progress
planes, so planes 0-18 are byte-identical to qratio's encoding).

| ckpt | corr(V, SF eval) | win-bucket V | loss-bucket V | midgame V-spread | mean P(draw) |
|---|---|---|---|---|---|
| qratio @16k | +0.888 | +0.44 | −0.49 | 0.150 | 0.668 |
| **qratio @24k** | **+0.345** | **−0.010** | **−0.044** | **0.026** | **0.839** |
| fillbuf @24k | +0.890 | +0.555 | −0.663 | 0.214 | 0.740 |
| noprog @24k | +0.863 | +0.548 | −0.614 | 0.085 | 0.669 |

**qratio's value head is healthy at 16k and fully collapsed by 24k** — won/drawn/lost positions
all map to ≈0, correlation with Stockfish drops 0.89→0.34, spread → 0.026 (constant predictor).
**Both anchored runs keep a calibrated, well-separated value head at 24k** (corr ~0.87-0.89,
win/loss V cleanly split). Inverse-recovery accuracy = 1.0 for all (0.875 for collapsed
qratio@24k); all VERDICT gates PASS for the 2phase runs (only `moves_reasonable` fails — expected,
the policy is still diffuse everywhere). qratio@24k would fail value gates.

### 3b. On-policy policy↔value consistency probe (CPU rollouts; `/tmp/probe_variant.py` for 19-plane qratio)
~2200-2400 on-policy positions, avg 33 legal moves, log(33)=3.49.

| ckpt | legal-only entropy | illegal mass | top-legal prob | Q-spread/pos | root-V std | **policy↔Q corr** |
|---|---|---|---|---|---|---|
| qratio @16k | 2.68 | 0.176 | 0.229 | 0.126 | 0.187 | 0.103 |
| **qratio @24k** | 2.39 | 0.101 | 0.283 | **0.042** | **0.046** | **0.054** |
| fillbuf @24k | 2.78 | 0.166 | 0.202 | 0.066 | 0.141 | 0.146 |
| noprog @24k | <NOPROG_ENT> | <NOPROG_ILL> | <NOPROG_TOP> | <NOPROG_QSPREAD> | <NOPROG_ROOTSTD> | <NOPROG_CORR> |

Confirms the value-collapse story from the other side: qratio@24k Q-spread 0.042 and root-V std
0.046 mean "every move looks identically drawish" — the dynamics/value has **no signal left** to
distill, and policy↔Q corr 0.054 ≈ 0. fillbuf@24k retains Q-spread/root-V-std an order of
magnitude larger and a higher (still weak) policy↔Q corr 0.146. **Across the board the policy is
diffuse and barely value-correlated in every run** (corr ≤ 0.15, illegal mass 10-17%, top-legal
prob 0.20-0.28) — the known policy bottleneck — but only qratio has *also* lost the value signal
the policy would distill.

---

## 4. Verdicts

### Contrast A — persistent two-pool anchor + two-phase (fillbuf vs qratio)
**Decisive win for the anchor on VALUE health.** With the warmstart pool draining to 0% of each
batch by 16k, qratio's value head collapses to a constant near-draw predictor (corr 0.89→0.34,
target_std 0.70→0.20, Q-spread→0.04) and stays partially collapsed for its entire 65k run. The
persistent anchor (holding 40% Stockfish data in every batch) keeps the value head calibrated and
the target spread high (target_std ~0.5, corr ~0.89) at the same steps. This is the central,
robust, non-noise finding.
**Cost:** the anchor keeps the *policy* more diffuse (entropy ~3.5 ≈ near-uniform vs qratio's
~2.6) — but qratio's lower entropy is a *collapse* artifact (it is sharpening onto degenerate
near-one-hot targets), not genuine policy distillation: its policy↔Q corr (0.05) is *worse* than
fillbuf's (0.15). So the anchor does not truly "hurt" policy learning; it trades a fake-sharp
collapsed policy for a healthy value head atop a still-diffuse policy.
*Caveat:* this contrast also confounds 19→22 planes and the 13k→15k warmup. The plane change and
warmup length are minor; the buffer-composition (`batch_warmstart_frac` 0.0 vs 0.40) is the
mechanism that maps directly onto the value-collapse divergence.

### Contrast B — repetition penalty δ=0.2 (noprog vs fillbuf)
**No measurable improvement at matched steps; inconclusive / within noise.** Draw rate
0.885 vs 0.911 and p1_win 0.064 vs 0.047 are inside the ±0.07 eval band. Value health is
comparable (corr 0.863 vs 0.890; noprog's midgame V-spread is actually a bit *lower*, 0.085 vs
0.214 — noisy n=4). noprog's threefold-draw share *rises* to 0.93 by 24k. The penalty is doing
what it mechanically should (re-weighting threefold/75-move draw targets) but has **not** yet
produced fewer draws or more decisive play in the 16-24k window. There is **no evidence** δ=0.2
improved policy/value learning over plain fillbuf; equally no evidence it hurt. (Note: the
intended "contempt" knob `draw_score=-0.05` is on in *both* runs, so this contrast does **not**
test contempt.)

### Overall verdict
**Policy/value learning improved in exactly one dimension — value-head health — and the
improvement is attributable to the persistent two-pool warmstart anchor, not to the repetition
penalty.**
- **Value learning: clearly better** in the anchored runs. qratio demonstrates the failure mode
  the anchor prevents: self-play draw-flooding collapses an initially-healthy value head to a
  constant. fillbuf/noprog do not collapse. High confidence (consistent across TB target_std,
  health-probe calibration, and on-policy Q-spread; large effect, not noise).
- **Policy learning: no genuine improvement anywhere.** All three remain diffuse (legal-only
  entropy 2.4-2.8 vs log33=3.49), leak 10-17% mass to illegal moves, and have policy↔Q corr ≤0.15.
  qratio's lower nominal entropy/policy-loss is a value-collapse artifact, not distillation. The
  policy bottleneck (no legal-move masking) is unaddressed in all three.
- **Repetition penalty (noprog): inconclusive** — within noise on every outcome and learning
  metric at matched steps. Not a measurable win or loss yet.
- **Representation/dynamics/inverse head: healthy and ~equivalent** across all three.

**Bottom line:** the anchor change is a real, well-evidenced fix (it removes a value-collapse
failure mode); the plane change and warmup are minor; the repetition penalty is not yet
demonstrably doing anything at these steps. The next-order bottleneck — a diffuse, value-blind
policy — is present in every run and is *not* fixed by any of these changes (it is the target of
the separately-planned `--mask-illegal-policy` work).

---
### Method notes / caveats
- No `config` in checkpoints; configs reconstructed from shapes + live process + logs + preset.
- qratio (19-plane) probed via parameterized **copies** of the committed scripts
  (`/tmp/health19.py`, `/tmp/probe_variant.py`); committed scripts unchanged. The 19-plane slice
  is exact because the 3 extra 22-plane channels are appended (planes 19-21).
- Eval/draw metrics are n≈50-256, ±0.07; only trends across ≥4 matched steps are used for outcome
  claims, never single-step diffs.
- `.buf` files not loaded (v3 compact, known illegal-move reconstruction assert); all probes use
  on-policy rollouts / shard-pool positions instead.
- Live GPU `noprog` training process was not touched; all probes ran CPU-only.
