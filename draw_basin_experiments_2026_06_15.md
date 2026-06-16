# Draw-Basin Diagnostic Experiments — 2026-06-15

Run under study: `2026_06_15_2phase_qsplit` → `2026_06_15_2phase_fillbuf`
(22-plane, two-phase warmstart→self-play @15k, split q_ratio 0.5/0.1, 300-game
two-pool anchor, 128k Stockfish warmstart pool). Checkpoints at 29k and 55k–68k.

## TL;DR

The value head is **healthy** (calibration 0.89 on teacher positions), the
representation linearly decodes win/loss at AUC≈1.0, and the dynamics/inverse
heads are fine. Yet self-play is **~96% draws** (95% threefold) and **move
quality is degrading** over training. Five experiments converge on one cause:

> **The model cannot convert won positions (2% conversion rate). Because it
> draws everything, the value head learns Vᵖ ≈ draw on its own positions, MCTS
> finds no improvement gradient, and the policy is frozen at a self-reinforcing
> fixed point. This is a missing-signal problem, not a search or a bug.**

Search knobs (sims, draw_score) and opening randomization do **not** help.
The fix must inject decisive signal: conversion curriculum, draw/repetition
penalty, reward shaping, or value-target relabeling toward V*.

---

## Experiment A — Repetition-penalty precondition: are the draws winnable?

**Method.** 150 threefold-draw self-play games from the 55k buffer; at each
game's repetition-decision point (first `board.is_repetition(2)`), evaluate the
board with Stockfish depth 14, STM-POV.

**Result.**
| metric | value |
|---|---|
| mean \|eval\| | **743 cp** |
| median \|eval\| | 646 cp |
| \|eval\| < 50 (equal, correct draw) | **4%** |
| \|eval\| ≥ 150 (decisive, winnable) | **90%** |
| \|eval\| ≥ 300 (clearly winning) | 80% |
| signed eval (10/50/90 pct) | −693 / +390 / +1989 |

**Conclusion.** Repetition draws are overwhelmingly **winnable, not equal** — the
model shuffles away ~4–7 pawn advantages. Penalizing repetition is well-targeted
(punishes thrown-away wins, not correct draws). ✅ Precondition met.

---

## Experiment B — Opening randomization A/B (self-play only, 55k, 48 games/cell)

**Method.** Self-play on the 55k checkpoint at `random_opening_plies ∈ {0,8,16,32}`.

**Result.**
| rand_open | draw% | decisive% | 3fold% | avg_len |
|---|---|---|---|---|
| 0  | 90% | 10% | 85% | 106 |
| 8 (live) | 92% | 8% | 90% | 96 |
| 16 | 98% | 2% | 98% | 111 |
| 32 | 88% | 12% | 83% | 131 |

**Conclusion.** Draw rate is flat at 88–98% across the whole sweep — no trend.
Breaking opening symmetry does **not** reduce draws → draws are
**conversion-driven, not symmetry-driven**. ❌ Not a lever.

---

## Experiment C — Conversion probe (self-play from won positions, 55k)

**Method.** Seed self-play from 48 clearly-won Stockfish positions (STM eval
+0.6…+0.95, midgame); model plays both sides; measure whether the winning side wins.

**Result.**
| outcome | rate |
|---|---|
| **converted (winning side won)** | **2%** (1/48) |
| drew | 98% (47/48) |
| lost (blundered the win) | 0% |
| avg game length | 153 |

**Conclusion.** The model converts **2%** of clearly-won positions. 0% losses →
it doesn't hang the win, it simply can't make progress and shuffles to a draw.
This is the conversion deficit, near-total, quantified.

---

## Experiment D — Search-knob A/B (does more search help?), 55k

**Method.** Self-play 2×2: `num_simulations ∈ {200,800}` × `draw_score ∈ {−0.05,−0.30}`.

**Result.** Draw rate **92–100% across all four cells** — no effect from 4× the
simulations (literature uses ~800–1600) nor from a 6× stronger draw penalty.

**Conclusion.** Search is not the lever. With a flat value landscape, more sims
back up to the same ~0. ❌

---

## Experiment E — Move quality vs Stockfish over training (29k vs 68k)

**Method.** ACPL (avg centipawn loss vs Stockfish depth 12) + top-1 match +
blunder rate, 30 self-play games × 8 positions each, per checkpoint.

**Result.**
| checkpoint | ACPL | median | top1% | blunder%(>300) | good%(<50) |
|---|---|---|---|---|---|
| 29k | 386 | 250 | 7% | 45% | 20% |
| 68k | 443 | 377 | 2% | 57% | 10% |

**Conclusion.** Move quality **regressed** 29k→68k by every metric. (Absolute
ACPL is inflated by the won-but-shuffling context; the *direction* and the
style-agnostic eval-based metrics are the trustworthy part. top1-match partly
reflects drift from Stockfish *style*, so weight ACPL/blunders more.) The draw
loop is being reinforced into the policy over time.

---

## Experiment F — Value head on the model's OWN won positions (the capstone)

**Method.** 120 repetition-decision positions from self-play where Stockfish says
STM is clearly winning (>+3 pawns), reconstructed in-distribution with full
8-frame history, evaluated by the **model's own value head** (55k).

**Result.**
| | value |
|---|---|
| Stockfish eval (STM) | **+9.0 pawns** (clearly won) |
| Model value head (STM) | **V = +0.03** (says draw) |
| Model WDL | P(W)=0.04, **P(D)=0.95**, P(L)=0.01 |
| corr(model V, Stockfish) | **−0.03** (zero) |
| reads as ~draw (\|V\|<0.2) | 96% |

**Conclusion.** Up ~9 pawns, the value head outputs 95% draw, uncorrelated with
the truth. Reconciles the "healthy value head" paradox: the probe's 0.89 was on
**teacher** positions; on the model's **own** positions V predicts **Vᵖ = draw**.
Combined with the frozen-rep probe (win/loss decodable at AUC≈1.0), this is a
**labeling/target problem, not a representation problem** — the network knows
it's winning; the head, trained on draw outcomes, calls it a draw. ⇒ the fix is
value-target relabeling toward V* / decisive-signal generation, NOT search.

## Synthesis — why MCTS doesn't fix this, and why it's a fixed point

MCTS is a policy-*improvement* operator: it amplifies the value signal over the
reachable search horizon; it does not create one. The trap:

1. The value head is calibrated to **Vᵖ** (outcome under the *current* policy),
   not V*. On the model's **own** won positions it correctly outputs ≈draw,
   because under its policy those positions *do* draw (Experiment C: 2% convert).
   (The probe's 0.89 was measured on *teacher/Stockfish* positions; the
   representation decodes the win at AUC 1.0 — the signal is present, the head
   maps it to draw.)
2. At a won self-play position, MCTS evaluates every continuation with V → all
   ≈0 → no move preferred → it shuffles. **No conversion gradient.**
3. More sims don't help (Exp D): conversion needs 20–50-ply lookahead beyond any
   sim count, and V can't distinguish "progressing" from "static" (both Vᵖ-draw).
   AlphaZero's 800–1600 sims worked because their V was informative.
4. **Self-reinforcing fixed point:** weak policy → Vᵖ=draw everywhere → MCTS finds
   no improvement → policy stays weak → self-play all draws → V re-confirms draws.
   Evidence we're *in* it (not climbing out): move quality degraded (Exp E), draw
   rate rose 0.89→0.96, eval-vs-random flat/down. "Run it longer" won't ignite.
5. **Why no bootstrap:** AlphaZero from random gets decisive games (blunders→wins)
   that seed V with a gradient. Our Stockfish-warmstarted model is in a *dead
   zone* — solid enough not to blunder, not good enough to convert — so self-play
   is all draws and produces no decisive signal. We skipped the phase that
   generates the bootstrap.

## What actually moves the needle (none are "more sims" / "more steps")

- **Conversion curriculum** — seed self-play from won positions so the model must
  convert or visibly fail → generates decisive games to learn from.
- **Draw/repetition penalty** (implemented, `cf6cb9f`, feature branch) — removes
  the free-draw escape; Exp A confirms it's well-targeted (90% winnable).
- **Reward shaping** (material/progress) — dense signal so V isn't flat over the
  horizon (`reward_support_size` is currently 1 = no shaped reward).
- **Value-target relabeling toward V*** — use the representation's win signal
  (AUC 1.0) instead of pure Vᵖ outcomes (q_ratio pushed further).
- **Lower `warmstart_sample_frac`** (0.4→0.1) — frees the policy from diffuse
  Stockfish-soft-label imitation (entropy 2.5→3.3 correlated with the regression).

## Open / not yet run
- 68k-vs-29k head-to-head (Stockfish-free skill check) — offered, not run.
- Short fine-tune A/B of the repetition penalty — deferred.

## Findings 2026-06-16 (run `2026_06_15_2phase_noprogpen`, phase 2)

### Legal-move policy masking — IMPLEMENTED (this branch)
On-policy probe at 22k (`scripts/probe_policy_value_consistency.py`):
- Policy leaks **17% of mass onto illegal moves** (deferred bug #1, measured).
- Legal-only entropy 2.83 vs log(33)=3.50 → diffuse but not uniform; top legal
  move only 0.15–0.19.
- **Value/dynamics HAS signal**: per-position Q-spread 0.22, best move +0.12,
  root value std 0.24 (NOT collapsed). The bottleneck is the **policy**, only
  0.19-correlated with the model's own Q, picking moves ~0.08 Q worse than its
  value-best.
Reference impls (muzero-general, LightZero, DeepMind pseudocode) do NOT mask the
policy loss — fine when ~all actions are legal (Atari/small boards), weak for
chess (99% illegal). Fix: `config.mask_illegal_policy` — keep the standard
FULL-softmax CE (which learns legality for free via the shared normalizer) and
ADD an illegal-mass penalty on top, driving illegal_mass below the CE's ~17%
floor; uses stored `legal_actions_list`. Logs `policy/illegal_mass`. Opt-in
`--mask-illegal-policy`.

IMPORTANT (fix 2026-06-16, commit 93f0b30): the first implementation RENORMALIZED
the CE softmax over legal moves — that is shift-invariant over the legal logits
and gives illegal logits zero gradient, so it CANNOT teach legality (caught live:
`policy/illegal_mass` frozen at ~0.98, `entropy_pred` pinned at log(4672) for 700
steps). Root cause is exact: full_CE = masked_CE − log P(legal); renormalizing
drops the −log P(legal) legality term. Corrected to full-softmax CE + penalty,
which immediately drops illegal_mass (0.99→0.90 by step 300, accelerating).

### The penalty RELOCATES draws, doesn't reduce them
Phase-2 trajectory (15k→23k): no-progress draws 0.08→**0.00** (δ penalty worked),
but threefold draws 0.56→**0.84** and overall draw rate 0.81→**0.90**; p1_win
0.09→0.04; value/target_std 0.72→**0.51**. The penalty changed draw *composition*
but the rate ROSE and decisiveness FELL — the policy still can't convert, so
draws relocate (no-progress → threefold) and multiply. ⇒ More contempt/δ = more
relabeling, not more conversion. The lever is the policy (masking + warmstart
fade), not the value-side knobs.

## Potential follow-up: treat threefold repetition as an ILLEGAL move
Mask the 3rd-repetition-completing move during self-play generation (natural
extension of the legal-mask machinery). Steelman: Exp A says 90% of repetition
draws are winnable, so forcing play on mostly yields *correct* decisive data.
Risks: (1) changes the rules — fortress / perpetual-check positions become forced
losses → value head mislabels them vs real chess; (2) RELOCATION — masking only
the threefold move lets a diffuse policy drift to the 50-move drain instead (the
phase-2 data above is direct evidence of this relocation effect). If tried: do it
in self-play GENERATION only, keep eval + value targets on honest rules, and
watch whether decisive-game fraction actually rises and whether the health probe
shows over-pessimism on known-drawn endgames. Sequence AFTER the masking fix —
masking attacks the cause (policy can't convert); illegal-repetition the symptom.

## PROPOSED next run (queued — not yet launched; revised 2026-06-16 post 3-run comparison)
Revised after `run_comparison_2026_06_16.md`, which corrected the framing: (a) BOTH
noprog & fillbuf carry `draw_score=-0.05` (we never tested contempt=0); (b) the
penalty effect is WITHIN NOISE; (c) the two-pool anchor DECISIVELY prevents value
collapse (qratio, no anchor, collapsed to a constant near-draw predictor by 24k;
both anchored runs stayed calibrated); (d) draining the anchor did NOT genuinely
sharpen the policy (qratio's lower entropy was a value-collapse artifact) — the
diffuse policy is universal and untouched in all three runs.
1. `--mask-illegal-policy` — THE headline change. The diffuse-policy bottleneck is
   universal (legal-only entropy 2.4–2.9, 10–17% illegal mass, policy↔Q ≤0.15) and
   untouched by anything tried; masking is the only lever aimed at it.
2. `--warmstart-sample-frac 0.3` (down from 0.4) — GENTLE fade only. The anchor is
   what protects the value head (proven), and draining it does NOT sharpen the
   policy, so this is a cautious half-step, not the policy fix it was framed as.
   Monitor `value/target_std` — if it slips, the fade went too far.
3. `--repetition-penalty 0.35` + `--repetition-penalty-decay 0.93` (NEW) — surgical
   δ (threefold + 75-move) with EXPONENTIAL per-ply weighting (discount-style):
   full δ=0.35 at the drawn position, γ=0.93 geometric decay backward (half-strength
   ~9.6 plies back, soft tail — penalizes a long no-progress shuffle over its whole
   length, unlike the linear window's hard cutoff). Expectations TEMPERED: the
   penalty effect is within noise; per-ply weighting makes it better-targeted, not
   guaranteed-effective. Low-cost secondary test riding the masking run.
HOLD fixed: `draw_score=-0.05`, `selfplay_q_ratio=0.1`, warmup 15000,
warmstart-buffer-size 300. (Net: masking is the one big variable → clean attribution.)

Full command (swap run-id):
```
.venv/bin/python -u scripts/train.py \
  --game chess --run-id 2026_06_16_maskpolicy_fade \
  --stockfish-injection-path data/stockfish_injection \
  --stockfish-injection-games 300 --stockfish-injection-interval 256 \
  --self-play-warmup-steps 15000 --warmstart-buffer-size 300 \
  --warmstart-sample-frac 0.2 --repetition-penalty 0.35 \
  --repetition-penalty-decay 0.93 \
  --mask-illegal-policy
```
(warmstart-sample-frac set to 0.2 at launch 2026-06-16 — user opted for the
stronger fade over the 0.3 compromise; watch value/target_std for collapse risk.)
Watch: `policy/illegal_mass` (→0), `policy/entropy_pred` (should drop below
noprogpen's ~3.5 if masking sharpens the policy), `value/target_std` (must NOT
crater from the 0.3 fade — the anchor's protective job), and the draw/decisiveness
trend vs noprogpen at matched steps.
Optional fast sanity check first: fine-tune from `noprogpen` ~24k+ checkpoint with
`--mask-illegal-policy` for a few k steps — if illegal_mass drops and entropy_pred
falls, the mechanism is confirmed cheaply before the full fresh 2-phase run.

δ exponential schedule (δ=0.35, γ=0.93): plies-before-draw → δ: 0→.350, 2→.303,
5→.243, 10→.169, 20→.082, 30→.040, 50→.009.

## Follow-up: POTENTIAL-BASED reward shaping (PBRS) — the principled spec
Sequenced escalation after the masking+anchor run, if draws persist. Research
brief 2026-06-16 (Ng/Harada/Russell 1999, "Policy Invariance Under Reward
Transformations").

REFRAME (important): PBRS does NOT "penalize draws." Our draws are CONVERSION
FAILURES (Exp A: ~90% winnable), not genuine draws. PBRS gives the search a dense
per-move gradient toward PROGRESS so the model CONVERTS winnable positions → draws
fall because wins RISE, not because draws are punished. It is policy-invariant, so
it provably will NOT bias against genuinely-drawn positions (that would be
contempt — non-potential-based, objective-distorting, and our δ penalty already
showed that's a weak/within-noise lever). So PBRS is the right tool for (a)
convert-winnable; it is NOT a tool for (b) avoid-genuine-draws.

FORM: shaping reward F(s,s') = γ·Φ(s') − Φ(s), Φ a state potential. Ng et al.
proved this is the necessary+sufficient form for the optimal-policy set to be
UNCHANGED (telescopes to a trajectory-constant → cannot be hacked / cannot distort
the optimum). Only naive shaping (r' = r + arbitrary F) hacks/distorts.

CONCRETE (anti-shuffle): Φ(s) = −λ·(no_progress_clock(s)/100) — 0 at a fresh
position, more negative as shuffling drags on (we already track the halfmove clock
+ repetition count in planes 19–21, so Φ is a valid Markov state fn). Then a
shuffling move (clock climbs) → small NEGATIVE F (discouraged); a progress move
(pawn push / capture resets clock) → POSITIVE F (rewarded). Broader alternative if
too narrow: Φ = material balance (same PBRS form, denser "toward winning"). The
potential MUST be external (clock/material) — a LEARNED potential (model's own
value, cf. "Bootstrapped Reward Shaping" arXiv:2501.00989) re-introduces the
self-referential drift we rejected.

HOW IT BITES: shaped reward → reward head → MCTS Q = reward − γ·V; MCTS sums
rewards along simulated lines, so it shapes the search over its lookahead.
CORRECTION to the earlier note: this does NOT require changing reward_support_size
— at K=1 the head is already 3 bins over {−1,0,+1} and represents fractional
rewards exactly (verified), so NO architecture change / NO checkpoint
incompatibility. Work is just: populate GameHistory.rewards with F in self-play +
tune λ. LIMIT: WDL value does not bootstrap the shaped reward, so shaping reaches
only as deep as the search tree (plain outcome at leaves) — a real WDL+shaping cap.

WHY BETTER THAN THE δ PENALTY: dense (every move) vs sparse (terminal); per-move
credit vs smeared; policy-invariant (genuine draws untouched) vs contempt-like
mislabeling; no objective distortion vs distortion.

CAVEAT: PBRS policy-invariance is proven for standard MDPs; in MuZero (learned
dynamics, MCTS, WDL value) it's APPROXIMATE — treat as a well-motivated heuristic.
Still the most paper-defensible shaping (preserves the optimum → a learning
accelerator, not a new objective; vanilla AlphaZero uses no shaping at all).

DECISION RULE (after the masking + anchor run):
- draws fall (value/Q gains spread, conversion improves) → done, no shaping needed.
- draws persist BUT value/target_std healthy + policy sharp → conversion-depth
  problem → PBRS no-progress (or material) potential is the justified next step.
- value/target_std collapses despite the 0.4 anchor → value-stability problem
  first (anchor/architecture), shaping won't help yet.
