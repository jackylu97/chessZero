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
