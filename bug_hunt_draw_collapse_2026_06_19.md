# Draw-Collapse Bug Hunt (2026-06-19)

6 investigators tracing ACTUAL values through MCTS + the target pipeline on `cold2_pc@6000` (CPU, numpy
MCTS, 200 sims, real buffer) + adversarial prosecutor. Verdict: **BOTH** — fundamentally V^π≈draw
(dominant), with **three genuine but second-order bugs** stacked on top. Search math is correct;
**no numpy↔tensor divergence** (re-confirmed).

## The decisive proof (value-traced)

The value head **ranks ACROSS positions but is ~26× blinder WITHIN a position:**
- within-position sibling-Q std = **0.0076** (median 0.0055; 95% of positions < 0.02; best−median margin ~0.009)
- across-position root-V std = **0.202**; ratio within/across = **0.038**
- head P(draw) > 0.8 in **93%** of positions; WDL mean ≈ [0.07, 0.87, 0.06]
- KQvK (a trivial win): head says **~draw** (V≈−0.08), and on an in-distribution K+Q-vs-K shuffle it says **L=0.42 (losing!)**

So PUCT has almost no value gradient between sibling moves → selection is driven by prior + Dirichlet noise →
no signal to convert wins → the policy drifts to repetition. This drives **both** the 99% draw rate **and**
the game-length collapse (the model gets *better* at reaching a repetition).

## Two stories the prior audits over-weighted — REFUTED by the traces

- **"Phantom win banked into a repetition" is NOT dominant.** Mean root_value *entering* repetitions = **+0.061** (≈draw); only **8%** of rep-games show tail root_value rising. The search walks into repetitions because they're valued the **same** as everything (flat), not because it thinks it's winning.
- **Reward-head hallucination is OOD-only.** In-distribution mean-of-max |reward| over visited children = **0.0006** (true reward 0). The big +0.66 spike was only on a constructed zero-history KQvK. Not a driver.

## Three genuine bugs (your instinct — real, but second-order amplifiers, not the cause)

1. **Live self-play has ZERO in-search repetition awareness.** `forced_draw_actions` exists only in numpy
   `BatchedMCTS` (`mcts.py:644-650`) and is passed **only by diagnostic scripts** — never by production
   self-play (`self_play.py:174,298,525`). `TensorMCTS.run_batch_gpu` has no such param; `root_terminal_draws`
   isn't even a config field; the `triton` select backend has no terminal-draw path. **But** the forced-draw
   flip test shows it only redirects to moves of *equal/worse* Q (Kg3 −0.010 → Kg4 −0.028) — it changes *which*
   shuffle, not whether the side converts. So fixing it alone barely moves the draw rate.
2. **`selfplay_q_ratio` dilutes decisive targets (self-referential).** `eval_to_wdl(root_value≈0)=[0.12,0.76,0.12]`,
   so a clean WIN one-hot at `q=0.5` becomes `[0.56,0.38,0.06]` (scalar +1.0 → **+0.48**) — 38% draw mass injected.
   Conditional: negligible when root_value is correctly large (+0.85→scalar +0.89), but rep-game root_values are
   ~0, so it bites. **Explains "q_ratio=0.5 collapses fastest."** Fix: keep `selfplay_q_ratio=0.0` until the value
   head is externally calibrated.
3. **`repetition_penalty` is STM-symmetric** (`replay_buffer.py:316-335`): the Draw→Loss tilt applies to *both*
   sides, so the objectively-*winning* side of a won-but-shuffled position is taught it's **losing**. Since 94.6%
   of games are rep-draws, this reaches most late plies. Fix: `rep_penalty=0` for cold start, or gate the tilt on
   the position not being winning for STM (root_value/eval ≤ ~0).

## The crux

**Only injecting within-position value resolution breaks the basin.** Every search-side / cheap lever is
near-inert without it (the flip test proves redirected moves have equal/worse Q). The two ways to inject it:
- **(a) Win-adjudication** — stop playing objectively-won positions out to threefold; label them wins. *(Previously vetoed by the user.)*
- **(b) Persistent external (Stockfish) per-position value supervision** on self-play positions — the `external_values`
  path already exists in `make_target`/`_wdl_target_at`. (This is effectively the warm-start/anchor route.)

The cheap guards (#2 `selfplay_q_ratio=0`, #3 `repetition_penalty` STM-gating) and the #1 structural fix
(wire repetition-awareness into `run_batch_gpu`) are worth doing but are **secondary** — they slow the
collapse, they don't reverse it.
