# MCTS / self-play repetition diagnosis + correctness audit — 2026-06-17

Context: conv-head run (`2026_06_17_convhead`) entered self-play at step 15360 and the
draw composition reverted toward repetition shuffles (threefold rate 0.28 → 0.59 within
~1500 steps). Investigated by (a) tracing real self-play games from `checkpoint_15000.pt`
with the MCTS root tree captured per move (`scripts/trace_threefold.py`), and (b) a
5-dimension correctness audit of BOTH search engines:
- `src/mcts/mcts.py` — serial `MCTS` + numpy `BatchedMCTS` (offline trace / reanalyze)
- `src/mcts/tensor_mcts.py` — `TensorMCTS`, the LIVE gpu-resident self-play engine

## Behavioral finding (11 threefold games traced, all consistent)

The repetition draws are NOT the flat-value (q≈0) regime. At the repetition positions:
- **H4 (no repetition awareness):** `root_value` does not drop toward 0 at the repetition —
  it *rises* (last-12-ply means +0.17…+0.53 vs whole-game ~+0.06–0.11). Completing a
  3-fold produces ZERO value penalty.
- **H3 (value prefers the repeating move):** because the shuffle re-enters a position the
  net still scores as winning, the repeating move carries the HIGHEST q_mover. Chosen-move
  q at repetition vs non-repetition positions: +0.33/+0.42/+0.53/+0.36/+0.33 vs +0.11/+0.05/
  +0.23/+0.11/+0.09 (3–6× higher).
- **H1 rejected:** q_spread at critical plies is large (0.16–0.34, up to 0.68), root_value
  strongly positive — the value head is confident, just wrong.
- **H2 rejected:** repeating-move prior is modest (0.10–0.30), top-prior in only ~2–3/12
  critical plies; visits follow Q. At rep positions the repeating move got only **44% of
  visits on average** — the majority went to (correctly explored) alternatives that came
  back evaluated ≤ the shuffle.

Mechanism: these are objectively WON endgames (K+P vs K, extra pawn) the weak policy can't
convert. The value still says "+0.5 winning"; the search has no repetition signal, so the
shuffle keeps full winning value and is the top-Q move; MCTS banks a phantom win into the
draw. This is V^π≈draw at the endgame-conversion level (see [[value-sibling-ranking]]).

## MCTS correctness audit — results

**Core search math is CORRECT and the two engines agree (no divergence found):**
- PUCT scoring: identical across serial / numpy / TensorMCTS-torch / Triton kernel.
  Explore/exploit balance is healthy — a fresh prior-0.05 sibling scores 0.89 vs a 50-visit
  high-Q child's 0.835, so the formula does NOT starve good-but-low-prior moves.
- Value backprop / negamax: both engines numerically identical to ~1.1e-16; sign flips,
  leaf POV, discount, reward all correct and consistent backup↔selection.
- MinMaxStats: per-game/per-search scope (not shared across batch — verified), consistent.
  Inherent (not a bug) wide-span Q-compression: ±1 leaf values shrink a real 0.1 Q-edge to
  ~0.05 normalized → amplifies (does not cause) value-blindness.
- The a9906eb-class numpy-vs-tensor divergence: none on any dimension checked.

**Three real exploration-relevant issues (none is a tree-search math bug):**

1. **`dirichlet_alpha=0.1` is peaked (config) — but TUNING IT DOES NOT HELP (tested).** The
   static analysis (Dir(0.1) over ~10 moves puts ~0.66 weight on one move; a prior-0.06
   converting move lands at median post-noise prior 0.045) is correct about the PRIOR, but an
   α sweep refutes the recommendation to raise it. At sims=200/temp=0.1, varying α∈{0.1,0.3,
   1.0}: threefold 44%→56%→81% (rose, didn't fall), decisive flat (noise), and — decisively —
   the chosen-move visit fraction at repetition positions stayed ~0.44 across ALL α (visit
   entropy ~1.31–1.36, flat). Root noise perturbs only the first few sims; after 200 sims the
   visit distribution re-converges onto the value verdict, which α doesn't touch. **Do NOT
   raise α — it's not a lever; the value is binding.** (Was a wrong recommendation in an
   earlier draft.)

2. **Leaf nodes expand the full UNMASKED action space — both engines** (`mcts.py:146-147,640`;
   `tensor_mcts.py:1195-1197`). Known-deferred issue #1, confirmed live on both paths: the
   dynamics is queried on illegal actions and illegal subtrees can siphon visits/value from
   legal moves at depth, wasting search budget. Severity MEDIUM. **Fix: leaf-node legal
   masking.**

3. **No terminal / repetition awareness inside search — both engines** (HIGH for this
   symptom). Value comes only from `recurrent_inference` on latent states; there is no
   draw/terminal/repetition branch in select/expand/backprop. A line shuffling into a
   threefold is scored by the network's value of the (won) latent position → no value drop.
   Inherent to latent-model MuZero, not a regression. **Fix: MCTS-level repetition penalty
   (detect repeated positions along path/history and penalize Q) + win adjudication**, since
   the learned latent model structurally cannot track repetition.

Subtree reuse is confirmed OFF everywhere (would carry stale visits onto the previously-
chosen shuffle move if enabled — keep off until the value/terminal issue is resolved).

## Lever experiments (15k checkpoint, scripts/trace_threefold.py)

All on checkpoint_15000, 16 games/cell (baseline 24), seed 0. EVERY search-side lever fails
to fix the repetition → the binding constraint is the value/target, not search.
- Baseline (sims=200, temp=0.1): 46% threefold, 12.5% decisive (24 games).
- sims (200→800, temp 0.1): threefold 46%→75%, decisive 12.5%→6%. MORE search HURT (amplifies
  the miscalibrated value; replicates the sim-scaling probe). Holds at temp 0.75 too
  (cell C→D: threefold 12%→44%, decisive 31%→6%).
- temp (0.1→0.75, sims 200): threefold 46%→12%, decisive 12.5%→31% BUT unfinished →31% — temp
  only scatters the shuffle via late-game randomness/blunders (not skillful conversion); the
  decisive bump is noise- and blunder-driven, with games wandering to the ply cap.
- dirichlet_alpha (0.1→0.3→1.0, sims 200, temp 0.1): threefold 44%→56%→81% (rose), decisive
  flat, chosen-move visit frac at rep positions ~0.44 for ALL α. α is not a lever (see above).
Conclusion: sims hurt, temp only scatters via blunders, α does nothing — search/exploration
cannot fix it; the fix is target-side (win adjudication) + structural (in-search repetition
penalty). A bigger value head or more sims does not help.

## Implications for the NEXT run (do NOT change mid-isolation on convhead)

The real fix is target-side/structural: **win adjudication** (correct decisive labels) +
**MCTS-level repetition penalty** (the latent model can't see repetition). Medium: leaf-node
legal masking (stops budget leaking into illegal subtrees). Search/exploration knobs do NOT
help — tested: more sims hurt, temperature only scatters via blunders, and dirichlet_alpha
tuning does nothing (chosen-move visit frac at rep positions is ~0.44 for α∈{0.1,0.3,1.0}).
A bigger value head doesn't help either. See [[value-sibling-ranking]].
