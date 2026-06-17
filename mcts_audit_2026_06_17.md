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

1. **`dirichlet_alpha=0.1` is too peaked (config).** Dir(0.1) over ~10 endgame moves puts
   ~0.66 weight on one move (~2 effective moves). A converting move at base prior 0.06 lands
   at MEDIAN post-noise prior 0.045 (noise usually moves mass AWAY from it) and its noise
   coordinate is ~0 about 49% of the time → starved ~half the searches. AlphaZero-chess used
   0.3. **Fix: raise `dirichlet_alpha` 0.1 → 0.3.** (config comment already flags 0.1 as a
   temporary compromise.)

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

- Baseline (sims=200, temp_final=0.1): 46% threefold, 12.5% decisive (24 games).
- sims=800, temp=0.1 (cell B, 16 games): 75% threefold, 6% decisive — MORE search HURT
  (amplifies miscalibrated value; replicates the sim-scaling probe).
- temp/alpha sweeps: in progress (see /tmp/tf_C, /tmp/tf_D, /tmp/tf_alpha_*).

## Implications for the NEXT run (do NOT change mid-isolation on convhead)

Cheap, in config: `dirichlet_alpha` 0.1 → 0.3. Medium: leaf-node legal masking. The real fix
is still target-side/structural: win adjudication + MCTS-level repetition penalty. A bigger
value head or more sims does NOT help (more sims hurt). See [[value-sibling-ranking]].
