# KLDGain adaptive search — useful for ChessZero? (2026-06-25)

Analysis of whether to adopt Lc0's **KLDGain** (adaptive search-stopping by
visit-distribution convergence) in our MuZero self-play. Short answer: **conceptually
powerful and aligned with the "more sims for endgames" intuition, but DEFER** — it
fights our lockstep GPU-batched self-play, it directly anti-synergizes with the TB
probe (the exact Lc0×KLD failure), and it assumes a calibrated value head we're
still fixing.

---

## 1. What KLDGain actually is (Lc0, primary-source verified)

Lc0's `KldGainStopper` (`src/search/classic/stoppers/stoppers.cc:157–184`) periodically
computes the KL divergence between the **root visit-count distribution now** and the
**same distribution `KLDGainAverageInterval` nodes ago** (default 100), divides by the
nodes elapsed → "information gain per node," and **stops search when that drops below
`MinimumKLDGainPerNode`**. Once extra playouts stop changing the visit distribution,
more search is uninformative → halt.

It is **adaptive both ways**: obvious positions converge fast → fewer nodes; sharp
positions keep shifting → more nodes.

| Param | Default | Notes |
|---|---|---|
| `MinimumKLDGainPerNode` | **0.0 (off)** in the binary | training runs set it (~`0.000005`) |
| `KLDGainAverageInterval` | 100 nodes | snapshot spacing |
| `SmartPruningFactor` | 1.33 UCI / **0.0 self-play** | a *separate* stopper (budget-based), not KLDGain |

**Reported impact** (Lc0 blog 6/2019; Veedrac Test40/50 writeup): threshold `0.000005`
gave **+120 Elo in training matches at an average of only 766 nodes/move** vs fixed 800,
with sharp positions searched **up to ~5300 nodes**. So it's a *better-allocation* win,
not a pure speed-for-quality trade.

> Confidence: formula, params, defaults, and the +120/766/5300 figures are primary-source
> verified. The *mechanism* of the DTZ-boost×KLD interaction (§4) is **not** documented —
> inference only.

## 2. Why it's tempting for us

- We use a **fixed** `num_simulations` (200/400/800). The A100 run shows 800 sims clearly
  helps quality. KLDGain would let us spend that budget **smartly**: cut sims where the
  policy is already confident, pour them into sharp/endgame positions.
- This *is* the principled version of the "increase sims by # of plies / in endgames"
  idea we floated — allocate by policy non-convergence (which correlates with sharpness)
  rather than a hand-set ply schedule.
- On the A100 the efficiency framing is "same quality cheaper → more games/day," and the
  quality framing is "same budget, allocated better → stronger targets."

## 3. Why it's hard for us — the architectural mismatch (the crux)

Our self-play is **GPU-batched in lockstep**: `BatchedMCTS`/`tensor_mcts.run_batch_gpu`
runs N parallel games (256) through the **same** number of simulations, one batched
forward pass per sim step across all games. That lockstep is the source of our GPU
throughput.

KLDGain needs **per-game variable node counts** (game A stops at 50, game B runs to 800).
In a lockstep batch you can't stop one game without either:

- **(a) padding** — keep all games running to the max → you waste exactly the compute
  KLDGain was meant to save (no efficiency gain), or
- **(b) ragged batching** — drop converged games from the batch mid-search, shrinking it
  over the sim loop. Feasible, but the tail sims run on tiny batches (GPU underutilized),
  it breaks the static-shape fast path, and adds real control-flow complexity. The
  node-count savings may not convert to wall-clock on a GPU that's happiest at full width.

Lc0 orchestrates per-game search threads on CPU, so per-game early-stop is natural there.
We traded that flexibility for batched throughput; KLDGain wants the flexibility back.

## 4. Direct anti-synergy with our TB probe (the Lc0×KLD trap)

KLDGain measures *visit-distribution convergence*. Our **soft TB value bias** sharpens the
visit distribution at TB positions on purpose → it would **converge fast** → KLDGain would
**cut search exactly at the endgame/TB positions where we want MORE search.**

That is almost certainly the "**DTZ policy boost disabled due to bad interaction with
KLD**" that Lc0 hit: a boost concentrates visits immediately → KLD reads ~zero gain →
the stopper fires after the minimum interval → near-zero-search, low-quality samples in
exactly the endgames the boost was meant to teach. We'd reproduce it. Mitigable (floor
the node count / exempt TB plies from the KLD stopper), but it's added coupling.

## 5. It assumes a calibrated value — which we're mid-fixing

KLDGain treats "visit distribution converged" as "position resolved." But with a
flat/miscalibrated value (our known issue; value_score is gated until visits>0 so the
**prior dominates early** — see the value-drives-selection finding), the distribution can
converge **fast onto the wrong move**. KLDGain would then stop early on a confidently-wrong
target and bank it. KLDGain is safe **after** the value head is well-calibrated; applying
it during the value-target fix risks freezing premature, wrong-but-confident targets.

## 6. A cheaper proxy that captures most of the upside

If the goal is "more search in endgames," a **phase/material-bucketed sim schedule**
(e.g. raise `num_simulations` once piece count drops below a threshold, or bucket games by
phase and run each bucket at its own fixed sim count) is a coarse static approximation of
KLDGain that is **batching-friendly** (uniform sim count within a batch) and trivial to
implement. Less optimal than dynamic KLD, but no lockstep conflict and no value-calibration
dependency.

## 7. Verdict

**Defer KLDGain.** It's relevant and powerful in Lc0, but for us right now:

1. It fights lockstep GPU-batched self-play; the savings may not survive ragged batching
   (significant engineering for uncertain wall-clock gain on an A100 that wants full width).
2. It directly anti-synergizes with the TB probe — it would starve search at endgames (the
   exact Lc0×KLD failure) unless we exempt TB plies.
3. It presumes a calibrated value head; we're literally in the middle of calibrating it
   (the DTZ value relabeling).

**Sequence instead:** (1) finish the value-target relabeling and confirm the value head
ranks winning positions by progress; (2) if we still want adaptive compute, start with the
cheap phase/material sim schedule (more sims in endgames); (3) only build true KLDGain if
profiling shows the fixed-sim budget is the bottleneck — and if so, implement it as
ragged-batch with a **TB-position node floor** so it can't starve the endgames we care
about.

### Sources
- lc0 master: `src/search/classic/stoppers/stoppers.cc` (`KldGainStopper`), `common.cc:597–665`
  (option defaults).
- Lc0 blog, "What's going on with training!" (June 2019) — KLD purpose + DTZ-boost
  disabled "due to bad interaction with KLD."
- Veedrac, "Leela Chess — Test40, Test50, and beyond" — +120 Elo / 766 avg / 5300 max /
  threshold 0.000005.
