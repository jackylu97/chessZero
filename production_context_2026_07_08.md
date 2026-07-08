# Production Run — Full Context Handoff (2026-07-08)

Companion to `production_config_2026_07_08.md` (settings + evidence). This doc is
the campaign context: what problem the run attacks, what was discovered on the
way, what every mechanism is for, and how to read the run while it's alive.
Deep detail: `strategy_2026_07_02.md` §1–§17, `setup_vs_precedent_2026_07_03.md`.

## 1. The problem

MuZero chess self-play at our scale learns to WIN vs weak opposition but cannot
CONVERT won endgames: winning positions get shuffled into 50-move/threefold
draws. Root cause (measured, 2026-06-17): the self-play value target is the
outcome under the current policy — unconverted wins get labeled ≈draw, the
value head learns "winning ≈ draw", search loses its gradient, self-reinforcing.
The previous production run (800 sims, PER, conv net) hit 90% vs random purely
on MIDGAME mating power; it had the same endgame hole. The vs-random metric
does not measure conversion — the diag does.

## 2. The conversion mechanism (the campaign's central discovery)

Search converts endgames through MATE BEACONS: when search finds a real mate
in-tree, the edge's predicted reward (+1) stacks with the mated child's value
(≈−1, γ=1) giving |Q|≈2 spikes that tower over ordinary winning branches
(≈0.9). This is technically a deviation from MuZero's absorbing-state
convention — and it is load-bearing: muting the reward channel drops conversion
0.20→0.048 on identical weights. Consequences:
- The dynamics-REWARD HEAD is the most conversion-critical component. Its
  PRECISION (not firing on non-mating moves) is everything: phantom rewards
  hand out beacon-credit everywhere and search dithers.
- The reward head's training diet comes from mate-dense demonstration data
  (anchor/fill). The minimal-arm ablation proved this: without anchor, the
  head stays dark and conversion sits at the value-only floor.
- `scripts/probe_reward_precision.py` is the standing instrument (recall /
  false-fire on mate-in-1 sets).

## 3. What each mechanism in the config is FOR

- **Gumbel MuZero (m=16)**: per-step policy improvement guarantee; π′ targets;
  deterministic argmax evals (cleaner curves).
- **Uniform sampling**: replaced PER. NOTE: PER in the old stack was
  ACCIDENTALLY doing decisive-signal concentration; the new stack does that
  deliberately (fill/anchor/mixture). Don't re-add PER on top.
- **Rollout fill**: at a game's first decisive in-TB ply, truncate and finish
  with TB-optimal play → whole-trajectory TRUE z + on-distribution demos.
  This is the direct fix for the §1 root cause ("truth restoration").
- **TB anchor** (20k-game archive, cycling injection): perfect demonstration
  games. Primary role discovered late: mate-example DENSITY for the reward
  head. Volume now capped/chosen (anchor_max_size).
- **TB relabels**: value (hard one-hot), DTM (moves-left), policy (DAgger soft
  win-preserving sets at learner-visited states, 0.5→0.2). Oracle truth at the
  states the model actually reaches.
- **Endgame seeds + DTM curriculum**: practice from won positions, easy→hard;
  seed mate_rate = the honest in-run conversion signal (resign-exempt so
  conversion means MATE).
- **Batch mixture schedule**: declarative composition (warmstart/anchor/
  selfplay per phase). Replaces stage-emergent accidents (warmup was silently
  60% anchor before task #19).
- **Opening ε-mixture**: self-play game diversity with ZERO policy targets on
  opening plies (the legacy random-opening trained the policy toward random
  moves).
- **Moves-left head + search utility**: Lc0-style "win sooner" tiebreak.
- **Material head (aux)**: world-model regularizer on the raw latent.
- **from→to policy head**: NEW (arch sweep 2026-07-08). Moves scored as
  bilinear from/to square-token relations. +48% proxy conversion vs conv head.
  Boldest element; watch its tripwire.

## 4. Run genealogy (what beat what, all at matched instruments)

| run | config | eval band | conversion (Gumbel@400) | notes |
|---|---|---|---|---|
| control (07_04 gumbel) | hybrid 4.85M, 400 sims | 50–60 | **0.152 @30k** | first beacon-functional run; reward FF matured 0.38→0.12 |
| v1 bundle | + symmetry(+bugs) | ~parity | 0.02–0.04 | KILLED: augmentation parity bug corrupted transitions (§11–13) |
| v2 + reward guard | partial fix | ~parity | 0.068 | guard couldn't reach the trunk; superseded by root-cause fix |
| v3 | bundle − symmetry + 8-plane reward head | 56–70 | 0.113 @30k / **0.160 @160-ply** (KQvK 0.45) | production base |
| minimal | truth-only (no seeds/anchor/DAgger) | ≈v3 | **0.048** | proved scaffolding = reward-head fuel, not policy teaching |
| XL (07_06) | 24M, 2× schedule | records (70s) | 0.107 @32k → **0.047 @60k after 200-sim period** | scale validated for strength; sims cut starved the technique loop (§16) |

## 5. Hard-won operational rules

1. **Sims 400 floor.** Too little search fails COMPOUNDINGLY (the technique
   loop starves silently); too much costs linearly. Never cut search based on
   a static A/B — only dynamic evidence (parallel arm / A/B/A) counts.
2. **Measure under training conditions.** A diag with the wrong search regime
   (PUCT vs Gumbel, wrong sims) can misread a model by 8×. All standard
   instruments pin Gumbel@400.
3. **Static component probes can all pass while the system fails** — the
   v1/v2 corruption lived in TRANSITIONS (parity bug); every single-position
   probe was clean. When components probe healthy and behavior is broken,
   suspect the couplings; audit code, not just outputs.
4. **Accidentally load-bearing mechanisms need deliberate replacements, not
   removal** (PER's decisive bias → fill/anchor/mixture; reward double-count
   beacons → kept intentionally; 800-sims → 400 floor + games/step).
5. **Loss levels are not comparable across target-regime changes** (π′ vs
   visits, hard vs soft z, PER vs uniform display bias). Compare fixed-probe
   metrics (SF-agreement) and behavioral instruments (diag) across runs.
6. **Index-mapping code gets a deterministic rule-engine test before it
   trains** (symmetry equivariance, from→to LUT parity). This class of bug is
   silent, catastrophic, and cheap to test for.

## 6. Reading the run day-to-day

- `production_probes.log`: reward precision + conversion diag per 10k — the
  two numbers that matter. Expected shape: precision false-fire falls toward
  ≤0.12 while recall climbs; conversion flat-ish through warmup then climbing
  from ~40k. Compare against §4 table.
- TensorBoard: `loss/policy_loss_warmstart` (SF-agreement — quality + overfit
  tripwire, records ~0.81), `batch/frac_*` (mixture compliance),
  `seed/mate_rate` (honest in-run conversion), `self_play/tb_fill_rate`,
  `buffer_decisive_frac` (healthy ~0.75+), `train/grad_norm` (isolated spikes
  fine), resign_false_positive (calibration).
- eval-vs-random saturates ≈70%+ (draws = unconverted wins). Late-run signal
  lives in the diag, not the eval. Eval ladder (fixed engine opponent) is the
  known gap.

## 7. Parked with re-open conditions

- **Sibling-zero reward contrast targets** (§12 design): re-open if reward
  false-fire stalls >0.25 mid-run. Rule-derived true labels; design ready.
- **Material search utility** (task #15): re-open only on measured
  minimal-margin-trade-down losses.
- **Symmetry augmentation**: parity-fixed (a0fd5af, SAFE_ELEMENTS) but
  shelved; if ever revived, reward-precision probe is its gatekeeper.
- **A/B/A sims leg**: XL paused at 68k (resumable) — conversion recovery at
  400 sims would complete the §16 causality proof.
- **Eval ladder**: top backlog item; needed within the first days of
  production for late-run visibility.

## 8. State of the repo at launch

- Branch `perf/deferred-tb-relabel`, all work committed through the arch sweep.
- Local GPU (5090): XL paused at checkpoint_68000 (resumable via
  `scripts/_launch_hybrid_xl.sh` — currently configured to resume @62k ckpt
  at 400 sims; update the --resume path to 68000 first if resuming).
- Key artifacts: `data/tb_anchor/` (20k games), `data/endgame_seeds_train.txt`
  (+`.dtm` sidecar), `data/stockfish_injection/`, `data/syzygy`+`data/gaviota`,
  `data/tb5_test.pkl` (diag positions), proxy checkpoints in scratchpad.
