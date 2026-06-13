# Plan: Draw Instrumentation, Adjudication, and Repetition Observability

Date: 2026-06-12
Status: planned (nothing landed; current run `2026_06_12_lr1e3_buf2x` is the
decisive-resampling control and should not be touched mid-run)

## Context

The remaining draw-basin failure mode after the bug-hunt fixes (STM encoding,
past-terminal masking, per-sample value weight) and the June fixes (inverse
dynamics, decisive resampling) is the pure self-reinforcing loop documented in
run `2026_06_02_invdyn_full`: flat value head → MCTS visits mirror the prior
(Grill et al. — no search variant escapes flat Q) → drawish play → self-play
draw rate 0.76→0.92 → every ply labeled `[0,1,0]` → value head stays flat.

Decisive resampling (`decisive_sample_frac=0.5`, landed) amplifies whatever
decisive signal exists but creates none. The three items below attack the
target side directly: know what the draws are (instrumentation), stop labeling
artifact draws as draws (adjudication), and make repetition state visible to
the network (observation planes) so repetition-avoidance is learnable at all.

Sequencing: **instrumentation → adjudication → repetition planes** (planes are
checkpoint-breaking; bundle with the next from-scratch run). The `draw_score`
bump (−0.05 → −0.10/−0.15) is an orthogonal cheap lever, search-time only,
revertible mid-experiment.

---

## 1. Draw-cause instrumentation

### Why
The 0.92 draw rate is one number hiding five causes (mate-less games can end
stalemate / threefold / 75-move / ply-cap). Every design decision below — and
the `max_plies` wall-clock lever in GPU_UTILIZATION_ACTION_ITEMS.md — depends
on the breakdown, which is currently discarded at the moment it's computed.

### The causes already exist as tensors
`src/games/chess_gpu.py:1706-1708` (`_step_batch_impl`):
```python
done = mate | stalemate | seventy_five_move | ply_cap | threefold
```
One line later only `done` and `winner` survive.

### Changes
- `chess_gpu.py`: add `termination: (N,) int8` to the batched state
  (0=running, 1=mate, 2=stalemate, 3=75-move, 4=threefold, 5=ply-cap), updated
  with the same sticky `torch.where(state.done, old, new)` pattern as `winner`
  (line 1722).
- `chess.py`: derive the same enum from `board.outcome().termination` plus the
  ply-cap branch.
- `self_play.py`: copy `state.termination` into `GameHistory` at the three
  places `state.winner` is copied today (lines ~123, ~341, ~545).
- `replay_buffer.py`: `GameHistory.termination` field + round-trip in
  `to_compact_dict` / `from_compact_dict`.
- `trainer.py` logging, next to `self_play/draw_rate`:
  - `self_play/term/{mate,stalemate,seventyfive,threefold,plycap}_rate`
  - avg game length per cause (sizes the dead-draw tail)
  - **mean target draw mass per training batch** — mean P(D) over the WDL
    targets actually sampled in `_train_step`. This is the ≈0.82 quantity from
    bug_hunt §3 and the single best scalar for watching the loop tighten or
    release. One line; targets are already in hand.

### Retroactive option (no code change)
`save_buffer=True` is back on as of today's run. A one-off script can replay
each `.buf` game's action list through python-chess (the compact-decode path
already does this) and classify endings — gives the cause distribution for the
*current* run from its first checkpoint, and doubles as ground truth to
cross-validate the GPU `termination` tensor.

### Tests
Cause cross-validation GPU-vs-python-chess in the existing cross-val harness
(termination is exactly the class of thing the pin-detection bug says to
cross-validate).

---

## 2. Draw adjudication (Stockfish rescoring)

### Why
`_wdl_target_at` (`replay_buffer.py:204`) gives every ply of a
`game_outcome==0` game the one-hot `[0,1,0]`. A 900-ply cap-hit shuffle with a
queen-up side emits 900 "this is a draw" labels — an unfinished game labeled
by a timeout. Precedent: Lc0's rescorer relabels training outcomes via syzygy
tablebases; Stockfish-at-the-terminal is the poor man's version. Stockfish 16
is installed on the box; `eval_to_wdl(α=4, β=2)` already exists.

### Mechanism
1. End of each self-play batch: for games whose termination cause ∈
   `adjudicate_causes`, reconstruct the final position (replay actions through
   python-chess, ~26 ms/game) and run one Stockfish eval (depth ~10-12). Map
   through `eval_to_wdl` to soft `(P_W, P_D, P_L)`.
2. Store as a **new field** `GameHistory.adjudicated_wdl` — NOT
   `external_values`: `bool(g.external_values)` is the warmstart classifier in
   `save_game` and both stratified samplers; overloading it would misroute
   adjudicated games into the warmstart pool and out of decisive resampling.
3. `_wdl_target_at`: for self-play games with `adjudicated_wdl`, emit it
   instead of `[0,1,0]`, with the same per-ply STM flip as the one-hot path.
4. If the adjudicated scalar |P_W − P_L| > `adjudicate_relabel_threshold`
   (~0.5), also relabel `game_outcome = ±1` so decisive resampling picks the
   game up. Soft WDL remains the target; the hard label only drives sampling.

Soft targets remove the thresholding problem: `eval_to_wdl` maps near-zero
evals to ~`[0.12, 0.76, 0.12]`, so a genuinely dead cap-hit fortress is
relabeled to nearly-draw. The sigmoid is the cutoff.

### Decayed blend — required for repetition draws (and right for all causes)
Flat adjudication erases the avoidance signal: labeling the final repetition
position "+0.9 for the better side" teaches the search that repeating costs
nothing. Instead, decay toward the actual outcome over the tail:

```
target(ply) = (1 − λ(ply)) · eval_to_wdl(adjudicated) + λ(ply) · [0,1,0]
```

with λ ramping 0→1 over the last ~30 plies. The trajectory then shows value
declining from +0.9 to 0 as the shuffle approaches the repetition — the
*contrast* is the penalty, and MCTS sees conversion lines hold value while
repetition-bound lines decay. No falsified outcome at the terminal.

### Which causes
| Cause | Adjudicate? | Reasoning |
|---|---|---|
| ply-cap | yes, default-on | not a chess outcome; z=0 is pure artifact |
| threefold / 50-move | yes at cold start, flag-gated | z=0 on a +5 position teaches "material is worthless"; turn off once the policy can convert (instrumentation shows it: threefold-rate falls as mate-rate rises) |
| stalemate | no | real outcome, rare |
| 75-move | treat as 50-move | same shuffle pathology |

### Asymmetry — why no blanket threefold penalty
Repetition is only a failure for the side that stood better; for the worse
side, forcing repetition is correct play. A symmetric "threefold = bad" target
teaches the future defender to avoid the drawing resource. Adjudication is the
oracle for who stood better, and the decayed blend produces the penalty
asymmetrically for free (only the better side's value has anywhere to fall).

**Held in reserve — explicit tail contempt:** if the contrast gradient is
insufficient, blend the better side's tail toward loss-leaning (e.g.
`[0, 0.85, 0.15]`) instead of pure draw. Genuine target-side contempt;
flag-gate and plan to anneal off once mate-rate rises, since at convergence it
biases the value head away from game theory.

### Cost
~230 draws/batch × (26 ms replay + ~50 ms eval) ≈ 17 s serial vs a 30-60 min
self-play batch. Parallelizes over the 112 cores if it ever matters.

### Interaction warning: three multipliers on one small set
Adjudication grows the decisive pool; decisive resampling (frac=0.5) then
oversamples it; PER (α=0.6) prioritizes it again (large TD vs a flat head).
When adjudication lands, drop `decisive_sample_frac` toward 0.3 — adjudication
rebalances more honestly (grows the pool rather than oversampling a tiny one).

### Config
`adjudicate_causes` (default `("ply_cap",)`), `adjudicate_depth` (default 12),
`adjudicate_relabel_threshold` (default 0.5), `adjudicate_decay_plies`
(default 30).

### New code
`src/training/adjudicate.py` — replay + engine pool + eval_to_wdl mapping;
hook at end of self-play in `trainer.py`. Tests with a stubbed engine for
target construction (decay ramp, STM flip, relabel threshold).

---

## 3. Repetition-count + no-progress observation planes

### Why the network currently cannot see repetitions
Threefold (FIDE 9.2) is the same position (placement + side to move +
castling rights + EP availability) occurring three times **anywhere in the
game**, not consecutively. Occurrences can be 30 moves apart.

The observation is 19 planes; none encode repetition state, and plane 18 is
the **fullmove number** — not the halfmove clock — so 50-move progress is also
invisible. `history_frames=8` does not compensate: even the tightest shuffle
(4-ply cycle, occurrences at t, t+4, t+8) puts the first occurrence exactly
one ply outside the 8-frame window at the moment the draw fires. The net can
at most see "occurred once before," which is the wrong resolution — the
decision-relevant difference is between one prior occurrence (harmless) and
two (next repetition ends the game). Longer cycles and spread-out occurrences
are fully invisible, and the halfmove counter (up to 100 plies) is
unrecoverable from any 8-ply window.

AlphaZero hands the network precomputed per-frame repetition-count planes plus
a no-progress plane for exactly this reason. Without them, any repetition
penalty (item 2's decayed blend included) asks the network to avoid a tripwire
it cannot observe at decision time.

### The data already exists in both engines
- GPU: the Zobrist history buffer used for threefold detection
  (`chess_gpu.py:1688`) holds occurrence counts; halfmove clock is in the
  state struct.
- CPU: `board.is_repetition(2)` / `is_repetition(1)` and
  `board.halfmove_clock`.

### Changes
- `chess.py` `to_tensor` + `chess_gpu.py` `to_tensor_batch`: add
  - plane: position occurred ≥1 time before (binary, full-plane broadcast)
  - plane: position occurred ≥2 times before (binary)
  - plane: halfmove clock / 100 (scalar broadcast)
  - bump `num_planes` 19 → 22; consider fixing plane 18 to keep fullmove or
    drop it (decide at implementation).
- Cross-validate GPU vs python-chess on repetition counts in the existing
  harness (the Zobrist reset-on-irreversible semantics must match
  python-chess's since-last-irreversible window).

### Breaking change
`num_planes` changes the network input channel count (×8 history frames):
existing checkpoints and any stored observations are invalidated. Land with
the next from-scratch run. Note: warmstart shards store observations only
implicitly (compact format replays actions), so the asymmetric pool generating
today remains usable — observations are reconstructed at load time by whatever
`to_tensor` is current. Verify this at implementation.

---

## 4. q_ratio (Lc0 Q-blend value targets)

### What it is
`config.q_ratio` already exists (default 0.0, chess preset 0.0):

```
target = q_ratio · q_mcts + (1 − q_ratio) · z
```

where `z` is the one-hot game outcome and `q_mcts` is the MCTS root estimate
captured during self-play. Replaces the blunt game-level label with a graded
per-position target: every ply of a draw currently gets an identical `[0,1,0]`;
under the blend, plies where search saw an advantage keep that signal. This is
the per-position complement to §2's per-game adjudication — and the natural
follow-on to the decayed blend, since a TD-style graded tail is exactly what
q_mcts provides without an external engine.

### Why it is sequenced AFTER the value head un-flattens
Verified during the 2026-06-02 investigation (not landed): from a flat value
head, q_mcts ≈ 0 everywhere, so blending dilutes the decisive `|z|=1` signal
~15% toward draw at q_ratio=0.15 — it amplifies whatever the value head
already knows, including "nothing." Enable only once the head shows spread.
Concretely: gate on a health metric rather than a step count — e.g.
pred_std/target_std from the in-loop diagnostics or the
`eval_checkpoint_health.py` value-spread probe clearing a threshold. Lc0
default is 0.0 (pure z); start at 0.25 and tune.

### The schema gap (the reason it never landed)
The value target is a WDL distribution, but the search only produces scalars:
`GameHistory.root_values` stores the root mean Q (`self_play.py:116/199/320`),
and TensorMCTS backs up scalar `node_value_sum: (N, M)`
(`tensor_mcts.py:291`). Two implementation options:

**Option A — scalar→WDL approximation (no MCTS change, no schema change).**
Map the already-stored `root_values[ply]` scalar through an eval_to_wdl-style
transform to a soft `(P_W, P_D, P_L)` at target-construction time in
`_wdl_target_at`. Cheap (a config flag and ~10 lines in `make_target`), works
retroactively on buffered games, and reanalyze already refreshes
`root_values` in place so blended targets freshen for free. Weakness: P_D is
ill-posed from a scalar — Q=0 cannot distinguish a dead-draw from a
balanced-sharp position, so the mapping manufactures draw probability where
the search may have seen none.

**Option B — true WDL backup (faithful, moderate change).**
Store per-node WDL: extend `node_value_sum`/`child_value_sum` to `(..., 3)`
in TensorMCTS (and the python MCTS node accumulator), backing up the
prediction head's WDL distribution with a W↔L swap per ply parity instead of
scalar negation (D is side-invariant, so the flip is clean). Capture the
root's visit-weighted WDL into a new `GameHistory.root_wdl` list (compact-dict
round-trip included); reanalyze updates it alongside `root_values`. This is
what Lc0 actually does. Costs: TensorMCTS surgery (the value tensors feed the
compiled/Triton PUCT path — Q for selection stays the collapsed scalar, only
backup storage widens), schema bump, ~3× value-storage memory (negligible
next to node_hidden).

**Recommendation:** land Option A first behind the existing `q_ratio` flag —
it is a config-gated afternoon and answers "does Q-blending help at all"
cheaply. Build Option B only if A shows value but the manufactured-P_D
artifact is visible (e.g. predicted P_D inflates on sharp positions).

### Interactions
- §2 decayed adjudication and q_ratio both grade draw targets; they compose
  (adjudication fixes the game-level label, q_ratio adds per-ply texture) but
  should be introduced one at a time to keep runs interpretable.
- Reanalyze: under Option A, blended targets automatically track reanalyzed
  `root_values` — which also means the §"reanalyze health gate" matters more
  once q_ratio > 0 (a collapsed network reanalyzing would write q≈0 into the
  blend). Enable q_ratio and the reanalyze gate together.

### Config / code
`q_ratio` (exists), `q_ratio_health_gate` (new: pred_std/target_std floor
before the blend activates), optional `q_wdl_mode: "scalar_map" | "backup"`.
Touch points: `replay_buffer.py::_wdl_target_at` (blend), `self_play.py` ×3
(root_wdl capture, Option B only), `tensor_mcts.py`/`mcts.py` (Option B only),
`trainer.py` (gate + logging: `train/q_ratio_active`, mean |q_mcts| in
sampled batches).

---

## Recommended order

1. Instrumentation (§1) — days, zero training-behavior risk, sizes everything.
   Run the retroactive `.buf` analysis on the current run's first checkpoint
   immediately.
2. Adjudication (§2) behind flags, ply-cap-only first, then threefold/50-move
   per what §1 shows. Drop `decisive_sample_frac` 0.5 → 0.3 when enabling.
3. q_ratio Option A (§4) behind its health gate, together with the reanalyze
   health gate — armed early, activates itself only once the value head shows
   spread.
4. Repetition/no-progress planes (§3) bundled with the next from-scratch run,
   alongside any other checkpoint-breaking changes queued by then.
5. Reserve: `draw_score` −0.10, explicit tail contempt, q_ratio Option B
   (true WDL backup) if Option A validates the direction.
