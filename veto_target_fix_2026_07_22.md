# Root-Draw Veto Target Fix + POV Parity — 2026-07-22

## Context

At ~103k steps of `2026_07_08_prod_xl_ft_s800`, training loss and endgame
conversion were improving while full-game Elo vs the fixed 60k reference was
flat (−95..−147). A four-way buffer investigation (behavioral audit, target
integrity audit, MCTS interaction audit, phase-wise fit probe) tested four
hypotheses:

| Hypothesis | Verdict |
|---|---|
| Model defers winning to the endgame (midgame passivity) | **Refuted** — advantage onset EARLIER (ply 36→28), games 17% shorter, MG cp-loss flat |
| Endgame eating middlegame capacity | **Not supported** — SF-injection middlegame set statistically flat 60k→102k on all metrics |
| Policy/value target bug | **Refuted** (full 9,940-game scan clean) — but two small latent bugs found, fixed below |
| MCTS interaction we didn't account for | **CONFIRMED** — the root-draw veto was selection-only in the production engine |

## The bug

`--root-terminal-draws` intends: drawing root children pinned to `draw_score`
(−0.05 contempt) during **selection only**, with **0.0 backed up**; a winning
side avoids the draw, a losing side keeps it; π′ targets and the played move
respect the veto. The numpy engine (`mcts.py`) implements exactly this
(terminal children never expand; every visit backs up 0.0;
`select_action_gumbel` then sees genuine ~0 Q).

The tensor engine (`tensor_mcts.py` — ALL production self-play) applied the
pin in Sequential-Halving/PUCT selection but:

1. **`_expand` had no terminal short-circuit** — a pinned child was expanded
   through `recurrent_inference` and backed up the network's phantom-win
   value (the repetition-shuffle regime: the drawing move keeps full winning
   value in the net's eyes).
2. **`_gumbel_finalize` ignored `_root_term_mask` entirely** — the stored
   `gumbel_policy` (π′ training target) kept full phantom-win mass on the
   drawing move, and the A_{n+1} argmax could still play it.
3. **Reanalyze passed no veto in either backend** — and measurably re-poisoned
   targets (+60-80% relative π′ mass on vetoed moves outside TB).
4. **Reanalyze also backfilled opening-mix plies** whose empty policies were a
   deliberate loss mask (no search ran there) — unmasking plies whose
   flat-value roots make π′ tie-chaotic. Plausible contributor to the
   measured opening regression (prior top-1 vs SF-d8 43.3→35.8%, value
   Spearman 0.61→0.34 with rising confidence, 60k→102k).

Buffer measurements (checkpoint_102000.buf): 6-8% of organic plies have
non-empty forced-draw sets; on TB-won positions ~7% of stored targets kept
majority mass on the drawing move; organic threefold still 6% with the veto
"on". Training/serving skew: eval + play_web run the numpy engine WITH the
correct veto — the improvement measured by loss never targeted the drawing
behavior that caps Elo. Why it slipped: the 07-18 validation exercised the
numpy engine only; tensor tests asserted **visits**, never π′/action.

## Fixes (all landed 2026-07-22)

1. **`tensor_mcts.py`** — full numpy parity:
   - `_apply_root_term_mask()` (called on every `run_batch_gpu` AND
     `run_batch` shim entry): maps the mask onto root slots, severs any
     reused subtree, clears phantom stats. Also kills a latent staleness bug
     (the shim never reset `_root_term_mask`).
   - `_expand`: pinned root children are TERMINAL — never linked
     (`child_node_idx` stays −1), node reward zeroed, `leaf_value = 0.0`
     (the recurrent_inference output is discarded — same trade as numpy).
   - `_gumbel_finalize`: explicit `raw_q_m → 0.0` pin at masked slots so π′
     and A_{n+1} see the draw backup (belt-and-braces over the terminal fix).
   - `run_batch` shim gained `forced_draw_actions` (numpy signature parity).
   - Verified: phantom-win repro now byte-matches numpy (action, π′ to 4dp,
     root_value 0.870 vs pre-fix 0.900 with π′(F) 0.1185→0.0000).
2. **`trainer._reanalyze`** — veto-aware and opening-aware:
   - Replays each sampled game on a python-chess board
     (`Trainer._forced_draw_sets`, gated scan: ~0.5 ms/ply, ~1 min per
     1024-game call) and passes `forced_draw_mask`/`forced_draw_actions` to
     the search — same veto the original self-play targets ran under.
   - Skips positions whose stored policy is empty (`_policy_is_empty`) —
     opening-mix plies keep their loss mask. New metric:
     `reanalyze/opening_plies_skipped`.
3. **POV parity (latent, fixed while in there)**:
   - `make_target._outcome_onehot` (and the scalar td=−1 path) now derive STM
     color from `start_fen` parity instead of assuming ply 0 = white. Old
     behavior inverted value targets on black-start seeds at the terminal
     index (923 buffer positions) and would have flipped EVERY seeded ply the
     moment `tb_value_weight < 1`.
   - `game_outcome` is now white-POV EVERYWHERE. Legacy anchor dicts
     (first-mover POV, `gen_tb_anchor_games` pre-fix) are normalized on load:
     compact dicts now carry `outcome_pov="white"`; unmarked `tb_authored`
     black-start games get a one-shot sign flip in `from_compact_dict` (.buf
     resumes) and `_inject_tb_anchor_games` (the on-disk archive, which lacks
     the `tb_authored` key until injection stamps it). The archive itself is
     unchanged on disk; the generator now writes white-POV + marker.

## Tests

- `test_tensor_mcts_terminal_draws.py`: 4 new tests asserting
  **gumbel_policy / gumbel_action / zero backup / no expansion** under the
  veto (winning side avoids, losing side keeps), plus a tensor↔numpy parity
  test — the quantities the old visits-only tests could not see.
- `test_reanalyze_tensor_mcts.py`: `_forced_draw_sets` replay unit test
  (repetition at the right ply only), veto-mask-passed integration test
  (monkeypatched `run_batch_gpu` capture), opening-mask preservation test.
- `test_outcome_pov.py` (new): WDL + scalar parity on white/black starts,
  compact round-trip marker, legacy-anchor flip, seeded-game no-flip.

## Expected effects / watch list

- Self-play: organic threefold rate should fall from ~6% (A_{n+1} now
  actually avoids the draw when winning); stored π′ mass on vetoed moves
  → ~0 for winning side. `reanalyze/opening_plies_skipped` ≈ 6 × batch games.
- Value: root_value no longer inflated by phantom draw-child contributions
  (small; root_values are training-inert in this config).
- Opening regression: the reanalyze unmasking stops — expect the opening
  stratum (prior top-1 / value Spearman vs SF-d8, `phase_*` probe artifacts
  in the session scratchpad) to stabilize; recovery of already-learned noise
  takes ~1 buffer turnover (~10k steps).
- Anchors: black-start anchors (~half) now carry white-POV outcomes; their
  terminal-ply value targets are unchanged in effect (the two POV errors
  previously cancelled) — no metric jump expected.
- NOT expected to change: TB-ply targets (hard values dominate), resign
  policy, mixture schedule. This bundle deliberately changes only the
  veto/target-consistency layer — one-variable discipline at the run level.

## Rollout

Code-side only (no new flags). Takes effect at the next production restart —
bundled via the standing `train_logs/launch_ft2_reanalyze.sh` procedure
(restart at next checkpoint, .buf quiescence rule, verify
"Loaded replay buffer (N games)").
