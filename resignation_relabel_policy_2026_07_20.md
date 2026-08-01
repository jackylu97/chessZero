# Resignation Relabel Policy — 2026-07-20 (`resign_draws_only`)

## Motivation

The original resignation mechanic (`decisive_signal_plan_2026_06_23.md`) is a
**post-hoc** pass: games always play out fully; if a side's STM root value stays
below `resign_threshold` (−0.9) for `resign_consecutive` (5) of its own moves,
the game was truncated at that ply and relabeled a decisive loss for that side —
regardless of the natural outcome. Its purpose is label protection: a truly lost
position that a weak policy shuffles into a draw should carry a LOSS label.

Three findings (2026-07-19/20 investigation) motivated revising it:

1. **False-positive rate ran 11–22%** (holdout-measured) vs AlphaZero's ≤5%
   target — i.e. ~15% of relabels stamped a loss on a game whose trigger side
   did **not** actually lose. Since ~42% of games trigger, ≈6% of all buffer
   games were draws relabeled as losses — manufactured value-label noise.
2. **The marginal contribution was small**: ~85% of triggered games end in a
   natural loss for the trigger side anyway (the label is a no-op there). The
   mechanic's unique label contribution ≈ 6 pp of buffer decisiveness — it was
   heavily load-bearing in the draw-basin era, and has decayed as conversion
   improved (`win_natural_rate` 0.34→0.45).
3. **Truncation was discarding conversion demonstrations**: for the ~36% of
   games that are true positives, the discarded tail is precisely the winner
   converting into a mate — the position class this project treats as precious
   everywhere else (the seeded-game exemption exists for exactly this reason).
4. **Self-reference**: the trigger is the value head's own sustained verdict,
   and the relabel feeds that verdict back as ground truth — a bias amplifier
   in the same family as the `selfplay_q_ratio` feedback risk. Under the old
   rule, ALL of its label changes were arbitrated solely by the (imperfect)
   value head.

## The policy (`config.resign_draws_only = True`, CLI `--resign-draws-only`)

**Decisive natural outcomes are never overwritten by the value head.**

| Scenario | Trigger fired | Action | Rationale |
|---|---|---|---|
| Decisive, trigger side lost (true positive) | yes | **nothing** — natural loss label, full tail kept | label was already correct; tail = winner's conversion demo |
| Decisive, trigger side WON (comeback) | yes | **nothing** — win kept | "positions the value head called dead sometimes convert" is the anti-pessimism signal the head needs |
| Draw (threefold / no-progress / stalemate / insufficient / **ply-cap**), oracle-free | yes | **flip to loss for trigger side + truncate at resign ply** | the surviving mechanic: protect the decisive label a weak policy shuffled away |
| Draw, final position **TB-certified drawn** | yes | **veto** — draw stands (`resign_tb_veto`) | never override an oracle with the value head's opinion |
| Draw, holdout (~15%) | yes | measurement only (unchanged) | historical FP estimator / behavioral control |
| Any, `tb_filled` | — | skipped (unchanged) | fill's oracle already acted |
| Seeded (`start_fen`), `resign_exempt_seeded` | — | skipped (unchanged) | seed labels are TB-true; tails are the practice material |
| Both sides trigger | — | earliest side wins (unchanged) | rare in practice (decision 2026-07-20) |

Pipeline order (unchanged): deferred TB per-ply relabels → `tb_rollout_fill` →
resignation. Because fill catches any game whose outcome contradicts a decisive
in-TB ply, **resignation's draw-flips only ever operate in the oracle-free gap**.

### Design invariant

> After this change, every label overwrite in the pipeline is either
> **oracle-backed** (TB fill, TB per-ply hard values) or **confined to draws in
> the oracle-free gap**, arbitrated by the value head but FP-monitored on the
> full population and vetoed by TB where the oracle reaches. Decisive outcomes
> are never overwritten by a non-oracle.

## Metrics

- `self_play/resign_trigger_rate` — trigger fired (old ~40% semantics).
- `self_play/resign_trigger_fp_rate` — trigger side did not lose, measured on
  **all** triggered games (~7× the holdout sample; possible because self-play
  always plays out).
- `self_play/resign_tb_veto_rate` — draw-flips vetoed by the TB-drawn final.
- `self_play/resignation_rate` — **semantics change**: now the draw-flip rate
  (expected ~6%), no longer ~40%. Annotate dashboards at the switch step.
- `resign_holdout_rate` / `resign_false_positive_rate` — unchanged estimator,
  kept for series continuity.

## Expected effects / watch list

- Buffer positions +10–20% (kept tails: late-game, decisive, saturated-value)
  → `avg_game_length` up, buffer RAM up, reanalyze calls ~proportionally longer.
- Draw rate in labels: unchanged vs old policy except the removed FP flips
  (~6% of games revert to honest draws) and preserved comeback wins (~1–2%).
- Watch: value MAE and the 150–400cp calibration bin (more saturated-value
  positions could dilute mid-range value learning); per-channel policy CE
  (loser-side tail plies add low-signal policy gradient); `resign_trigger_fp_rate`
  (the value head's pessimism calibration, now precisely measured).
- Rollback: single flag (`--resign-draws-only` off) restores legacy behavior.

## Status

- Implemented 2026-07-20: `src/training/self_play.py::_apply_resignation`,
  config field, CLI flag, trainer metrics, `tests/test_resignation_relabel.py`
  (10 tests, green).
- **Staged, not yet active**: added to `train_logs/launch_ft2_reanalyze.sh`;
  takes effect at the next production restart (one-variable discipline — the
  mixture-recovery experiment resolves first).
