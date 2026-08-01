"""resign_draws_only relabel policy (2026-07-20).

See resignation_relabel_policy_2026_07_20.md. Verifies, against the scenario
matrix, that under the new policy:
1. Decisive natural outcomes are NEVER overwritten (true losses keep label AND
   tail; comeback wins keep the win).
2. Oracle-free draws with a trigger are flipped to a loss + truncated (the
   surviving mechanic), including ply-cap draws.
3. A TB-certified drawn FINAL position vetoes the flip.
4. Holdout still measures FP; full-population trigger stats are set on every
   triggered game.
5. Seeded (with exemption) and tb_filled games remain untouched.
6. Legacy mode (flag off) still flips + truncates decisive games (back-compat).
7. Black-to-move seed parity attributes the resigning color correctly.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from src.training.replay_buffer import GameHistory
from src.training.self_play import _apply_resignation

NAN = float("nan")


def _cfg(**over):
    base = dict(resign_enabled=True, resign_threshold=-0.9, resign_consecutive=3,
                resign_holdout_frac=0.0, resign_exempt_seeded=True,
                resign_draws_only=True)
    base.update(over)
    return SimpleNamespace(**base)


def _game(outcome, n_ply=12, trigger_side=1, start_fen=None, tb_values=None,
          tb_filled=False):
    """Synthetic history: `trigger_side` (0=ply-parity even, 1=odd) has root_values
    below threshold on its last 3 own moves; the other side is fine."""
    h = GameHistory(game_name="chess")
    h.start_fen = start_fen
    h.tb_filled = tb_filled
    h.game_outcome = float(outcome)
    h.actions = list(range(n_ply))
    h.policies = [(None, None)] * n_ply
    h.rewards = [0.0] * n_ply
    h.legal_actions_list = [[0]] * n_ply
    h.observations = [None] * (n_ply + 1)
    rv = []
    for p in range(n_ply):
        if p % 2 == trigger_side and p >= n_ply - 6:
            rv.append(-0.95)          # trigger side despairing near the end
        else:
            rv.append(0.2)
    h.root_values = rv
    if tb_values is not None:
        h.tablebase_values = tb_values
    return h


def _resign_ply(n_ply=12, trigger_side=1, need=3):
    """Ply at which the synthetic trigger fires (3rd own-move below thr)."""
    own = [p for p in range(n_ply) if p % 2 == trigger_side and p >= n_ply - 6]
    return own[need - 1]


def test_decisive_loss_untouched_tail_kept():
    # Odd plies = second mover triggers; outcome -1: with default (white-start)
    # parity, odd-ply trigger side is BLACK; black "actually lost" iff outcome=+1.
    # Use trigger_side matching the loser: outcome=-1 -> white lost -> white(even) triggers.
    h = _game(outcome=-1.0, trigger_side=0)
    _apply_resignation([h], _cfg())
    assert h.game_outcome == -1.0
    assert len(h.actions) == 12          # tail kept
    assert not h.resigned
    assert h.resign_triggered
    assert not h.resign_trigger_fp       # trigger side really lost


def test_comeback_win_kept():
    # Even-ply (white) side triggers but the game outcome is WHITE WINS (+1).
    h = _game(outcome=1.0, trigger_side=0)
    _apply_resignation([h], _cfg())
    assert h.game_outcome == 1.0         # win preserved
    assert len(h.actions) == 12
    assert not h.resigned
    assert h.resign_triggered and h.resign_trigger_fp


def test_draw_flipped_and_truncated():
    h = _game(outcome=0.0, trigger_side=0)   # white triggers
    p = _resign_ply(trigger_side=0)
    _apply_resignation([h], _cfg())
    assert h.resigned
    assert h.game_outcome == -1.0        # white resigns -> black wins
    assert len(h.actions) == p           # truncated at resign ply
    assert len(h.observations) == p + 1
    assert h.resign_triggered and h.resign_trigger_fp  # (it was a draw)


def test_plycap_draw_treated_as_draw():
    # Ply-cap "draws" carry game_outcome 0.0 like any draw — flip-eligible.
    h = _game(outcome=0.0, trigger_side=1)
    _apply_resignation([h], _cfg())
    assert h.resigned and h.game_outcome == 1.0   # black (odd, white-start) resigns


def test_tb_drawn_final_vetoes_flip():
    tbv = [NAN] * 10 + [0.0, 0.0]        # final in-TB plies certified DRAWN
    h = _game(outcome=0.0, trigger_side=0, tb_values=tbv)
    _apply_resignation([h], _cfg())
    assert not h.resigned
    assert h.game_outcome == 0.0         # draw stands — oracle beats value head
    assert h.resign_tb_veto


def test_tb_decisive_final_does_not_veto():
    tbv = [NAN] * 10 + [1.0, -1.0]       # final in-TB plies decisive
    h = _game(outcome=0.0, trigger_side=0, tb_values=tbv)
    _apply_resignation([h], _cfg())
    assert h.resigned                    # no veto; flip proceeds


def test_holdout_measures_without_acting():
    h = _game(outcome=0.0, trigger_side=0)
    _apply_resignation([h], _cfg(resign_holdout_frac=1.0))
    assert h.resign_holdout
    assert h.resign_false_positive       # drew, would-be resigner didn't lose
    assert not h.resigned and h.game_outcome == 0.0 and len(h.actions) == 12


def test_seeded_exempt_and_tb_filled_exempt():
    seeded = _game(outcome=0.0, trigger_side=0, start_fen="8/8/8/8/8/4k3/8/4K2R w K - 0 1")
    filled = _game(outcome=0.0, trigger_side=0, tb_filled=True)
    _apply_resignation([seeded, filled], _cfg())
    for h in (seeded, filled):
        assert not h.resigned and not h.resign_triggered and h.game_outcome == 0.0


def test_legacy_mode_still_flips_decisive():
    # Back-compat: flag off -> old behavior (comeback WIN gets flipped+truncated).
    h = _game(outcome=1.0, trigger_side=0)
    p = _resign_ply(trigger_side=0)
    _apply_resignation([h], _cfg(resign_draws_only=False))
    assert h.resigned and h.game_outcome == -1.0
    assert len(h.actions) == p


def test_black_start_seed_parity_without_exemption():
    # Seeded game starting BLACK to move; even plies are BLACK's moves. Even-ply
    # trigger => BLACK resigns => outcome +1 (white/player-1 wins).
    fen = "4k3/8/8/8/8/8/8/QQQ1K3 b - - 0 1"
    h = _game(outcome=0.0, trigger_side=0, start_fen=fen)
    _apply_resignation([h], _cfg(resign_exempt_seeded=False))
    assert h.resigned and h.game_outcome == 1.0


def _no_trigger_game(outcome, n_ply=12):
    h = _game(outcome, n_ply=n_ply)
    h.root_values = [0.2] * n_ply        # never below threshold
    return h


def test_row_a_decisive_no_trigger_untouched():
    h = _no_trigger_game(outcome=1.0)
    _apply_resignation([h], _cfg())
    assert h.game_outcome == 1.0 and len(h.actions) == 12
    assert not h.resigned and not h.resign_triggered and not h.resign_holdout


def test_row_h_draw_no_trigger_untouched():
    h = _no_trigger_game(outcome=0.0)
    _apply_resignation([h], _cfg())
    assert h.game_outcome == 0.0 and len(h.actions) == 12
    assert not h.resigned and not h.resign_triggered


def test_both_sides_trigger_earliest_wins():
    # Both sides despair, but the EVEN side (white, white-start) accumulates its
    # 3rd sub-threshold own-move first: even plies 0,2,4 vs odd plies 3,5,7.
    h = _game(outcome=0.0)
    rv = [0.2] * 12
    for p in (0, 2, 4):
        rv[p] = -0.95                    # white's 3rd hit at ply 4
    for p in (3, 5, 7):
        rv[p] = -0.95                    # black's 3rd hit at ply 7 (later)
    h.root_values = rv
    _apply_resignation([h], _cfg())
    assert h.resigned
    assert h.game_outcome == -1.0        # WHITE (earliest) resigns -> black wins
    assert len(h.actions) == 4           # truncated at white's trigger ply


def test_flip_clears_draw_type_flags_and_truncates_all_arrays():
    h = _game(outcome=0.0, trigger_side=0)
    h.draw_by_repetition = True
    h.draw_by_no_progress = True
    p = _resign_ply(trigger_side=0)
    _apply_resignation([h], _cfg())
    assert h.resigned and not h.draw_by_repetition and not h.draw_by_no_progress
    for arr in (h.actions, h.policies, h.root_values, h.rewards, h.legal_actions_list):
        assert len(arr) == p
    assert len(h.observations) == p + 1


def test_holdout_on_decisive_keeps_metric_semantics():
    # Holdout sampling happens on ALL triggered games (metric continuity with the
    # historical estimator), even though under draws_only a decisive game would
    # be untouched regardless. Natural outcome must survive either way.
    h = _game(outcome=-1.0, trigger_side=0)   # true positive
    _apply_resignation([h], _cfg(resign_holdout_frac=1.0))
    assert h.resign_holdout and not h.resign_false_positive
    assert h.game_outcome == -1.0 and len(h.actions) == 12 and not h.resigned
