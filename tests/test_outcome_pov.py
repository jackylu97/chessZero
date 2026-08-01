"""game_outcome POV conventions (2026-07-22).

Buffer-wide convention: ``game_outcome`` is WHITE-POV. ``make_target`` derives
side-to-move from start_fen parity (ply 0 is white only for standard starts).
Before the fix, ``_outcome_onehot`` assumed ply 0 = white, inverting value
targets on black-start seeded games wherever the TB blend didn't mask it
(923 buffer positions at the terminal index; every ply the moment
tb_value_weight < 1). Legacy anchor archives stored FIRST-MOVER POV and are
normalized on load via the ``outcome_pov`` marker.
"""

from __future__ import annotations

import numpy as np
import torch

from src.training.replay_buffer import GameHistory

# Black to move; both sides have simple legal moves.
_BLACK_START_FEN = "4k3/8/8/8/8/8/8/QQ2K3 b - - 0 1"

WDL_W = np.array([1.0, 0.0, 0.0], dtype=np.float32)
WDL_L = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _synthetic_history(n_ply=4, outcome=1.0, start_fen=None, game_name="chess"):
    """Minimal decisive GameHistory; observations are zero tensors (material
    weight is 0 in these tests so their content is irrelevant)."""
    A = 8
    gh = GameHistory(game_name=game_name)
    gh.start_fen = start_fen
    gh.game_outcome = float(outcome)
    for _ in range(n_ply):
        gh.observations.append(torch.zeros(1, 3, 3))
        gh.legal_actions_list.append(list(range(A)))
        pol = np.full(A, 1.0 / A, dtype=np.float32)
        gh.policies.append(pol)
        gh.actions.append(0)
        gh.root_values.append(0.0)
        gh.rewards.append(0.0)
    gh.observations.append(torch.zeros(1, 3, 3))
    return gh


def _value_targets(gh, value_head_type="wdl", td_steps=-1):
    _obs, _mask, _actions, values, _rewards, _policies = gh.make_target(
        state_index=0, num_unroll_steps=len(gh.actions) - 1,
        td_steps=td_steps, discount=1.0, action_space_size=8,
        value_head_type=value_head_type,
        material_value_weight=0.0, tb_value_weight=0.0,
    )
    return values


def test_wdl_outcome_parity_standard_start():
    """White wins from a standard start: even plies (white STM) get the W
    one-hot, odd plies the L one-hot."""
    gh = _synthetic_history(outcome=1.0, start_fen=None)
    values = _value_targets(gh)
    np.testing.assert_array_equal(values[0], WDL_W)
    np.testing.assert_array_equal(values[1], WDL_L)
    np.testing.assert_array_equal(values[2], WDL_W)


def test_wdl_outcome_parity_black_start():
    """Black-start seed, black (the starter) wins → outcome is -1 WHITE-POV.
    Ply 0 STM is BLACK = the winner → W one-hot. The pre-fix code produced
    the exact inverse (L at ply 0)."""
    gh = _synthetic_history(outcome=-1.0, start_fen=_BLACK_START_FEN)
    values = _value_targets(gh)
    np.testing.assert_array_equal(values[0], WDL_W)
    np.testing.assert_array_equal(values[1], WDL_L)
    np.testing.assert_array_equal(values[2], WDL_W)


def test_scalar_outcome_parity_black_start():
    """Same parity rule on the scalar (support) MC path (td_steps=-1)."""
    gh = _synthetic_history(outcome=-1.0, start_fen=_BLACK_START_FEN)
    values = _value_targets(gh, value_head_type="support")
    assert values[0] == 1.0   # black STM at ply 0, black won
    assert values[1] == -1.0
    assert values[2] == 1.0


def _anchor_history():
    """A real (replayable) black-start chess game marked tb_authored."""
    from src.games.chess import ChessGame

    g = ChessGame()
    A = g.action_space_size
    gh = GameHistory(game_name="chess")
    gh.start_fen = _BLACK_START_FEN
    gh.tb_authored = True
    gh.game_outcome = -1.0  # black (first mover) won — WHITE-POV convention
    state = g.reset_from_fen(_BLACK_START_FEN)
    for _ in range(2):
        legals = g.legal_actions(state)
        a = int(legals[0])
        gh.observations.append(g.to_tensor(state))
        gh.legal_actions_list.append(list(legals))
        pol = np.zeros(A, dtype=np.float32)
        pol[a] = 1.0
        gh.policies.append(pol)
        gh.actions.append(a)
        gh.root_values.append(0.0)
        gh.rewards.append(0.0)
        state, _, _ = g.step(state, a)
    gh.observations.append(g.to_tensor(state))
    return g, gh


def test_compact_roundtrip_marks_and_keeps_white_pov():
    g, gh = _anchor_history()
    d = gh.to_compact_dict()
    assert d["outcome_pov"] == "white"
    gh2 = GameHistory.from_compact_dict(d, g)
    assert gh2.game_outcome == -1.0  # marked dict: no flip


def test_legacy_anchor_dict_is_flipped_on_load():
    """Unmarked tb_authored black-start dict = legacy anchor archive
    (first-mover POV: +1 means the BLACK first mover won) → normalized to
    white-POV (-1) exactly once."""
    g, gh = _anchor_history()
    d = gh.to_compact_dict()
    del d["outcome_pov"]
    d["game_outcome"] = 1.0  # legacy first-mover POV
    gh2 = GameHistory.from_compact_dict(d, g)
    assert gh2.game_outcome == -1.0


def test_legacy_seeded_dict_is_not_flipped():
    """Seeded (non-anchor) games were ALWAYS white-POV — the migration must
    not touch them even when the marker is absent."""
    g, gh = _anchor_history()
    gh.tb_authored = False
    d = gh.to_compact_dict()
    d.pop("outcome_pov", None)
    assert "tb_authored" not in d
    d["game_outcome"] = -1.0  # already white-POV
    gh2 = GameHistory.from_compact_dict(d, g)
    assert gh2.game_outcome == -1.0
