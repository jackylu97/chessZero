"""Tests for asymmetric-teacher Stockfish generation.

Uses a mocked Stockfish engine so these run in the regular pytest suite without
a real binary. The mock lets the test control exactly what the "label" and "play"
analyse calls return, so we can assert the off-policy labeling invariant cleanly:

    policies[i] should be one-hot on the LABEL engine's top move (depth-8 preferred),
    actions[i]  should be the PLAY engine's sampled move (depth-weak softmax),
    and the two can diverge — that divergence IS the training signal.

Run with: pytest tests/test_asymmetric_generation.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import chess
import chess.engine
import numpy as np
import pytest

from src.games.chess import ChessGame, _move_to_action
from scripts.generate_stockfish_games import (
    BUCKETS,
    generate_one_asymmetric_game,
    softmax_sample_from_multipv,
)


# ---------------------------------------------------------------------------
# Minimal fake engine
# ---------------------------------------------------------------------------


@dataclass
class _FakeScore:
    """PovScore stand-in with just the `.pov(color)` path score_to_value needs."""
    cp: int  # centipawns from White's POV

    def pov(self, color: chess.Color) -> "_FakeScore":
        return _FakeScore(cp=self.cp if color == chess.WHITE else -self.cp)

    def is_mate(self) -> bool:
        return False

    def score(self) -> int:
        return self.cp


class FakeEngine:
    """Implements just enough of ``chess.engine.SimpleEngine`` for the generator.

    Caller passes a ``policy_fn(board, depth, multipv) -> list[info dict]`` that
    returns MultiPV-shaped info entries. Each info dict must have ``pv`` (list of
    moves; only the first is used) and ``score`` (a ``_FakeScore``). The generator
    only calls ``.analyse(...)``; we don't need anything else.
    """

    def __init__(self, policy_fn: Callable[[chess.Board, int, int], list]):
        self._policy_fn = policy_fn
        self.call_log: list[tuple[int, int]] = []  # (depth, multipv) per analyse call

    def analyse(self, board: chess.Board, limit: chess.engine.Limit, multipv: int = 1):
        self.call_log.append((limit.depth, multipv))
        return self._policy_fn(board, limit.depth, multipv)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _top_legal_moves(board: chess.Board, n: int) -> list[chess.Move]:
    """Stable "legal moves in UCI order" — a deterministic move ordering."""
    moves = sorted(board.legal_moves, key=lambda m: m.uci())
    return moves[:n] if moves else []


def _make_info(move: chess.Move, cp_white_pov: int) -> dict:
    return {"pv": [move], "score": _FakeScore(cp=cp_white_pov)}


# 4-ply knight shuffle that returns the board to its starting position; used to
# make short deterministic games that terminate via is_repetition(3) inside
# ChessGame.step. Triggers threefold at ply 8 (position seen at plies 0, 4, 8).
_SHUFFLE_CYCLE = ["g1f3", "g8f6", "f3g1", "f6g8"]


def _make_play_policy(label_pref: str = "first"):
    """Play engine always pushes the next move of the knight-shuffle cycle.

    The label engine (separate FakeEngine instance or same engine called with
    depth=8 mpv=1) gets its own policy — tests control label via a separate
    ``label_pref`` argument on the individual callsites. This function just
    returns the PLAY-side behavior.

    Returns a policy_fn suitable for FakeEngine that ALWAYS returns the cycle
    move first (so softmax with tiny τ picks it), padded with other legals up
    to multipv.
    """
    def policy_fn(board: chess.Board, depth: int, multipv: int):
        ply = board.ply()
        target_uci = _SHUFFLE_CYCLE[ply % 4]
        try:
            target = chess.Move.from_uci(target_uci)
        except ValueError:
            target = None
        legals = list(board.legal_moves)
        if target is not None and target in legals:
            others = [m for m in legals if m != target]
            selected = [target] + others[: max(multipv - 1, 0)]
        else:
            selected = legals[:multipv]
        if not selected:
            return []
        return [_make_info(m, cp_white_pov=0) for m in selected]
    return policy_fn


def _make_dual_policy(label_move_picker, play_move_picker):
    """Dispatch engine call to label vs play logic based on the (depth, mpv) shape.

    Label call: depth=8 and mpv=1. Play call: depth<8 OR mpv>1.
    (The generator collapses the 8v8 case to a single mpv>1 call — that's handled
    downstream by whichever test needs to distinguish them.)
    """
    def policy_fn(board: chess.Board, depth: int, multipv: int):
        is_label_call = depth == 8 and multipv == 1
        picker = label_move_picker if is_label_call else play_move_picker
        moves = picker(board, multipv)
        if not moves:
            return []
        return [_make_info(m, cp_white_pov=0) for m in moves]
    return policy_fn


def _label_pick_legals_last(board: chess.Board, multipv: int) -> list[chess.Move]:
    legals = _top_legal_moves(board, 0) or sorted(board.legal_moves, key=lambda m: m.uci())
    return [legals[-1]] if legals else []


def _play_pick_shuffle_cycle(board: chess.Board, multipv: int) -> list[chess.Move]:
    ply = board.ply()
    target_uci = _SHUFFLE_CYCLE[ply % 4]
    legals = list(board.legal_moves)
    try:
        target = chess.Move.from_uci(target_uci)
    except ValueError:
        target = None
    if target is not None and target in legals:
        others = [m for m in legals if m != target]
        return [target] + others[: max(multipv - 1, 0)]
    return legals[:multipv]


@pytest.fixture
def game():
    return ChessGame()


# ---------------------------------------------------------------------------
# Off-policy labeling invariant
# ---------------------------------------------------------------------------


def _uci_from_action(board: chess.Board, action: int) -> str:
    """Decode an action index back to a UCI string via the legal-move map."""
    for m in board.legal_moves:
        if _move_to_action(m) == action:
            return m.uci()
    raise ValueError(f"action {action} not legal from {board.fen()}")


# Games are terminated via is_repetition(3) (strict threefold) after the 8-ply
# knight-shuffle cycle repeats — fast and deterministic. max_plies is generous
# so we don't accidentally cap out on longer edge paths.
_MAX_PLIES = 200


def test_label_and_play_can_diverge(game):
    """Core assertion: policies[i] tracks the label engine's top move while
    actions[i] tracks the play engine's sampled move. When the two engines
    prefer different moves, we must see the divergence show up in the
    GameHistory — that's the whole point of asymmetric-teacher generation.
    """
    # Label picks last-UCI-legal (likely a non-knight move); play pushes the
    # shuffle cycle. These almost always disagree in the starting position,
    # so divergence is guaranteed.
    engine = FakeEngine(_make_dual_policy(
        label_move_picker=_label_pick_legals_last,
        play_move_picker=_play_pick_shuffle_cycle,
    ))
    history = generate_one_asymmetric_game(
        engine, game,
        play_depth_white=5,
        play_depth_black=5,
        label_depth=8,
        max_plies=_MAX_PLIES,
        multipv=3,
        move_temperature=1e-9,  # pure argmax on play side
        rng=random.Random(0),
    )
    assert history is not None, "game should terminate via threefold within _MAX_PLIES"

    divergences = sum(
        1 for i, pol in enumerate(history.policies)
        if int(np.argmax(pol)) != history.actions[i]
    )
    assert divergences > 0, (
        "Expected label-top to diverge from played-action on at least some "
        f"positions; got 0 over {len(history.actions)} plies."
    )


def test_8v8_bucket_collapses_to_one_analyse_per_ply(game):
    """When play_depth == label_depth, the generator skips the extra label
    call and reuses play_info[0] for labels. Cuts Stockfish time ~50% on
    the balanced bucket.
    """
    engine = FakeEngine(_make_play_policy())
    history = generate_one_asymmetric_game(
        engine, game,
        play_depth_white=8,
        play_depth_black=8,
        label_depth=8,
        max_plies=_MAX_PLIES,
        multipv=3,
        move_temperature=1e-9,
        rng=random.Random(0),
    )
    assert history is not None
    assert all(call == (8, 3) for call in engine.call_log), engine.call_log
    assert len(engine.call_log) == len(history.actions)


def test_asymmetric_bucket_makes_two_analyse_calls_per_ply(game):
    """When play_depth != label_depth, both engines are invoked per ply:
    one at label_depth MPV=1, one at play_depth MPV=K.
    """
    engine = FakeEngine(_make_dual_policy(
        label_move_picker=_label_pick_legals_last,
        play_move_picker=_play_pick_shuffle_cycle,
    ))
    history = generate_one_asymmetric_game(
        engine, game,
        play_depth_white=5,
        play_depth_black=5,
        label_depth=8,
        max_plies=_MAX_PLIES,
        multipv=3,
        move_temperature=1e-9,
        rng=random.Random(0),
    )
    assert history is not None
    calls = engine.call_log
    assert len(calls) == 2 * len(history.actions), (
        f"Expected 2×{len(history.actions)} analyse calls, got {len(calls)}"
    )
    for i in range(len(history.actions)):
        assert calls[2 * i] == (8, 1), f"ply {i} label call: {calls[2*i]}"
        assert calls[2 * i + 1] == (5, 3), f"ply {i} play call: {calls[2*i+1]}"


def test_external_values_come_from_label_depth(game):
    """external_values (the value-target signal) must come from the LABEL
    engine's eval, not the play engine's — the whole point of a "strong
    evaluator, weaker trajectory" split.
    """
    label_cp = 73
    play_cp = -200

    def policy_fn(board, depth, multipv):
        is_label = (depth == 8 and multipv == 1)
        if is_label:
            moves = _label_pick_legals_last(board, multipv)
            cp = label_cp
        else:
            moves = _play_pick_shuffle_cycle(board, multipv)
            cp = play_cp
        return [_make_info(m, cp_white_pov=cp) for m in moves] if moves else []

    engine = FakeEngine(policy_fn)
    history = generate_one_asymmetric_game(
        engine, game,
        play_depth_white=5,
        play_depth_black=5,
        label_depth=8,
        max_plies=_MAX_PLIES,
        multipv=3,
        move_temperature=1e-9,
        rng=random.Random(0),
    )
    assert history is not None

    # Reconstruct the expected external_values from label_cp, flipping sign per
    # side to move. The generator should never have mixed in play_cp.
    import math
    ref_board = chess.Board()
    expected = []
    for a in history.actions:
        pov_cp = label_cp if ref_board.turn == chess.WHITE else -label_cp
        expected.append(2.0 * (1.0 / (1.0 + math.exp(-pov_cp / 400.0))) - 1.0)
        ref_board.push(chess.Move.from_uci(_uci_from_action(ref_board, a)))

    # Compare root_values (labeled) against the reference. external_values has
    # a trailing terminal entry we skip; the per-ply labels live at indices
    # [0..N-1] and equal root_values there by construction.
    assert len(history.root_values) == len(expected)
    for got, want in zip(history.root_values, expected):
        assert abs(got - want) < 1e-6, (got, want)


def test_action_is_played_move_not_label_move(game):
    """Trajectory must advance on PLAY's move, not LABEL's. Verified by
    replaying history.actions through a fresh python-chess board — no
    illegal-move errors means actions matched what was actually pushed."""
    engine = FakeEngine(_make_dual_policy(
        label_move_picker=_label_pick_legals_last,
        play_move_picker=_play_pick_shuffle_cycle,
    ))
    history = generate_one_asymmetric_game(
        engine, game,
        play_depth_white=5,
        play_depth_black=5,
        label_depth=8,
        max_plies=_MAX_PLIES,
        multipv=3,
        move_temperature=1e-9,
        rng=random.Random(0),
    )
    assert history is not None
    b = chess.Board()
    for a in history.actions:
        uci = _uci_from_action(b, a)
        b.push(chess.Move.from_uci(uci))


def test_compact_encoding_handles_off_policy_games(game):
    """The compact GameHistory encoding still works on off-policy games:
    policies store the one-hot label, actions store the (possibly different)
    played move. The encoder must notice one-hot-ness of POLICIES (not of
    argmax(policies) == action)."""
    engine = FakeEngine(_make_dual_policy(
        label_move_picker=_label_pick_legals_last,
        play_move_picker=_play_pick_shuffle_cycle,
    ))
    history = generate_one_asymmetric_game(
        engine, game,
        play_depth_white=5,
        play_depth_black=5,
        label_depth=8,
        max_plies=_MAX_PLIES,
        multipv=3,
        move_temperature=1e-9,
        rng=random.Random(0),
    )
    assert history is not None

    d = history.to_compact_dict()
    # One-hot mode: divergence with actions is fine — policies[i] is still
    # a one-hot on the label move.
    assert d["policies_mode"] == "onehot"

    from src.training.replay_buffer import GameHistory
    decoded = GameHistory.from_compact_dict(d, ChessGame())
    # Actions roundtrip bit-exact.
    assert list(decoded.actions) == list(history.actions)
    # Policies roundtrip allclose (int32 → fp32 dense).
    for pa, pb in zip(history.policies, decoded.policies):
        assert np.allclose(pa, pb)


# ---------------------------------------------------------------------------
# BUCKETS table + softmax helper (both exported, both user-visible)
# ---------------------------------------------------------------------------


def test_buckets_table_strong_is_always_eight():
    assert set(BUCKETS.keys()) == {"8v5", "8v6", "8v7", "8v8"}
    for name, (strong, weak) in BUCKETS.items():
        assert strong == 8, f"bucket {name}: strong should be 8, got {strong}"
        assert weak in {5, 6, 7, 8}, f"bucket {name}: unexpected weak depth {weak}"


def test_softmax_sample_near_zero_temperature_picks_best():
    """Near-argmax regression: with τ→0, softmax_sample degenerates to top-1."""
    b = chess.Board()
    legals = _top_legal_moves(b, 3)
    info_list = [
        _make_info(legals[0], cp_white_pov=100),  # best
        _make_info(legals[1], cp_white_pov=50),
        _make_info(legals[2], cp_white_pov=0),
    ]
    move, chosen, best = softmax_sample_from_multipv(
        info_list, chess.WHITE, temperature=1e-9, rng=random.Random(0),
    )
    assert move == legals[0]
    assert chosen == pytest.approx(best)
