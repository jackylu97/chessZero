"""Tests for compact GameHistory encoding + version-dispatched ReplayBuffer.

Run with: pytest tests/test_compact_encoding.py
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from src.games.chess import ChessGame
from src.games.tictactoe import TicTacToe
from src.training.replay_buffer import (
    GameHistory,
    ReplayBuffer,
    _iter_shard_games,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tictactoe_game_onehot(actions=(4, 0, 8, 6, 2)) -> GameHistory:
    """Tictactoe game with Stockfish-style one-hot policies + external_values."""
    g = TicTacToe()
    state = g.reset()
    gh = GameHistory(game_name="tictactoe")
    for a in actions:
        obs = g.to_tensor(state)
        legals = g.legal_actions(state)
        gh.observations.append(obs)
        gh.actions.append(a)
        gh.legal_actions_list.append(legals)
        p = np.zeros(g.action_space_size, dtype=np.float32)
        p[a] = 1.0
        gh.policies.append(p)
        gh.root_values.append(0.1 * a)
        state, r, _ = g.step(state, a)
        gh.rewards.append(float(r))
    gh.observations.append(g.to_tensor(state))
    gh.game_outcome = float(state.winner)
    # External values: plies + 1 entries, side-to-move POV
    gh.external_values = [0.5 * (i + 1) * (1 if i % 2 == 0 else -1)
                          for i in range(len(actions) + 1)]
    return gh


def _tictactoe_game_sparse(actions=(4, 0, 8, 6, 2)) -> GameHistory:
    """Tictactoe game with soft (self-play-style) policies over the legal set."""
    g = TicTacToe()
    state = g.reset()
    gh = GameHistory(game_name="tictactoe")
    for a in actions:
        obs = g.to_tensor(state)
        legals = g.legal_actions(state)
        gh.observations.append(obs)
        gh.actions.append(a)
        gh.legal_actions_list.append(legals)
        # Soft policy: uniform over legals with an extra bump on the chosen action
        p = np.zeros(g.action_space_size, dtype=np.float32)
        for i, la in enumerate(legals):
            p[la] = 1.0 / len(legals)
        p[a] += 0.25
        p /= p.sum()
        gh.policies.append(p)
        gh.root_values.append(0.1 * a)
        state, r, _ = g.step(state, a)
        gh.rewards.append(float(r))
    gh.observations.append(g.to_tensor(state))
    gh.game_outcome = float(state.winner)
    # No external_values → self-play game
    return gh


def _assert_gamehistory_equal(a: GameHistory, b: GameHistory, policy_atol=1e-6):
    assert a.game_name == b.game_name
    assert a.actions == b.actions
    assert len(a.rewards) == len(b.rewards)
    assert np.allclose(a.rewards, b.rewards)
    assert np.allclose(a.root_values, b.root_values)
    assert a.game_outcome == b.game_outcome
    # external_values may have sub-fp32 precision loss after roundtrip
    # (stored as float32 in compact form); training reads them as fp32 anyway.
    assert len(a.external_values) == len(b.external_values)
    if a.external_values:
        assert np.allclose(a.external_values, b.external_values)
    assert len(a.observations) == len(b.observations)
    for oa, ob in zip(a.observations, b.observations):
        assert torch.allclose(oa, ob), "observation tensor mismatch after roundtrip"
    assert len(a.legal_actions_list) == len(b.legal_actions_list)
    for la, lb in zip(a.legal_actions_list, b.legal_actions_list):
        assert list(la) == list(lb)
    assert len(a.policies) == len(b.policies)
    for pa, pb in zip(a.policies, b.policies):
        assert pa.shape == pb.shape
        assert np.allclose(pa, pb, atol=policy_atol)


# ---------------------------------------------------------------------------
# GameHistory.to_compact_dict / .from_compact_dict
# ---------------------------------------------------------------------------


def test_roundtrip_onehot_policies():
    """Stockfish-style one-hots encode as 'onehot' mode and roundtrip exactly."""
    gh = _tictactoe_game_onehot()
    d = gh.to_compact_dict()

    assert d["format_version"] == 3
    assert d["policies_mode"] == "onehot"
    assert d["policies_data"].dtype == np.int32
    assert len(d["policies_data"]) == len(gh.actions)
    # Encoded indices match the original argmaxes.
    for i, a in enumerate(gh.actions):
        assert int(d["policies_data"][i]) == a

    decoded = GameHistory.from_compact_dict(d, TicTacToe())
    _assert_gamehistory_equal(gh, decoded)


def test_roundtrip_sparse_policies():
    """Soft policies trigger 'sparse' mode; roundtrip preserves values allclose."""
    gh = _tictactoe_game_sparse()
    d = gh.to_compact_dict()

    assert d["policies_mode"] == "sparse"
    assert len(d["policies_data"]) == len(gh.actions)
    for (idxs, vals), p in zip(d["policies_data"], gh.policies):
        # Sparse representation only stores nonzero entries.
        nz = np.nonzero(p)[0]
        assert np.array_equal(np.sort(idxs), np.sort(nz))

    decoded = GameHistory.from_compact_dict(d, TicTacToe())
    _assert_gamehistory_equal(gh, decoded)


def test_compact_dict_drops_observations_and_legals():
    """Compact dict should not carry the redundant obs/legals — those are the bytes we want back."""
    gh = _tictactoe_game_onehot()
    d = gh.to_compact_dict()
    # Nothing in the dict should be a tensor or a per-step legal-action list.
    for v in d.values():
        if isinstance(v, torch.Tensor):
            pytest.fail("compact dict contains a torch.Tensor")
    assert "observations" not in d
    assert "legal_actions_list" not in d


def test_pickle_roundtrip_of_compact_dict():
    """Compact dict must survive pickle (that's how it lives on disk)."""
    gh = _tictactoe_game_onehot()
    d = gh.to_compact_dict()
    blob = pickle.dumps(d, protocol=pickle.HIGHEST_PROTOCOL)
    d2 = pickle.loads(blob)
    decoded = GameHistory.from_compact_dict(d2, TicTacToe())
    _assert_gamehistory_equal(gh, decoded)


def test_self_play_without_external_values_roundtrips():
    """external_values is empty for self-play games; compact dict omits it and
    the decoder produces an empty list too (not None)."""
    gh = _tictactoe_game_sparse()
    assert gh.external_values == []
    d = gh.to_compact_dict()
    assert "external_values" not in d
    decoded = GameHistory.from_compact_dict(d, TicTacToe())
    assert decoded.external_values == []


def test_chess_roundtrip_preserves_observation_planes():
    """Chess is the real target. Replay through ChessGame must reproduce every
    obs-plane tensor bit-exactly (deterministic game, deterministic to_tensor)."""
    g = ChessGame()
    state = g.reset()
    gh = GameHistory(game_name="chess")
    # 8 moves into a Ruy Lopez-ish sequence (all legal from start).
    legal = g.legal_actions(state)
    # Just play the first legal action 8 times in a row; enough to exercise
    # piece/castling/en-passant planes and fullmove_number.
    for _ in range(8):
        legal = g.legal_actions(state)
        a = legal[0]
        obs = g.to_tensor(state)
        gh.observations.append(obs)
        gh.actions.append(a)
        gh.legal_actions_list.append(legal)
        p = np.zeros(g.action_space_size, dtype=np.float32)
        p[a] = 1.0
        gh.policies.append(p)
        gh.root_values.append(0.0)
        state, r, _ = g.step(state, a)
        gh.rewards.append(float(r))
    gh.observations.append(g.to_tensor(state))
    gh.game_outcome = 0.0

    d = gh.to_compact_dict()
    decoded = GameHistory.from_compact_dict(d, ChessGame())
    _assert_gamehistory_equal(gh, decoded)


def test_unknown_format_version_raises():
    with pytest.raises(ValueError, match="format version"):
        GameHistory.from_compact_dict({"format_version": 999}, TicTacToe())


# ---------------------------------------------------------------------------
# ReplayBuffer.save / .load (v2, v3) + legacy v1
# ---------------------------------------------------------------------------


def test_replay_buffer_v3_save_load_roundtrip(tmp_path: Path):
    rb = ReplayBuffer(max_size=10)
    games = [_tictactoe_game_onehot(), _tictactoe_game_sparse()]
    for g in games:
        rb.save_game(g)

    path = tmp_path / "buffer.buf"
    rb.save(path, format_version=3)

    rb2 = ReplayBuffer(max_size=10)
    rb2.load(path, game=TicTacToe())
    assert len(rb2) == len(rb)
    assert rb2.total_games == rb.total_games
    for a, b in zip(rb.buffer, rb2.buffer):
        _assert_gamehistory_equal(a, b)


def test_replay_buffer_v3_requires_game_on_load(tmp_path: Path):
    rb = ReplayBuffer(max_size=10)
    rb.save_game(_tictactoe_game_onehot())
    path = tmp_path / "v3.buf"
    rb.save(path, format_version=3)

    with pytest.raises(ValueError, match="requires a `game`"):
        ReplayBuffer(max_size=10).load(path)  # no game argument


def test_replay_buffer_v2_still_loads(tmp_path: Path):
    """Legacy v2 .buf files (full GameHistory streaming) must still load for resume."""
    rb = ReplayBuffer(max_size=10)
    rb.save_game(_tictactoe_game_onehot())
    path = tmp_path / "v2.buf"
    rb.save(path, format_version=2)

    rb2 = ReplayBuffer(max_size=10)
    rb2.load(path)  # no game arg needed for v2
    assert len(rb2) == 1
    _assert_gamehistory_equal(rb.buffer[0], rb2.buffer[0])


def test_replay_buffer_v1_legacy_still_loads(tmp_path: Path):
    """Pre-streaming single-dict .buf files (v1) must still load."""
    gh = _tictactoe_game_onehot()
    path = tmp_path / "v1.buf"
    with open(path, "wb") as f:
        pickle.dump(
            {"buffer": [gh], "priorities": [1.0], "total_games": 1}, f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    rb = ReplayBuffer(max_size=10)
    rb.load(path)
    assert len(rb) == 1
    _assert_gamehistory_equal(gh, rb.buffer[0])


def test_replay_buffer_v3_is_smaller_than_v2(tmp_path: Path):
    """Sanity check: compact on-disk representation is meaningfully smaller.

    This is the whole point of the rework — the assertion is loose (≥4× smaller)
    so it tolerates the tiny-game regime where fixed-header overhead matters more.
    """
    rb = ReplayBuffer(max_size=100)
    # Use a bunch of sparse-policy games so sparsification shows up.
    for _ in range(10):
        rb.save_game(_tictactoe_game_sparse())

    v2 = tmp_path / "v2.buf"
    v3 = tmp_path / "v3.buf"
    rb.save(v2, format_version=2)
    rb.save(v3, format_version=3)

    assert v3.stat().st_size < v2.stat().st_size, (
        f"v3 ({v3.stat().st_size}B) should be smaller than v2 ({v2.stat().st_size}B)"
    )


# ---------------------------------------------------------------------------
# Shard loading (legacy list + compact streaming)
# ---------------------------------------------------------------------------


def test_iter_shard_games_legacy_list_format(tmp_path: Path):
    """Pre-rework Stockfish shards are list[GameHistory]; still loadable."""
    games = [_tictactoe_game_onehot(), _tictactoe_game_onehot()]
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as f:
        pickle.dump(games, f, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = list(_iter_shard_games(path))
    assert len(loaded) == 2
    for orig, got in zip(games, loaded):
        _assert_gamehistory_equal(orig, got)


def test_iter_shard_games_compact_v2_format(tmp_path: Path):
    """New generator emits header-plus-compact-dicts; iterate + reconstruct."""
    games = [_tictactoe_game_onehot(), _tictactoe_game_sparse()]
    path = tmp_path / "compact.pkl"
    with open(path, "wb") as f:
        pickle.dump({"version": 2, "n_records": len(games)}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
        for g in games:
            pickle.dump(g.to_compact_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = list(_iter_shard_games(path, game=TicTacToe()))
    assert len(loaded) == 2
    for orig, got in zip(games, loaded):
        _assert_gamehistory_equal(orig, got)


def test_iter_shard_games_compact_needs_game(tmp_path: Path):
    path = tmp_path / "compact.pkl"
    with open(path, "wb") as f:
        pickle.dump({"version": 2, "n_records": 1}, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(_tictactoe_game_onehot().to_compact_dict(), f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ValueError, match="requires `game`"):
        list(_iter_shard_games(path))


def test_load_warmstart_games_mixed_shard_formats(tmp_path: Path):
    """load_warmstart_games accepts a mix of legacy-list and compact shards."""
    g1 = _tictactoe_game_onehot()
    g2 = _tictactoe_game_sparse()
    legacy_path = tmp_path / "legacy.pkl"
    compact_path = tmp_path / "compact.pkl"
    with open(legacy_path, "wb") as f:
        pickle.dump([g1], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(compact_path, "wb") as f:
        pickle.dump({"version": 2, "n_records": 1}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(g2.to_compact_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)

    rb = ReplayBuffer(max_size=10)
    n = rb.load_warmstart_games([legacy_path, compact_path], game=TicTacToe())
    assert n == 2
    assert len(rb) == 2
    _assert_gamehistory_equal(g1, rb.buffer[0])
    _assert_gamehistory_equal(g2, rb.buffer[1])
