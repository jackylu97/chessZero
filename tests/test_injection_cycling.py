"""Stockfish injection pool CYCLES on exhaustion (2026-07-26) rather than
disabling. Warmstart is the only opening teacher; the original bounded
(non-cycling) pool silently starved it to 0% once the 128k-game supply ran dry
and the crash-recovery restarts reloaded .buf files that exclude warmstart
games. On exhaustion the pool must re-serve from the top so the supply never
vanishes (the mixture schedule still governs the actual share)."""

import threading
import types

from src.training.trainer import MuZeroTrainer


class _Buf:
    def __init__(self):
        self.games = []

    def save_game(self, g):
        self.games.append(g)

    def __len__(self):
        return len(self.games)


class _Writer:
    def add_scalar(self, *a, **k):
        pass


def _stub(n_shards=2, per_shard=5):
    """Minimal object exposing exactly what _inject_stockfish_games touches,
    with a fake _advance_injection_shard modelling a finite pool of
    n_shards*per_shard games that RE-SERVES whenever the index is reset to 0."""
    t = MuZeroTrainer.__new__(MuZeroTrainer)
    t.replay_buffer = _Buf()
    t._buffer_lock = threading.Lock()
    t.writer = _Writer()
    t.global_step = 1000
    t.game = None
    t._injection_shards = ["A", "B"][:n_shards]
    t._injection_shard_idx = 0
    t._injection_loaded = 0
    t._injection_shard_games = []

    def fake_advance(self):
        # Finite pool: shard `idx` holds `per_shard` games; exhausted at n_shards.
        # A cycle resets _injection_shard_idx to 0 -> serves shard 0 again.
        if self._injection_shard_idx >= n_shards:
            return False
        self._injection_shard_games = [
            f"s{self._injection_shard_idx}g{i}" for i in range(per_shard)
        ]
        self._injection_shard_idx += 1
        return True

    t._advance_injection_shard = types.MethodType(fake_advance, t)
    t._fastforward_injection = types.MethodType(lambda self: None, t)
    return t


def test_injection_cycles_past_exhaustion():
    pool = 2 * 5  # 10
    t = _stub(2, 5)
    # Ask for far more than the pool holds in one call: it must cycle, not stop.
    t._inject_stockfish_games(25)
    assert len(t.replay_buffer) == 25, "cycling did not keep the supply flowing"
    assert t._injection_shards, "pool was DISABLED (old behavior) instead of cycling"


def test_cycle_resets_cursor_not_monotonic():
    t = _stub(2, 5)  # pool = 10
    t._inject_stockfish_games(10)      # exactly one full pass
    assert t._injection_loaded == 10
    t._inject_stockfish_games(3)       # forces a cycle
    # After the cycle the cursor was reset to 0 then advanced by 3 — NOT 13 —
    # so the resume fast-forward (keyed on _injection_loaded) stays in-range.
    assert t._injection_loaded == 3, f"cursor not reset on cycle: {t._injection_loaded}"
    assert t._injection_shards
    assert len(t.replay_buffer) == 13


def test_genuinely_empty_pool_gives_up():
    # Zero-game pool (advance always False) must NOT infinite-loop; it disables.
    t = _stub(0, 0)

    def never(self):
        return False

    t._advance_injection_shard = types.MethodType(never, t)
    t._inject_stockfish_games(5)
    assert t._injection_shards == [], "empty pool should disable, not spin"
    assert len(t.replay_buffer) == 0
