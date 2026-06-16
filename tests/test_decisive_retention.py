"""Retention-weighted eviction (config.decisive_retention_multiplier).

A decisive game's age counts M× slower for eviction, so it persists ~M× longer
than a drawn game — lifting the buffer's decisive-game density. M=1 reproduces
plain oldest-first FIFO.
"""
from __future__ import annotations

from src.training.replay_buffer import ReplayBuffer, GameHistory


def _game(outcome=0.0, warm=False):
    g = GameHistory(game_name="chess")
    g.game_outcome = float(outcome)
    if warm:
        g.external_values = [0.1, 0.2]  # non-empty → counts as a warmstart game
    return g


def test_m1_is_plain_fifo():
    """multiplier=1.0 → oldest-first eviction, regardless of outcome."""
    buf = ReplayBuffer(max_size=3, decisive_retention_multiplier=1.0)
    games = [_game(outcome=1.0) for _ in range(5)]  # all decisive, but M=1 ⇒ no weighting
    for g in games:
        buf.save_game(g)
    # First two evicted FIFO; last three remain in order.
    assert [g is x for g, x in zip(buf.buffer, games[2:])] == [True, True, True]
    assert not any(g is games[0] or g is games[1] for g in buf.buffer)


def test_decisive_survives_draw_flood():
    """With M>1, a decisive game survives a flood of draws that would evict it
    several times over under FIFO."""
    buf = ReplayBuffer(max_size=3, decisive_retention_multiplier=10.0)
    X = _game(outcome=1.0)
    buf.save_game(X)
    for _ in range(8):                       # FIFO would drop X within 3 saves
        buf.save_game(_game(outcome=0.0))
    assert any(g is X for g in buf.buffer), "decisive game evicted despite high retention"
    # Draws are the ones being cycled out: buffer holds X + the two newest draws.
    assert sum(1 for g in buf.buffer if g.game_outcome == 0.0) == 2


def test_decisive_eventually_evicted_after_about_M():
    """Retention is BOUNDED: a decisive game IS eventually evicted once it is
    ~M× older than a draw (it doesn't live forever)."""
    M = 5
    buf = ReplayBuffer(max_size=2, decisive_retention_multiplier=float(M))
    X = _game(outcome=1.0)
    buf.save_game(X)
    # SP pool size 2; each new draw evicts the older draw until X is ~M× aged out.
    evicted_at = None
    for k in range(1, 40):
        buf.save_game(_game(outcome=0.0))
        if not any(g is X for g in buf.buffer):
            evicted_at = k
            break
    assert evicted_at is not None, "decisive game never evicted (unbounded retention)"
    # A draw in a size-2 pool lasts ~2 saves; X should last on the order of M× that.
    assert evicted_at >= M, f"decisive evicted too early ({evicted_at} < M={M})"


def test_two_pool_retention_isolated_from_warmstart():
    """Decisive retention applies within the self-play pool; the warmstart pool is
    a separate FIFO pool and is untouched by self-play draw traffic."""
    buf = ReplayBuffer(max_size=4, warmstart_max_size=2, decisive_retention_multiplier=10.0)
    w1 = _game(outcome=1.0, warm=True)
    buf.save_game(w1)
    X = _game(outcome=1.0)                   # decisive self-play game
    buf.save_game(X)
    for _ in range(6):                       # self-play draw flood (SP pool cap = 4-2 = 2)
        buf.save_game(_game(outcome=0.0))
    assert any(g is X for g in buf.buffer), "decisive self-play game evicted"
    assert any(g is w1 for g in buf.buffer), "warmstart game wrongly displaced by self-play"


def test_save_counter_advances_and_back_compat_default():
    """Default construction (no multiplier arg) is M=1 FIFO, and _save_counter
    advances per save (so birth stamps are monotonic)."""
    buf = ReplayBuffer(max_size=2)          # default multiplier
    assert buf.decisive_retention_multiplier == 1.0
    a, b, c = _game(1.0), _game(1.0), _game(1.0)
    for g in (a, b, c):
        buf.save_game(g)
    assert buf._save_counter == 3
    assert [g.birth for g in (a, b, c)] == [0, 1, 2]
    assert not any(g is a for g in buf.buffer)  # oldest evicted (FIFO)
