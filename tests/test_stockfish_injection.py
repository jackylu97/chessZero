"""Tests for Stockfish injection: pool consumption, fast-forward on resume,
pool-alive signal, and cursor persistence across checkpoints.

Run with: pytest tests/test_stockfish_injection.py
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import MuZeroConfig
from src.games.tictactoe import TicTacToe
from src.model.muzero_net import MuZeroNetwork
from src.training.replay_buffer import GameHistory
from src.training.trainer import MuZeroTrainer


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _mk_game(ext_val: float = 0.5, n_moves: int = 3) -> GameHistory:
    """Minimal tictactoe-shaped GameHistory with external_values populated."""
    g = GameHistory(game_name="tictactoe")
    for i in range(n_moves):
        g.observations.append(torch.zeros(1, 3, 3))
        g.actions.append(i)
        g.rewards.append(0.0)
        g.policies.append(np.zeros(9, dtype=np.float32))
        g.root_values.append(0.0)
        g.legal_actions_list.append(list(range(9)))
    g.observations.append(torch.zeros(1, 3, 3))
    g.game_outcome = 0.0
    g.external_values = [ext_val] * (n_moves + 1)
    return g


def _write_shard(path: Path, n_games: int) -> None:
    games = [_mk_game() for _ in range(n_games)]
    with open(path, "wb") as f:
        pickle.dump(games, f)


def _make_trainer(tmp_path: Path, *, injection_games: int = 2,
                  injection_interval: int = 10, buffer_size: int = 20,
                  min_buffer_size: int = 5) -> MuZeroTrainer:
    """Tiny tictactoe trainer sufficient for exercising injection methods.

    Network is real (required by the constructor) but stays unused — no
    training steps run in these tests, so the CPU+small-net cost is trivial.
    """
    config = MuZeroConfig(
        game="tictactoe",
        hidden_planes=8, num_residual_blocks=1,
        latent_h=3, latent_w=3, fc_hidden=8,
        training_steps=10,
        batch_size=2,
        stockfish_injection_games=injection_games,
        stockfish_injection_interval=injection_interval,
        replay_buffer_size=buffer_size,
        min_buffer_size=min_buffer_size,
        self_play_interval=10,
        reanalyze_interval=0,  # skip reanalyze path for tests
    )
    game = TicTacToe()
    network = MuZeroNetwork(
        observation_channels=game.num_planes,
        action_space_size=game.action_space_size,
        hidden_planes=config.hidden_planes,
        num_blocks=config.num_residual_blocks,
        latent_h=config.latent_h, latent_w=config.latent_w,
        input_h=3, input_w=3,
        fc_hidden=config.fc_hidden,
        value_support_size=config.value_support_size,
        reward_support_size=config.reward_support_size,
    )
    return MuZeroTrainer(
        config, game, network, run_id="injection_test",
        device="cpu",
        log_dir=str(tmp_path / "logs"),
        checkpoints_dir=str(tmp_path / "ckpts"),
    )


@pytest.fixture
def trainer(tmp_path):
    return _make_trainer(tmp_path)


# ---------------------------------------------------------------------------
# set_injection_shards + basic consumption
# ---------------------------------------------------------------------------


def test_set_injection_shards_stores_path_objects(trainer, tmp_path):
    paths = [str(tmp_path / "a.pkl"), str(tmp_path / "b.pkl")]
    trainer.set_injection_shards(paths)
    assert len(trainer._injection_shards) == 2
    assert all(isinstance(p, Path) for p in trainer._injection_shards)


def test_inject_within_single_shard(trainer, tmp_path):
    _write_shard(tmp_path / "s0.pkl", 5)
    trainer.set_injection_shards([str(tmp_path / "s0.pkl")])

    trainer._inject_stockfish_games(3)

    assert len(trainer.replay_buffer) == 3
    assert trainer._injection_loaded == 3
    assert len(trainer._injection_shard_games) == 2  # tail cached
    assert trainer._injection_shards  # pool still alive


def test_inject_crosses_shard_boundary(trainer, tmp_path):
    _write_shard(tmp_path / "s0.pkl", 3)
    _write_shard(tmp_path / "s1.pkl", 3)
    trainer.set_injection_shards([
        str(tmp_path / "s0.pkl"),
        str(tmp_path / "s1.pkl"),
    ])

    trainer._inject_stockfish_games(5)

    assert len(trainer.replay_buffer) == 5
    assert trainer._injection_loaded == 5
    assert trainer._injection_shard_idx == 2  # advanced into shard 1
    assert len(trainer._injection_shard_games) == 1  # 1 game cached in shard 1


def test_inject_exhausts_pool_clears_shards(trainer, tmp_path):
    _write_shard(tmp_path / "s0.pkl", 3)
    _write_shard(tmp_path / "s1.pkl", 2)
    trainer.set_injection_shards([
        str(tmp_path / "s0.pkl"),
        str(tmp_path / "s1.pkl"),
    ])

    trainer._inject_stockfish_games(100)  # overshoot

    assert len(trainer.replay_buffer) == 5
    assert trainer._injection_loaded == 5
    assert trainer._injection_shards == []  # pool marked dead


def test_inject_is_noop_without_shards(trainer, tmp_path):
    trainer._inject_stockfish_games(5)
    assert len(trainer.replay_buffer) == 0
    assert trainer._injection_loaded == 0


# ---------------------------------------------------------------------------
# Pool-alive signal (drives self-play / reanalyze gating in train loop)
# ---------------------------------------------------------------------------


def test_pool_alive_true_when_shards_attached(trainer, tmp_path):
    _write_shard(tmp_path / "s0.pkl", 3)
    trainer.set_injection_shards([str(tmp_path / "s0.pkl")])
    assert bool(trainer._injection_shards)


def test_pool_alive_false_after_exhaustion(trainer, tmp_path):
    _write_shard(tmp_path / "s0.pkl", 2)
    trainer.set_injection_shards([str(tmp_path / "s0.pkl")])
    trainer._inject_stockfish_games(10)
    assert not bool(trainer._injection_shards)


# ---------------------------------------------------------------------------
# Fast-forward on resume
# ---------------------------------------------------------------------------


def test_fastforward_full_shard_skip(trainer, tmp_path):
    """Cursor aligned to a shard boundary: entire skipped shards are loaded+dropped."""
    _write_shard(tmp_path / "s0.pkl", 3)
    _write_shard(tmp_path / "s1.pkl", 3)
    trainer.set_injection_shards([
        str(tmp_path / "s0.pkl"),
        str(tmp_path / "s1.pkl"),
    ])
    trainer._injection_loaded = 3  # s0 already consumed

    trainer._inject_stockfish_games(2)

    assert len(trainer.replay_buffer) == 2
    assert trainer._injection_loaded == 5
    assert trainer._injection_shard_idx == 2  # advanced to end of shard list


def test_fastforward_mid_shard(trainer, tmp_path):
    """Cursor mid-shard: only the consumed prefix of that shard is dropped."""
    _write_shard(tmp_path / "s0.pkl", 5)
    trainer.set_injection_shards([str(tmp_path / "s0.pkl")])
    trainer._injection_loaded = 2  # 2 games consumed in prior run

    trainer._inject_stockfish_games(1)

    assert len(trainer.replay_buffer) == 1
    assert trainer._injection_loaded == 3
    # 2 games still cached (positions 3 and 4 of the shard).
    assert len(trainer._injection_shard_games) == 2


def test_fastforward_cursor_past_pool_clears(trainer, tmp_path):
    """If the resume cursor exceeds the available pool, shards are cleared cleanly."""
    _write_shard(tmp_path / "s0.pkl", 3)
    trainer.set_injection_shards([str(tmp_path / "s0.pkl")])
    trainer._injection_loaded = 10  # larger than pool size

    trainer._inject_stockfish_games(5)

    assert len(trainer.replay_buffer) == 0  # nothing injected
    assert trainer._injection_shards == []  # pool dead


def test_fastforward_runs_only_once(trainer, tmp_path):
    """After the first injection post-resume, subsequent calls must not re-skip."""
    _write_shard(tmp_path / "s0.pkl", 5)
    trainer.set_injection_shards([str(tmp_path / "s0.pkl")])
    trainer._injection_loaded = 1

    trainer._inject_stockfish_games(1)
    assert trainer._injection_loaded == 2
    assert len(trainer.replay_buffer) == 1

    trainer._inject_stockfish_games(2)
    # Second call should add 2 more without re-skipping.
    assert trainer._injection_loaded == 4
    assert len(trainer.replay_buffer) == 3


# ---------------------------------------------------------------------------
# Integration with ReplayBuffer: FIFO, max-priority seed, reanalyze skip
# ---------------------------------------------------------------------------


def test_injected_games_get_max_priority(trainer, tmp_path):
    """save_game seeds new entries at current-max priority — fresh games get sampled."""
    _write_shard(tmp_path / "s0.pkl", 3)
    trainer.set_injection_shards([str(tmp_path / "s0.pkl")])

    trainer._inject_stockfish_games(3)

    priorities = trainer.replay_buffer._priorities
    assert len(priorities) == 3
    # All injected at max priority (1.0 on an empty buffer).
    assert all(p == max(priorities) for p in priorities)


def test_injected_games_fifo_evict_when_buffer_full(tmp_path):
    """buffer_size=3, inject 5 games → oldest 2 get FIFO-evicted."""
    trainer = _make_trainer(tmp_path, buffer_size=3)
    _write_shard(tmp_path / "s0.pkl", 5)
    trainer.set_injection_shards([str(tmp_path / "s0.pkl")])

    trainer._inject_stockfish_games(5)

    assert len(trainer.replay_buffer) == 3
    assert trainer._injection_loaded == 5
    assert trainer.replay_buffer.total_games == 5  # total counter not capped


def test_injected_games_excluded_from_reanalyze_sampling(trainer, tmp_path):
    """External-values games must not be reanalyzed — integration check."""
    _write_shard(tmp_path / "s0.pkl", 4)
    trainer.set_injection_shards([str(tmp_path / "s0.pkl")])

    trainer._inject_stockfish_games(4)

    idx, games = trainer.replay_buffer.sample_games_for_reanalyze(4)
    assert len(games) == 0  # all injected games have external_values → filtered


# ---------------------------------------------------------------------------
# Checkpoint persistence of the cursor
# ---------------------------------------------------------------------------


def test_cursor_written_to_checkpoint(trainer, tmp_path):
    _write_shard(tmp_path / "s0.pkl", 5)
    trainer.set_injection_shards([str(tmp_path / "s0.pkl")])
    trainer._inject_stockfish_games(3)
    assert trainer._injection_loaded == 3

    trainer._save_checkpoint(100)
    ckpt_path = trainer.checkpoint_dir / "checkpoint_100.pt"
    assert ckpt_path.exists()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert ckpt["injection_loaded"] == 3


def test_cursor_restored_by_load_checkpoint(trainer, tmp_path):
    _write_shard(tmp_path / "s0.pkl", 5)
    trainer.set_injection_shards([str(tmp_path / "s0.pkl")])
    trainer._injection_loaded = 7
    trainer._save_checkpoint(200)
    ckpt_path = trainer.checkpoint_dir / "checkpoint_200.pt"

    # Zero out in-memory state, then load should restore it.
    trainer._injection_loaded = 0
    trainer.load_checkpoint(str(ckpt_path))

    assert trainer._injection_loaded == 7


def test_checkpoint_without_cursor_field_defaults_to_zero(trainer, tmp_path):
    """Loading a pre-injection-era checkpoint must not crash; cursor = 0."""
    _write_shard(tmp_path / "s0.pkl", 2)
    trainer.set_injection_shards([str(tmp_path / "s0.pkl")])

    # Simulate an older checkpoint by saving with torch.save directly (no
    # ``injection_loaded`` key). Mirrors the pre-refactor save format.
    legacy_ckpt = {
        "step": 100,
        "model_state_dict": trainer.network.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
    }
    legacy_path = trainer.checkpoint_dir / "checkpoint_legacy.pt"
    torch.save(legacy_ckpt, legacy_path)

    trainer._injection_loaded = 42  # non-zero; should be clobbered to 0 on load
    trainer.load_checkpoint(str(legacy_path))

    assert trainer._injection_loaded == 0


# ---------------------------------------------------------------------------
# Cold-start bootstrap (run train() with training_steps=0 to exercise the
# bootstrap block without the main loop).
# ---------------------------------------------------------------------------


def test_bootstrap_injects_from_pool_when_attached(tmp_path):
    """Empty buffer + pool attached → bootstrap injects max(injection_games, min_buffer_size)."""
    trainer = _make_trainer(
        tmp_path, injection_games=2, min_buffer_size=5, buffer_size=20,
    )
    _write_shard(tmp_path / "s0.pkl", 10)
    trainer.set_injection_shards([str(tmp_path / "s0.pkl")])
    trainer.config.training_steps = 0  # skip main loop

    trainer.train()

    # max(2, 5) = 5 → bootstrap injects 5 games.
    assert len(trainer.replay_buffer) == 5
    assert trainer._injection_loaded == 5
    assert trainer._injection_shards  # pool still alive post-bootstrap


def test_bootstrap_skipped_when_buffer_not_empty(tmp_path):
    """Resume path: buffer already populated → bootstrap should not fire."""
    trainer = _make_trainer(tmp_path, injection_games=2, min_buffer_size=5)
    _write_shard(tmp_path / "s0.pkl", 10)
    trainer.set_injection_shards([str(tmp_path / "s0.pkl")])

    # Pre-populate the buffer as if a .buf was loaded.
    existing = _mk_game(ext_val=0.1)
    trainer.replay_buffer.save_game(existing)

    trainer.config.training_steps = 0
    trainer.train()

    # Buffer still has only the pre-existing game; bootstrap did not fire.
    assert len(trainer.replay_buffer) == 1
    assert trainer._injection_loaded == 0


# ---------------------------------------------------------------------------
# Phase-dependent value_loss_weight (warmstart vs selfplay)
# ---------------------------------------------------------------------------


def _trainer_with_phase_weights(tmp_path: Path, *, warm: float, selfp: float):
    t = _make_trainer(tmp_path)
    t.config.value_loss_weight_warmstart = warm
    t.config.value_loss_weight_selfplay = selfp
    return t


def test_value_loss_weight_uses_warmstart_while_pool_alive(tmp_path):
    """With phase fields set and the injection pool still alive, value weight
    reads the warmstart value."""
    t = _trainer_with_phase_weights(tmp_path, warm=1.0, selfp=0.25)
    t.set_injection_shards([str(tmp_path / "stub.pkl")])  # not read; just truthy
    assert t._current_value_loss_weight() == pytest.approx(1.0)


def test_value_loss_weight_switches_at_pool_exhaustion(tmp_path):
    """Once _injection_shards goes empty (the existing pool_alive signal), the
    weight flips to the self-play value."""
    t = _trainer_with_phase_weights(tmp_path, warm=1.0, selfp=0.25)
    t.set_injection_shards([str(tmp_path / "stub.pkl")])
    assert t._current_value_loss_weight() == pytest.approx(1.0)

    t._injection_shards = []  # simulate exhaustion (mirrors _inject_stockfish_games)
    assert t._current_value_loss_weight() == pytest.approx(0.25)


def test_value_loss_weight_falls_back_when_phase_fields_unset(tmp_path):
    """Back-compat: if a preset leaves both phase fields at None, the trainer
    uses the scalar value_loss_weight."""
    t = _make_trainer(tmp_path)
    t.config.value_loss_weight_warmstart = None
    t.config.value_loss_weight_selfplay = None
    t.config.value_loss_weight = 0.42  # sentinel
    t.set_injection_shards([str(tmp_path / "stub.pkl")])
    assert t._current_value_loss_weight() == pytest.approx(0.42)
    t._injection_shards = []
    assert t._current_value_loss_weight() == pytest.approx(0.42)


def test_chess_preset_has_phase_value_weights():
    """Sanity: next-run chess preset ships with 1.0 / 0.25 split."""
    from src.config import get_config
    cfg = get_config("chess")
    assert cfg.value_loss_weight_warmstart == pytest.approx(1.0)
    assert cfg.value_loss_weight_selfplay == pytest.approx(0.25)
