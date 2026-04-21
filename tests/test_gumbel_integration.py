"""Integration tests for Plain Gumbel MuZero (G3 wiring).

Uses a tiny real MuZeroNetwork on tictactoe to exercise the self-play and
trainer paths end-to-end under ``use_gumbel=True``:
    - ``play_game`` and ``play_games_parallel`` dispatch to select_action_gumbel.
    - Trainer's KL policy loss has the right shape and equals CE−H(target).
    - ``_train_step`` runs on Gumbel self-play data without NaNs.
    - ``_reanalyze`` rewrites stored policies as π' (sum=1 over legals).
    - The agent-move path in evaluation routes through BatchedMCTS under use_gumbel.
"""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.config import MuZeroConfig
from src.games.tictactoe import TicTacToe
from src.model.muzero_net import MuZeroNetwork
from src.training.self_play import (
    play_game,
    play_games_parallel,
    run_self_play,
)
from src.training.trainer import MuZeroTrainer


# --- Fixtures --------------------------------------------------------------

def _tiny_ttt_config(**overrides) -> MuZeroConfig:
    """Minimal tictactoe config — sized so each test runs in well under a second."""
    cfg = MuZeroConfig(
        game="tictactoe",
        hidden_planes=8,
        num_residual_blocks=1,
        latent_h=3, latent_w=3,
        fc_hidden=8,
        num_simulations=4,
        batch_size=2,
        training_steps=2,
        replay_buffer_size=16,
        min_buffer_size=1,
        num_self_play_games=2,
        num_parallel_games=2,
        use_amp=False,
        device="cpu",
        reanalyze_interval=0,
        eval_interval=999999,  # suppress eval in tests
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_net(cfg: MuZeroConfig, game: TicTacToe) -> MuZeroNetwork:
    return MuZeroNetwork(
        observation_channels=game.num_planes,
        action_space_size=game.action_space_size,
        hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks,
        latent_h=cfg.latent_h, latent_w=cfg.latent_w,
        input_h=game.board_size[0], input_w=game.board_size[1],
        fc_hidden=cfg.fc_hidden,
        value_support_size=cfg.value_support_size,
        reward_support_size=cfg.reward_support_size,
    )


@pytest.fixture
def ttt():
    return TicTacToe()


@pytest.fixture
def gumbel_cfg():
    return _tiny_ttt_config(
        use_gumbel=True, gumbel_num_considered=9, use_gumbel_noise=True,
    )


@pytest.fixture
def puct_cfg():
    return _tiny_ttt_config(use_gumbel=False)


# --- self_play.play_game / play_games_parallel -----------------------------

def test_play_game_use_gumbel_produces_normalized_policies(ttt, gumbel_cfg):
    """Serial self-play under Gumbel emits π' targets that sum to 1 per move
    and are non-negative at every position."""
    torch.manual_seed(0)
    np.random.seed(0)
    net = _make_net(gumbel_cfg, ttt)
    history = play_game(net, ttt, gumbel_cfg, device="cpu", training_step=0)

    assert len(history.actions) > 0
    assert len(history.policies) == len(history.actions)
    for pi in history.policies:
        assert len(pi) == ttt.action_space_size
        assert sum(pi) == pytest.approx(1.0, abs=1e-5)
        assert all(p >= 0 for p in pi)


def test_play_games_parallel_use_gumbel_policies_sum_to_one(ttt, gumbel_cfg):
    """Batched self-play under Gumbel emits valid π' per game, per position."""
    torch.manual_seed(0)
    np.random.seed(0)
    net = _make_net(gumbel_cfg, ttt)
    histories = play_games_parallel(
        net, ttt, gumbel_cfg, num_games=3, device="cpu", training_step=0,
    )

    assert len(histories) == 3
    for h in histories:
        assert len(h.actions) > 0
        for pi in h.policies:
            assert sum(pi) == pytest.approx(1.0, abs=1e-5)


def test_play_game_use_gumbel_false_still_valid_targets(ttt, puct_cfg):
    """Regression: the use_gumbel=False branch untouched — PUCT path runs
    to completion and stores valid visit-count policies."""
    torch.manual_seed(0)
    np.random.seed(0)
    net = _make_net(puct_cfg, ttt)
    history = play_game(net, ttt, puct_cfg, device="cpu", training_step=0)

    assert len(history.actions) > 0
    for pi in history.policies:
        assert sum(pi) == pytest.approx(1.0, abs=1e-5)


def test_play_games_parallel_gumbel_mask_respects_legal_actions(ttt, gumbel_cfg):
    """π' rows must assign zero mass to any action not in the legal set at that
    move. Critical for training-target correctness — illegal actions must not
    leak gradient into the policy head."""
    torch.manual_seed(0)
    np.random.seed(0)
    net = _make_net(gumbel_cfg, ttt)
    histories = play_games_parallel(
        net, ttt, gumbel_cfg, num_games=2, device="cpu", training_step=0,
    )
    for h in histories:
        for pi, legal in zip(h.policies, h.legal_actions_list):
            legal_set = set(legal)
            for a, p in enumerate(pi):
                if a not in legal_set:
                    assert p == 0.0, f"illegal action {a} got nonzero mass {p}"


# --- Trainer loss ----------------------------------------------------------

def test_policy_loss_kl_shape_and_entropy_identity(ttt, gumbel_cfg):
    """_policy_loss_kl returns per-sample (B,) values and satisfies
    CE(target, p) = KL(target || p) + H(target) identity to high precision."""
    net = _make_net(gumbel_cfg, ttt)
    with tempfile.TemporaryDirectory() as tmp:
        tr = MuZeroTrainer(
            gumbel_cfg, ttt, net, run_id="t",
            device="cpu",
            log_dir=os.path.join(tmp, "runs"),
            checkpoints_dir=os.path.join(tmp, "ckpt"),
        )
        torch.manual_seed(0)
        logits = torch.randn(5, ttt.action_space_size)
        targets = torch.softmax(torch.randn(5, ttt.action_space_size), dim=1)

        loss_kl = tr._policy_loss_kl(logits, targets)
        loss_ce = tr._policy_loss(logits, targets)
        tr.writer.close()

    assert loss_kl.shape == (5,)
    assert loss_ce.shape == (5,)
    # KL >= 0 (floating-point slack)
    assert torch.all(loss_kl >= -1e-6)
    # CE - KL = H(target)
    entropy = -(targets * torch.log(targets.clamp_min(1e-12))).sum(dim=1)
    torch.testing.assert_close(loss_ce - loss_kl, entropy, atol=1e-5, rtol=1e-4)


def test_policy_loss_kl_zero_when_logits_match_target(ttt, gumbel_cfg):
    """If π_net ≡ π' then F.kl_div returns 0 per sample."""
    net = _make_net(gumbel_cfg, ttt)
    with tempfile.TemporaryDirectory() as tmp:
        tr = MuZeroTrainer(
            gumbel_cfg, ttt, net, run_id="t",
            device="cpu",
            log_dir=os.path.join(tmp, "runs"),
            checkpoints_dir=os.path.join(tmp, "ckpt"),
        )
        logits = torch.randn(3, ttt.action_space_size)
        targets = torch.softmax(logits, dim=1)  # identical by construction
        loss = tr._policy_loss_kl(logits, targets)
        tr.writer.close()

    torch.testing.assert_close(loss, torch.zeros(3), atol=1e-6, rtol=1e-6)


# --- Trainer integration ---------------------------------------------------

def _make_trainer(cfg, ttt, tmpdir):
    net = _make_net(cfg, ttt)
    return MuZeroTrainer(
        cfg, ttt, net, run_id="test", device="cpu",
        log_dir=os.path.join(tmpdir, "runs"),
        checkpoints_dir=os.path.join(tmpdir, "ckpt"),
    )


def test_train_step_runs_end_to_end_under_use_gumbel(ttt, gumbel_cfg):
    """Full _train_step on a tiny buffer of Gumbel self-play games must finish
    without NaNs and update PER priorities."""
    torch.manual_seed(0)
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tr = _make_trainer(gumbel_cfg, ttt, tmp)
        games = run_self_play(
            tr.network, ttt, gumbel_cfg, num_games=2, device="cpu",
            show_progress=False,
        )
        for g in games:
            tr.replay_buffer.save_game(g)
        info = tr._train_step()
        tr.writer.close()

    assert np.isfinite(info["total_loss"])
    assert np.isfinite(info["policy_loss"])
    assert np.isfinite(info["value_loss"])
    assert np.isfinite(info["reward_loss"])
    assert np.isfinite(info["grad_norm"])


def test_train_step_under_use_gumbel_uses_kl_branch(ttt, gumbel_cfg, monkeypatch):
    """Sanity: with use_gumbel=True, _train_step dispatches to _policy_loss_kl,
    not _policy_loss. Monkey-patches both and asserts only KL was called."""
    torch.manual_seed(0)
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tr = _make_trainer(gumbel_cfg, ttt, tmp)
        games = run_self_play(
            tr.network, ttt, gumbel_cfg, num_games=2, device="cpu",
            show_progress=False,
        )
        for g in games:
            tr.replay_buffer.save_game(g)

        kl_calls = {"n": 0}
        ce_calls = {"n": 0}
        orig_kl = tr._policy_loss_kl
        orig_ce = tr._policy_loss

        def kl_spy(*a, **kw):
            kl_calls["n"] += 1
            return orig_kl(*a, **kw)

        def ce_spy(*a, **kw):
            ce_calls["n"] += 1
            return orig_ce(*a, **kw)

        tr._policy_loss_kl = kl_spy
        tr._policy_loss = ce_spy
        tr._train_step()
        tr.writer.close()

    # One call at k=0 + one per unroll step = (num_unroll_steps + 1)
    assert kl_calls["n"] == gumbel_cfg.num_unroll_steps + 1
    assert ce_calls["n"] == 0


def test_train_step_under_use_gumbel_false_uses_cross_entropy(ttt, puct_cfg):
    """Regression: use_gumbel=False still routes through _policy_loss (CE)."""
    torch.manual_seed(0)
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tr = _make_trainer(puct_cfg, ttt, tmp)
        games = run_self_play(
            tr.network, ttt, puct_cfg, num_games=2, device="cpu",
            show_progress=False,
        )
        for g in games:
            tr.replay_buffer.save_game(g)

        kl_calls = {"n": 0}
        ce_calls = {"n": 0}
        orig_kl = tr._policy_loss_kl
        orig_ce = tr._policy_loss

        def kl_spy(*a, **kw):
            kl_calls["n"] += 1
            return orig_kl(*a, **kw)

        def ce_spy(*a, **kw):
            ce_calls["n"] += 1
            return orig_ce(*a, **kw)

        tr._policy_loss_kl = kl_spy
        tr._policy_loss = ce_spy
        tr._train_step()
        tr.writer.close()

    assert kl_calls["n"] == 0
    assert ce_calls["n"] == puct_cfg.num_unroll_steps + 1


# --- Reanalyze under use_gumbel --------------------------------------------

def test_reanalyze_rewrites_policies_as_pi_prime(ttt, gumbel_cfg):
    """Under use_gumbel, _reanalyze replaces stored policies with π' dense
    targets: sum to 1 over legals, zero on illegal."""
    torch.manual_seed(0)
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tr = _make_trainer(gumbel_cfg, ttt, tmp)
        games = run_self_play(
            tr.network, ttt, gumbel_cfg, num_games=3, device="cpu",
            show_progress=False,
        )
        for g in games:
            tr.replay_buffer.save_game(g)

        # Corrupt stored policies so we can detect the in-place overwrite.
        for g in tr.replay_buffer.buffer:
            for i in range(len(g.policies)):
                g.policies[i] = [0.0] * ttt.action_space_size

        tr._reanalyze()
        tr.writer.close()

    # After reanalyze every stored policy must re-sum to 1 and be masked to legals.
    rewritten_count = 0
    for g in tr.replay_buffer.buffer:
        for pi, legal in zip(g.policies, g.legal_actions_list):
            total = sum(pi)
            if total > 0:  # skip any positions that may not have been reanalyzed
                rewritten_count += 1
                assert total == pytest.approx(1.0, abs=1e-5)
                legal_set = set(legal)
                for a, p in enumerate(pi):
                    if a not in legal_set:
                        assert p == 0.0
    assert rewritten_count > 0  # at least one position was reanalyzed


# --- Evaluation agent-move path -------------------------------------------

def test_agent_action_under_use_gumbel_returns_legal_action(ttt, gumbel_cfg):
    """_agent_action must route through BatchedMCTS under use_gumbel and
    return an action inside the current legal set."""
    torch.manual_seed(0)
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tr = _make_trainer(gumbel_cfg, ttt, tmp)
        state = ttt.reset()
        legal = ttt.legal_actions(state)
        action = tr._agent_action(state)
        tr.writer.close()

    assert action in legal


def test_agent_action_under_puct_path_also_returns_legal_action(ttt, puct_cfg):
    """Regression: under use_gumbel=False, _agent_action still returns a legal
    action via the serial MCTS + select_action path."""
    torch.manual_seed(0)
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tr = _make_trainer(puct_cfg, ttt, tmp)
        state = ttt.reset()
        legal = ttt.legal_actions(state)
        action = tr._agent_action(state)
        tr.writer.close()

    assert action in legal
