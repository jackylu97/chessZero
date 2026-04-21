"""Reanalyze must not touch warmstart games (GameHistory with external_values)."""

import numpy as np
import torch

from src.training.replay_buffer import GameHistory, ReplayBuffer


def _mk_game(ext: list[float] | None) -> GameHistory:
    g = GameHistory(game_name="chess")
    g.observations.append(torch.zeros(3, 4, 4))
    g.observations.append(torch.zeros(3, 4, 4))
    g.actions.append(0)
    g.rewards.append(0.0)
    g.policies.append(np.zeros(8, dtype=np.float32))
    g.root_values.append(0.0)
    g.legal_actions_list.append([0])
    if ext is not None:
        g.external_values = list(ext)
    return g


def test_sample_for_reanalyze_excludes_warmstart():
    buf = ReplayBuffer(max_size=100)
    for _ in range(3):
        buf.save_game(_mk_game(ext=[0.1, 0.2]))  # warmstart
    for _ in range(5):
        buf.save_game(_mk_game(ext=None))  # self-play
    idx, games = buf.sample_games_for_reanalyze(10)
    assert len(games) == 5
    assert all(not g.external_values for g in games)


def test_sample_for_reanalyze_empty_when_all_warmstart():
    buf = ReplayBuffer(max_size=100)
    for _ in range(4):
        buf.save_game(_mk_game(ext=[0.1, 0.2]))
    idx, games = buf.sample_games_for_reanalyze(10)
    assert len(games) == 0


def test_sample_for_reanalyze_self_play_only_unchanged():
    buf = ReplayBuffer(max_size=100)
    for _ in range(5):
        buf.save_game(_mk_game(ext=None))
    idx, games = buf.sample_games_for_reanalyze(3)
    assert len(games) == 3
