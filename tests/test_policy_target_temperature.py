"""Regression tests for bug_hunt_2026_06_13.md §A + §B.

The dense policy TRAINING TARGET returned by ``select_action`` (CPU) and
``select_action_gpu`` (GPU-resident self-play path) must be the RAW, deduped
visit fraction ``N(a)/ΣN(a)``, independent of the sampling temperature:

  §B: temperature must drive ACTION SELECTION only, never the stored target.
  §A: the GPU path must SUM per-slot visits per unique action BEFORE applying
      temperature (option-(b) duplicates split an action's visits across slots);
      it previously did a per-slot softmax and only summed duplicates afterwards.

Construction: action 10 appears in 7 slots @4 visits each (28 total) and
action 20 in 1 slot @20 visits. Correct raw target: p(10)=28/48, p(20)=20/48.
The SELECTION at low T must concentrate on action 10 (higher TOTAL visits), not
the higher-PER-SLOT action 20.
"""

import numpy as np
import pytest
import torch

from src.mcts.mcts import MCTSNode, select_action
from src.mcts.tensor_mcts import select_action_gpu


ACTION_SPACE = 32
TEMPS = [1.0, 0.5, 0.25, 0.1]

# Deduped ground truth.
P10 = 28.0 / 48.0
P20 = 20.0 / 48.0


def _make_cpu_root():
    # select_action consumes a node with DEDUPED child_actions/child_visits
    # (the numpy _build_compat_roots path already collapses duplicate slots).
    node = MCTSNode(visit_count=48)
    node.child_actions = np.array([10, 20], dtype=np.int64)
    node.child_visits = np.array([28.0, 20.0], dtype=np.float64)
    return node


def _gpu_slot_tensors():
    # Raw per-slot layout BEFORE dedup: 7 slots of action 10 @4 each, 1 slot of
    # action 20 @20, plus padding slots (-1) to exercise the valid mask.
    actions = [10, 10, 10, 10, 10, 10, 10, 20, -1, -1]
    visits = [4, 4, 4, 4, 4, 4, 4, 20, 0, 0]
    child_actions = torch.tensor([actions], dtype=torch.int32)
    child_visits = torch.tensor([visits], dtype=torch.int32)
    return child_actions, child_visits


def test_cpu_target_is_raw_and_temperature_invariant():
    targets = []
    for t in TEMPS:
        np.random.seed(0)
        _, probs = select_action(_make_cpu_root(), temperature=t)
        assert probs[10] == pytest.approx(P10, abs=1e-6)
        assert probs[20] == pytest.approx(P20, abs=1e-6)
        targets.append(probs)
    # Identical across all temperatures.
    for probs in targets[1:]:
        assert np.allclose(probs, targets[0], atol=1e-7)
    # T=0 (greedy) target is still the raw distribution, not one-hot.
    _, probs0 = select_action(_make_cpu_root(), temperature=0.0)
    assert probs0[10] == pytest.approx(P10, abs=1e-6)
    assert probs0[20] == pytest.approx(P20, abs=1e-6)


def test_gpu_target_is_raw_deduped_and_temperature_invariant():
    child_actions, child_visits = _gpu_slot_tensors()
    targets = []
    for t in TEMPS:
        _, dense = select_action_gpu(
            child_actions, child_visits, torch.tensor(t), ACTION_SPACE
        )
        d = dense[0].cpu().numpy()
        assert d[10] == pytest.approx(P10, abs=1e-6)
        assert d[20] == pytest.approx(P20, abs=1e-6)
        # No mass leaks to other actions.
        assert d.sum() == pytest.approx(1.0, abs=1e-6)
        targets.append(d)
    for d in targets[1:]:
        assert np.allclose(d, targets[0], atol=1e-7)
    # Greedy (T=0) target is also raw, not one-hot.
    _, dense0 = select_action_gpu(
        child_actions, child_visits, torch.tensor(0.0), ACTION_SPACE
    )
    d0 = dense0[0].cpu().numpy()
    assert d0[10] == pytest.approx(P10, abs=1e-6)
    assert d0[20] == pytest.approx(P20, abs=1e-6)


def test_cpu_and_gpu_targets_agree():
    child_actions, child_visits = _gpu_slot_tensors()
    for t in TEMPS:
        np.random.seed(0)
        _, cpu_probs = select_action(_make_cpu_root(), temperature=t)
        _, gpu_dense = select_action_gpu(
            child_actions, child_visits, torch.tensor(t), ACTION_SPACE
        )
        g = gpu_dense[0].cpu().numpy()
        # Compare the two non-zero entries (different lengths otherwise).
        assert cpu_probs[10] == pytest.approx(g[10], abs=1e-6)
        assert cpu_probs[20] == pytest.approx(g[20], abs=1e-6)


def test_selection_respects_total_visits_at_low_temperature():
    # At low T the SELECTED action must concentrate on action 10 (28 total
    # visits) — the higher-TOTAL-visit action — NOT action 20 (higher per-slot).
    child_actions, child_visits = _gpu_slot_tensors()

    # GPU: sample many times at T=0.1, expect overwhelmingly action 10.
    torch.manual_seed(0)
    big_actions = child_actions.repeat(2000, 1)
    big_visits = child_visits.repeat(2000, 1)
    actions, _ = select_action_gpu(
        big_actions, big_visits, torch.tensor(0.1), ACTION_SPACE
    )
    frac_10 = (actions == 10).float().mean().item()
    assert frac_10 > 0.95, f"expected selection to concentrate on action 10, got {frac_10}"

    # GPU greedy must pick action 10.
    g_action, _ = select_action_gpu(
        child_actions, child_visits, torch.tensor(0.0), ACTION_SPACE
    )
    assert int(g_action[0]) == 10

    # CPU: T=0 greedy picks the higher-total action 10.
    a0, _ = select_action(_make_cpu_root(), temperature=0.0)
    assert a0 == 10
    # CPU: low-T sampling concentrates on action 10.
    picks = []
    for i in range(2000):
        np.random.seed(i)
        a, _ = select_action(_make_cpu_root(), temperature=0.1)
        picks.append(a)
    frac_cpu_10 = np.mean([p == 10 for p in picks])
    assert frac_cpu_10 > 0.95, f"CPU selection should favor action 10, got {frac_cpu_10}"
