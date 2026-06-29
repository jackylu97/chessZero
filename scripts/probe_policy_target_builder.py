"""Probe: does select_action_gpu build the SAME deduped, temperature-independent
policy target as the numpy oracle select_action? Includes the duplicate-slot
(option-b) path that the numpy MCTS trace did not exercise on GPU."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.mcts.tensor_mcts import select_action_gpu
from src.mcts.mcts import select_action, MCTSNode


def numpy_target(child_actions, child_visits):
    """Replicate the deduped raw-visit-fraction target the way the numpy oracle
    would AFTER deduping (it dedups at expand time; here we emulate by summing
    duplicate slots, exactly like the GPU scatter_add)."""
    A = int(child_actions[child_actions >= 0].max()) + 1 if (child_actions >= 0).any() else 1
    dense = np.zeros(A, dtype=np.float64)
    for a, v in zip(child_actions, child_visits):
        if a >= 0:
            dense[a] += v
    tot = dense.sum()
    return dense / tot if tot > 0 else dense


def run_case(name, child_actions_np, child_visits_np, A, temps=(0.0, 0.1, 0.5, 1.0, 2.0)):
    ca = torch.tensor(child_actions_np[None], dtype=torch.int32)
    cv = torch.tensor(child_visits_np[None], dtype=torch.int32)
    ref = numpy_target(child_actions_np, child_visits_np)
    ref_full = np.zeros(A); ref_full[:len(ref)] = ref
    max_diff_across_temp = 0.0
    targets = []
    for T in temps:
        t = torch.tensor(float(T))
        action, dense = select_action_gpu(ca, cv, t, A)
        dense_np = dense[0].cpu().numpy().astype(np.float64)
        targets.append(dense_np)
        d = np.abs(dense_np - ref_full).max()
        max_diff_across_temp = max(max_diff_across_temp, d)
    # temperature-independence: all targets identical
    temp_indep = max(np.abs(targets[i] - targets[0]).max() for i in range(len(targets)))
    print(f"  {name}: max|gpu-oracle|={max_diff_across_temp:.2e}  "
          f"temp_independence(max spread over T)={temp_indep:.2e}")
    return max_diff_across_temp, temp_indep


def main():
    print("select_action_gpu vs numpy oracle (deduped raw visit fraction):")
    # Case A: unique actions, no duplicates.
    run_case("unique", np.array([3, 7, 1, 9, 2]), np.array([10, 5, 20, 1, 4]), A=12)
    # Case B: duplicate slots for the same action (option-b path).
    run_case("dup-slots", np.array([3, 3, 7, 7, 7, 1]), np.array([4, 6, 1, 2, 3, 30]), A=12)
    # Case C: padding (-1) slots present.
    run_case("with-pad", np.array([5, 2, -1, -1, 8]), np.array([12, 8, 0, 0, 3]), A=12)
    # Case D: single dominant action.
    run_case("dominant", np.array([4, 1, 9]), np.array([200, 0, 0]), A=12)
    # Case E: all zero visits -> all-zero target (fallback).
    run_case("zero-visits", np.array([4, 1, 9]), np.array([0, 0, 0]), A=12)

    # Random stress: 200 random K=50 visit vectors, compare to oracle at T=1 and
    # confirm temperature-independence.
    rng = np.random.default_rng(0)
    worst_d = 0.0; worst_ti = 0.0
    A = 4672
    for _ in range(200):
        K = 50
        acts = rng.integers(0, A, size=K).astype(np.int64)
        # inject duplicates
        acts[K//2:] = acts[:K - K//2]
        vis = rng.integers(0, 30, size=K).astype(np.int64)
        d, ti = run_case_silent(acts, vis, A)
        worst_d = max(worst_d, d); worst_ti = max(worst_ti, ti)
    print(f"\nRandom stress (200 x K=50, with dup injection):")
    print(f"  worst max|gpu-oracle| = {worst_d:.2e}")
    print(f"  worst temperature-independence spread = {worst_ti:.2e}")


def run_case_silent(child_actions_np, child_visits_np, A, temps=(0.0, 0.1, 1.0, 2.0)):
    ca = torch.tensor(child_actions_np[None], dtype=torch.int32)
    cv = torch.tensor(child_visits_np[None], dtype=torch.int32)
    ref = numpy_target(child_actions_np, child_visits_np)
    ref_full = np.zeros(A); ref_full[:len(ref)] = ref
    targets = []
    md = 0.0
    for T in temps:
        action, dense = select_action_gpu(ca, cv, torch.tensor(float(T)), A)
        dense_np = dense[0].cpu().numpy().astype(np.float64)
        targets.append(dense_np)
        md = max(md, np.abs(dense_np - ref_full).max())
    ti = max(np.abs(targets[i] - targets[0]).max() for i in range(len(targets)))
    return md, ti


if __name__ == "__main__":
    main()
