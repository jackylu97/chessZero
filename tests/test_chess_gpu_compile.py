"""Correctness equivalence: torch.compile of GpuChessGame ops must match eager.

The bench script (`scripts/bench_chess_compare.py`) wraps each op in
`torch.compile` to claim the speed-ups. If the compiled graph diverges from
eager (graph break to a different code path, dynamic-shape bug, etc.), all
of the perf numbers are meaningless. This test pins the contract.

Run on CUDA only — compile-on-CPU is heavy with TorchInductor and we don't
ship a CPU-compiled path in self-play. Skipped when no CUDA.
"""
import random

import chess
import pytest
import torch

from src.games.chess_gpu import GpuChessGame


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _random_positions(n: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    out: list[chess.Board] = []
    while len(out) < n:
        b = chess.Board()
        n_plies = rng.randint(0, 80)
        for _ in range(n_plies):
            if b.is_game_over(claim_draw=False):
                break
            b.push(rng.choice(list(b.legal_moves)))
        if not b.is_game_over(claim_draw=False) and not b.is_repetition(3):
            out.append(b.copy())
    return out


def test_to_tensor_compile_matches_eager():
    boards = _random_positions(200, seed=21)
    gg = GpuChessGame()
    state = gg.from_python_chess(boards, device="cuda")

    eager = gg.to_tensor_batch(state)
    compiled_fn = torch.compile(gg.to_tensor_batch, dynamic=False, fullgraph=False)
    compiled = compiled_fn(state)
    assert torch.allclose(eager, compiled, atol=1e-6), "to_tensor compiled != eager"


def test_legal_mask_compile_matches_eager():
    boards = _random_positions(200, seed=22)
    gg = GpuChessGame()
    state = gg.from_python_chess(boards, device="cuda")

    eager = gg.legal_mask(state)
    compiled_fn = torch.compile(gg.legal_mask, dynamic=False, fullgraph=False)
    compiled = compiled_fn(state)
    assert torch.equal(eager, compiled), (
        f"legal_mask compiled != eager; "
        f"{(eager != compiled).any(dim=-1).sum().item()} games differ"
    )


def test_step_batch_compile_matches_eager():
    """Pick legal action per game, step both eager and compiled, compare every field."""
    boards = _random_positions(200, seed=23)
    gg = GpuChessGame()
    state_e = gg.from_python_chess(boards, device="cuda")
    state_c = gg.from_python_chess(boards, device="cuda")

    mask = gg.legal_mask(state_e).cpu()
    actions_list: list[int] = []
    for i in range(len(boards)):
        nz = mask[i].nonzero(as_tuple=True)[0]
        actions_list.append(int(nz[0].item()) if len(nz) > 0 else 0)
    actions = torch.tensor(actions_list, dtype=torch.int64, device="cuda")

    eager_state, eager_r, eager_d = gg.step_batch(state_e, actions)
    compiled_fn = torch.compile(gg.step_batch, dynamic=False, fullgraph=False)
    comp_state, comp_r, comp_d = compiled_fn(state_c, actions)

    assert torch.equal(eager_state.pieces, comp_state.pieces), "pieces"
    assert torch.equal(eager_state.side, comp_state.side), "side"
    assert torch.equal(eager_state.castling, comp_state.castling), "castling"
    assert torch.equal(eager_state.ep, comp_state.ep), "ep"
    assert torch.equal(eager_state.halfmove, comp_state.halfmove), "halfmove"
    assert torch.equal(eager_state.fullmove, comp_state.fullmove), "fullmove"
    assert torch.equal(eager_state.ply, comp_state.ply), "ply"
    assert torch.equal(eager_state.done, comp_state.done), "done"
    assert torch.equal(eager_state.winner, comp_state.winner), "winner"
    assert torch.equal(eager_state.rep_count, comp_state.rep_count), "rep_count"
    assert torch.equal(eager_r, comp_r), "reward"
    assert torch.equal(eager_d, comp_d), "done(returned)"
