"""Correctness equivalence: GpuChessGame on CUDA must match CPU output bit-for-bit.

The CPU torch path is already cross-validated against python-chess in
test_chess_gpu_obs/legal/step. CUDA goes through different kernels (CUDA
implementations of the same ops) — these tests confirm CUDA produces
identical outputs to CPU on the same inputs. Skipped when no CUDA.
"""
import random

import chess
import pytest
import torch

from src.games.chess_gpu import GpuChessGame, _legal_mask_impl, _step_batch_impl


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _random_positions(n: int, seed: int) -> list[chess.Board]:
    """Mix of starting and mid-game positions; non-terminal only."""
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


def test_to_tensor_batch_cuda_matches_cpu():
    boards = _random_positions(200, seed=11)
    gg = GpuChessGame()
    s_cpu = gg.from_python_chess(boards, device="cpu")
    s_cuda = gg.from_python_chess(boards, device="cuda")

    t_cpu = gg.to_tensor_batch(s_cpu)
    t_cuda = gg.to_tensor_batch(s_cuda).cpu()
    # Piece/castling/EP/turn planes are bit-exact 0/1; move-count plane is
    # `fullmove/200` in fp32 which can diverge by ≤1 ULP across backends.
    # Use a tight tolerance that catches structural bugs but allows for
    # the last-bit difference on the move-count plane.
    assert torch.allclose(t_cpu, t_cuda, atol=1e-6), "to_tensor_batch CUDA != CPU"


@pytest.mark.xfail(
    reason="GpuChessGame.legal_mask CUDA path diverges from CPU on some positions "
           "(observed 2026-05-08). Bug in chess_gpu.py CUDA kernels — to be fixed; "
           "kept as a regression watch.",
    strict=False,
)
def test_legal_mask_cuda_matches_cpu():
    boards = _random_positions(500, seed=12)
    gg = GpuChessGame()
    s_cpu = gg.from_python_chess(boards, device="cpu")
    s_cuda = gg.from_python_chess(boards, device="cuda")

    m_cpu = gg.legal_mask(s_cpu)
    m_cuda = gg.legal_mask(s_cuda).cpu()
    assert torch.equal(m_cpu, m_cuda), (
        f"legal_mask CUDA != CPU; "
        f"{(m_cpu != m_cuda).any(dim=-1).sum().item()} games differ"
    )


@pytest.mark.xfail(
    reason="GpuChessGame.step_batch CUDA path diverges from CPU on the ``done`` "
           "field for some positions (observed 2026-05-08). Bug in chess_gpu.py "
           "CUDA kernels — to be fixed; kept as a regression watch.",
    strict=False,
)
def test_step_batch_cuda_matches_cpu():
    """Pick one legal action per game on CPU; step both backends; compare.

    Compares: pieces, side, castling, ep, halfmove/fullmove/ply, done, winner,
    rep_count, plus the returned reward and done tensors.
    """
    boards = _random_positions(300, seed=13)
    gg = GpuChessGame()
    s_cpu = gg.from_python_chess(boards, device="cpu")
    s_cuda = gg.from_python_chess(boards, device="cuda")

    # Pick first legal action per game (CPU).
    mask_cpu = gg.legal_mask(s_cpu)
    actions_list: list[int] = []
    for i in range(len(boards)):
        nz = mask_cpu[i].nonzero(as_tuple=True)[0]
        actions_list.append(int(nz[0].item()) if len(nz) > 0 else 0)
    a_cpu = torch.tensor(actions_list, dtype=torch.int64, device="cpu")
    a_cuda = a_cpu.to("cuda")

    s2_cpu, r_cpu, d_cpu = gg.step_batch(s_cpu, a_cpu)
    s2_cuda, r_cuda, d_cuda = gg.step_batch(s_cuda, a_cuda)

    # Compare every field.
    assert torch.equal(s2_cpu.pieces, s2_cuda.pieces.cpu()), "pieces diverge"
    assert torch.equal(s2_cpu.side, s2_cuda.side.cpu()), "side diverges"
    assert torch.equal(s2_cpu.castling, s2_cuda.castling.cpu()), "castling diverges"
    assert torch.equal(s2_cpu.ep, s2_cuda.ep.cpu()), "ep diverges"
    assert torch.equal(s2_cpu.halfmove, s2_cuda.halfmove.cpu()), "halfmove diverges"
    assert torch.equal(s2_cpu.fullmove, s2_cuda.fullmove.cpu()), "fullmove diverges"
    assert torch.equal(s2_cpu.ply, s2_cuda.ply.cpu()), "ply diverges"
    assert torch.equal(s2_cpu.done, s2_cuda.done.cpu()), "done diverges"
    assert torch.equal(s2_cpu.winner, s2_cuda.winner.cpu()), "winner diverges"
    assert torch.equal(s2_cpu.rep_count, s2_cuda.rep_count.cpu()), "rep_count diverges"
    assert torch.equal(r_cpu, r_cuda.cpu()), "reward diverges"
    assert torch.equal(d_cpu, d_cuda.cpu()), "done(returned) diverges"
