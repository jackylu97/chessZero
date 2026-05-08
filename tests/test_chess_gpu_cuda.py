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


def test_triton_kernels_handle_non_contiguous_inputs():
    """Regression: the Triton slider-attack kernels assume stride-1 layout via
    ``tl.load(ptr + offs)``. Bug 2026-05-08: the kernel wrappers didn't call
    ``.contiguous()`` on the seed tensor. Callers like ``_compute_pin_filter``
    pass column slices (e.g. ``state.pieces[:, P_KING]`` with stride 12),
    causing the kernel to read interleaved memory and produce wrong attack
    bitboards for batch indices > 0. Index 0 always reads correctly because
    offset 0 is stride-invariant.

    This test passes a non-contiguous seed (a column slice from a wider
    tensor) and verifies the wrapper handles it correctly by comparing
    against the same operation on a contiguous copy.
    """
    from src.games.chess_gpu_triton import (
        slider_attacks_8way,
        slider_attacks_8way_per_dir_empty,
    )

    n = 32
    # Build a wider int64 tensor on CUDA, take a column slice → stride > 1.
    wide = torch.randint(0, 1 << 60, (n, 12), dtype=torch.int64, device="cuda")
    wide |= 1 << 30  # sprinkle bits
    seed_strided = wide[:, 5]  # column slice, stride=12
    assert not seed_strided.is_contiguous(), "slice must be non-contiguous to trigger bug"

    empty = torch.full((n,), -1, dtype=torch.int64, device="cuda")  # all empty

    # The wrapper must produce the same result whether seed is contiguous or not.
    out_strided = slider_attacks_8way(seed_strided, empty)
    out_contig = slider_attacks_8way(seed_strided.contiguous(), empty)
    assert torch.equal(out_strided, out_contig), (
        "slider_attacks_8way produced different output for the same seed when "
        "passed via stride-12 slice vs contiguous copy — wrapper must call "
        ".contiguous() to be safe under non-stride-1 inputs."
    )

    # Same for per-dir-empty variant.
    empties = torch.full((n, 8), -1, dtype=torch.int64, device="cuda")
    out_strided_pd = slider_attacks_8way_per_dir_empty(seed_strided, empties)
    out_contig_pd = slider_attacks_8way_per_dir_empty(seed_strided.contiguous(), empties)
    assert torch.equal(out_strided_pd, out_contig_pd), (
        "slider_attacks_8way_per_dir_empty produced different output under "
        "non-contiguous seed — wrapper must call .contiguous()."
    )

    # End-to-end: verify legal_mask is correct on a non-trivial batch.
    # (This is what was actually broken in production.)
    import chess as pychess
    import random
    from src.games.chess import _move_to_action
    from src.games.chess_gpu import GpuChessGame

    rng = random.Random(99)
    boards = []
    while len(boards) < 50:
        b = pychess.Board()
        n_plies = rng.randint(0, 60)
        for _ in range(n_plies):
            if b.is_game_over(claim_draw=False): break
            moves = list(b.legal_moves)
            if not moves: break
            b.push(rng.choice(moves))
        if not b.is_game_over(claim_draw=False):
            boards.append(b)
    gg = GpuChessGame()
    s = gg.from_python_chess(boards, device="cuda")
    m = gg.legal_mask(s).cpu()
    for i, board in enumerate(boards):
        legal_pc = set(_move_to_action(mv, board.turn) for mv in board.legal_moves)
        legal_g = set(int(j) for j in m[i].nonzero(as_tuple=False).squeeze(-1))
        assert legal_pc == legal_g, (
            f"game {i} (FEN={board.fen()}): legal_mask CUDA vs python-chess differ. "
            f"missing={legal_pc - legal_g}, extra={legal_g - legal_pc}"
        )
