# GPU-Offload / Throughput Audit (2026-06-19)

5-agent codebase audit (self-play loop, tensor-MCTS, GPU chess env, training/buffer, sync sweep)
+ synthesis. All load-bearing claims were re-verified against source.

## Honest framing: "move to GPU" is the wrong question for self-play

Self-play is **launch-dispatch + host-sync bound on one CPU core**, NOT FLOP-bound — the env/MCTS are
*already* GPU-resident. So the self-play wins are **launch-count reduction + sync removal**, not offload.
The genuinely GPU-offloadable CPU work lives on the **training/buffer side** (dense-policy expansion in
`make_target`, the 8-frame history re-cat, obs dtype, sparse→dense scatter).

**Expected gains:** incremental fixes (compile, table-cache, sync removal, dedupe) ≈ **1.5–2.5×** on the
launch-bound loop. The hard ceiling only breaks with **CUDA-graph capture of the per-sim step** (now
feasible — see below) or a fused expand/backprop kernel.

## Quick wins (S-effort, low-risk, do first)

1. **`tensor_mcts_compile_net=True` in the chess preset** — it's OFF by default (`config.py:479`, absent from
   the chess preset), so production self-play ran the per-sim net forward *fully eager* (~40 kernels/sim ×200
   sims). The 1.4× fix targets exactly the bottleneck. *(cold2 runs already pass `--tensor-mcts-compile-net`;
   confirmed healthy/no-SIGILL — the local-handle approach avoids the old reanalyze/BatchedMCTS SIGILL path.)*
2. **Device-cache the constant lookup tables in `chess_gpu`** (PAWN/KNIGHT/KING/ZOBRIST_*/ACTION_*/BB_*) —
   currently re-`.to(device)`'d on *every* per-ply call → dozens of tiny H2D copies/launches/ply. Cache once. Bit-identical.
3. **Remove the two per-ply `.any()` host syncs** — `ep_valid.any()` (`chess_gpu.py:599`) and
   `is_black.any()` (`:1347`). Each forces a `cudaStreamSynchronize` every ply; the `torch.where` below each
   already gives the correct all-false result, so removal is behavior-neutral. *Also a prerequisite for graph capture.*
4. **Delete the redundant 2nd `_compute_check_resolve`** (`chess_gpu.py:1071-1073`) — identical args to `:1035`; rebind its returned `in_check`/`checkers`.
5. **`torch.from_numpy(is_weights).to(device, non_blocking=True)`** instead of `torch.tensor(..., device=cuda)` (`trainer.py:732`); same for the small scalar batch tensors.
6. **Pack `total_loss` into the single per-step `td_errors` host transfer** so the NaN-guard + priority update sync once/step, not twice.

## Medium-effort (training-side throughput; GPU-compute-bound side)

7. **Gate the trainer's per-step diagnostics behind `log_now`** — ~20 `.item()`/`.cpu()` syncs run *every* step
   (`trainer.py:917-991, 1000-1027`) but `loss_info` is only consumed at `step % log_interval` (`:250`); `log_now`
   already exists at `:903`. Only the `td_errors`→priority transfer must run every step. Defers ~20 syncs/step.
8. **Prefetch + pin + uint8/fp16 the training batch; stack 8-frame history on-GPU** — `sample_batch`→256 inline
   single-threaded `make_target()` calls run synchronously before the forward (no prefetch); `target_observations`
   is a ~138–664 MB fp32 tensor moved blocking (no pin/non_blocking), ~88% duplicated history frames. Transfer
   single-frame uint8 once + build the 8-frame stacks on GPU → ~138 MB→~20-35 MB H2D, hidden behind the prior step.
9. **Reuse the already-computed zobrist hash + rep-match-count** (`to_tensor_batch` re-hashes every ply though
   `_step_batch_impl` computed the identical `new_hash`/match set one call earlier — `chess_gpu.py:1799,1836` vs `:1384`).

## Structural bets (L-effort, the ceiling-breakers)

- **CUDA-graph the steady-state per-sim MCTS step.** `subtree_reuse=False` in the chess preset means
  `advance_root`/re-alloc never run, so the only graph-incompatibility left is the `_expand` scatter — which is
  data-dependent in *value* but *static in shape* (graphs allow content changes between replays). Capture
  select(Triton)+net-forward+expand+backprop after warmup, replay 200×/ply → collapses ~200×40 Python-driven
  launches to a few replays. **The one change that fundamentally removes single-core Python dispatch from self-play.**
  Prereqs: the `.any()` removals (#3) + RNG capture-safety.
- **Fused expand+backprop Triton kernel** (cuts per-sim launches ~40→~3) — fallback if graph capture is RNG/shape-unsafe.
- **torch.compile the whole per-ply env path** (`_step_batch_impl`+`_legal_mask_impl`+`to_tensor_batch`+`_zobrist_hash`) — fires hundreds of tiny eager kernels/ply today; static shapes. Prereqs: #2 + #3.
- **Vectorize the end-of-batch history build** (`self_play.py:611-633`, ~38k Python iters of per-element
  from_numpy/clone/copy/nonzero.tolist after the GPU goes idle) + **GPU-vectorize `make_target`** (sparse→dense scatter).

**Validation for any env/MCTS change:** bit-exactness vs `tests/test_chess_gpu_legal.py`, the threefold /
insufficient-material tests, and the `tensor_mcts` replay-equivalence oracle — this path is bit-sensitive.
