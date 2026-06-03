# GPU Utilization / Saturation — Action Items

_Analysis only — no code changes made. Written 2026-06-02 while run `2026_06_02_invdyn_full` was training._

## Context / the measured problem

- Self-play runs at **~40% GPU utilization** (measured via `gpu/util_pct` + `nvidia-smi`). So self-play is **CPU / sync / kernel-launch bound, not GPU-compute bound** — the GPU sits idle ~60% of the time waiting on CPU work (MCTS tree walk, env stepping, the eager PUCT select).
- Self-play **dominates wall-clock** (~30–60 min per batch at full sims=200/max_plies=1024). Training steps are GPU-bound but brief (a 512-step window is seconds).
- GPU **memory is wide open**: ~3–7 GB used of 24 GB.
- So the levers are: (a) raise self-play GPU saturation, and/or (b) cut total self-play work.

## Levers, ranked by ROI vs. risk

### 1. Bump `num_parallel_games` — cheapest, likely biggest easy win, low risk
- Currently 256 (= `batch_size`). Each MCTS simulation batches only 256 games, so launch/sync overhead isn't amortized → low util.
- With the large memory headroom, raising it (e.g. **512–1024**) means bigger batched network forwards per sim → more GPU work per launch → higher utilization and fewer per-game syncs.
- One config change. **Verify memory first:** TensorMCTS `node_hidden` scales as `parallel_games × (2·num_sims+? ) × C×H×W` (fp16). At 256/200/256ch it was ~3.4 GB fp16; doubling parallel games roughly doubles it — still fits 24 GB, but check.
- Note: a self-play batch then produces more games; may want to revisit `self_play_interval` so the train:self-play ratio stays sane.

### 2. Lower `max_plies` — biggest wall-clock cut (already queued, gated)
- Cuts the dead-draw tail (avg game 453 plies, up to the 1024 cap; most of it draw-shuffling). ~linear self-play time reduction.
- **Sequence after `q_ratio`** — capping relabels long games as artificial draws, worsening value saturation unless the q-blend supplies graded value first. See the next-run analysis bundle + q_ratio plan in memory.

### 3. Profile to pin the exact bottleneck — do first, cheap, no changes
- A one-shot `torch.profiler` trace of a single self-play batch would show exactly where the GPU-idle 60% goes: CPU MCTS ops vs env stepping vs syncs vs kernel-launch gaps.
- This confirms whether (1) `num_parallel_games`, (3) Triton re-enable, or env optimization is the right target instead of guessing.
- The observability patch already gives the macro signal (`self_play/plies_per_sec`, `gpu/util_pct`); the profiler gives the micro breakdown.

### 4. Re-enable the Triton MCTS backend — if the inductor SIGSEGV can be isolated
- We switched Triton → **eager** for stability (the autocast + autograd + **inductor** SIGSEGV when the inductor compile-worker pool coexists with the training backward). Eager is more CPU/launch-bound → contributes to the low util; Triton's fused PUCT-walk kernel does the whole depth walk in one launch/sim.
- Possible isolations (medium effort, WSL-risky): run self-play in a **subprocess** (inductor pool dies before the training backward), `torch._dynamo.reset()` between self-play and training, or otherwise tear down dynamo/inductor state at the phase boundary.
- Upside: recover Triton's ~1.2–2× select speedup and shrink the GPU-idle gaps. Only pursue if profiling (3) shows the select/launch overhead is a big share.

### 5. Async actor–learner (the "training + MCTS in separate threads" idea) — right architecture, wrong primitive, big project
- **The instinct is exactly right and is the standard AlphaZero/MuZero design**: decouple actors (self-play) from the learner (training) so the learner keeps the GPU busy while actors' CPU-bound work would otherwise leave it idle. It directly targets the 60% idle.
- **But not threads.** The self-play bottleneck is **CPU-bound** (MCTS tree ops, env, eager select), and Python's **GIL serializes CPU work across threads**. CUDA calls release the GIL so *some* overlap is possible, but the actual bottleneck (CPU MCTS orchestration) stays serialized → threads buy little.
- **Separate processes** (true actor-learner) would work (no GIL), but require: actor↔learner weight sync, a cross-process replay buffer, and tolerating stale-weight/off-policy actors (AlphaZero does this fine). That's a substantial refactor of the currently-synchronous train loop.
- **Single-GPU ceiling:** actor and learner time-slice one GPU; the gain is filling the actor's idle gaps with learner work → realistically **~1.3–1.7× wall-clock, not 2×**.
- **WSL stability:** more concurrent GPU work raises the SIGSEGV/instability risk we've been fighting.
- **Verdict:** correct long-term direction, but do the cheap levers first. Revisit async actor-learner once the recipe is settled and ideally in a more stable / multi-GPU environment.

### 6. Env-side (GPU chess) — modest
- `use_gpu_chess` + `use_gpu_resident_self_play` already minimize CPU↔GPU syncs (0 syncs/ply, 1 bulk transfer/batch). The deferred Triton-port of `chess_gpu.step_batch` (see memory) could shave env time. Modest absolute savings.

## Recommended order

**(3) profile → (1) `num_parallel_games`↑ → (2) `max_plies`↓ → (4) Triton-isolation → (5) async actor-learner (last, biggest).**

Start with profiling to confirm the bottleneck, then the one-line `num_parallel_games` bump is the highest-ROI low-risk change. Hold the async actor-learner refactor until the recipe is dialed in and the WSL/GPU environment is more trusted.
