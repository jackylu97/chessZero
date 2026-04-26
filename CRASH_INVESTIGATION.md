# Chess training SIGSEGV — active investigation (2026-04-23)

Session context snapshot for WSL-crash recovery. Delete when resolved.

## What happened

1. Re-ran the Stockfish-injection rehydration (`scripts/prehydrate_stockfish_shards.py`) to bypass the known `board.copy()` SIGILL in the 1-in-7500-games range. All 64 shards (43.4 GB) materialized as legacy `list[GameHistory]` pickles at `data/stockfish_asymmetric_v1_rehydrated/`.
2. One rehydrated shard (`bucket_8v7/worker_1/shard_0002.pkl`) came out truncated at 805 MB (the pre-resume session had crashed mid-write). Deleted and re-ran; it's now 1697 MB, in line with its siblings (1725/1735/1767).
3. Launched chess training against the rehydrated pool: run `2026_04_23_0008`. Crashed ~9 min in at step ~2400.
4. dmesg showed **SIGSEGV (signal 11), not SIGILL** — `segfault at 0 ip 0000651c10eb6a19 ... in python3.10`. Null-pointer deref inside the CPython binary. No Python traceback because `train.py` was not launched under the faulthandler bootstrap.
5. Rehydration had eliminated `board.copy()` from the shard-read path. At step 2400 (eval_interval=5000, self-play/reanalyze gated off until pool exhausts), no obvious `game.step()` caller should be live, yet the crash matches the same heap-corruption signature. Bug is somewhere other than the paths the rehydration neutralized.

## Current state (training resumed with diagnostics)

Resumed from `checkpoints/chess/2026_04_23_0008/checkpoint_2000.pt` under diagnostic instrumentation. No saved `.buf` existed at step 2000, so buffer is being rebuilt from the injection pool (cursor resumed at 2548 games consumed → 3048 after initial bootstrap).

### Launch command

Preferred (auto-restart on crash via supervisor):
```bash
tmux new -s chess_train \
  'PYTHONMALLOC=debug scripts/supervise_train.sh \
     --game chess \
     --run-id 2026_04_23_0008 \
     --stockfish-injection-path data/stockfish_asymmetric_v1_numpy'
```
The supervisor auto-discovers the latest `checkpoints/chess/<run-id>/checkpoint_*.pt`
and passes `--resume` itself — no need to specify it. On crash: backs off, relaunches,
caps total retries and consecutive no-progress retries. Drain cleanly with
`touch runs/chess/<run-id>/STOP`. Exits 0 on training completion, 130/143 on
user interrupt, otherwise last crash's rc.

Raw one-shot (no auto-restart, still useful for debugging a single attempt):
```bash
PYTHONMALLOC=debug PYTHONFAULTHANDLER=1 \
  .venv/bin/python scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess \
  --run-id 2026_04_23_0008 \
  --resume checkpoints/chess/2026_04_23_0008/checkpoint_2000.pt \
  --stockfish-injection-path data/stockfish_asymmetric_v1_numpy \
  2>&1 | tee -a chess_train.log
```

Note: training reads the canonical numpy-obs v1 pool at
`data/stockfish_asymmetric_v1_numpy`. The intermediate torch-obs and original
v2 compact pools have been renamed with `DEPRECATED_` prefixes — each has a
`DEPRECATED.md` inside. Safe to delete once the numpy pool is validated by a
full training run.

### tmux sessions
- `chess_train` — the training run above
- `pyspy` — rotating py-spy dump loop (below)

### py-spy dump loop (session `pyspy`)
```bash
mkdir -p py-spy-dumps
while true; do
  pid=$(pgrep -f 'scripts/train.py' | head -1)
  if [ -n "$pid" ]; then
    ts=$(date +%H%M%S)
    .venv/bin/py-spy dump --pid $pid --locals > py-spy-dumps/${ts}.txt 2>&1 \
      || echo "[$(date +%H:%M:%S)] py-spy attach failed (pid=$pid)"
    ls -t py-spy-dumps/*.txt 2>/dev/null | tail -n +61 | xargs -r rm
  fi
  sleep 30
done
```

### ptrace prerequisite
py-spy needs `kernel.yama.ptrace_scope=0` (session-only, resets at reboot):
```bash
sudo sysctl kernel.yama.ptrace_scope=0
```

## Diagnostic coverage armed

- **`PYTHONMALLOC=debug`** — CPython guarded allocator; turns heap corruption into a crash at the corruption site (not the next-touch site).
- **Faulthandler bootstrap** (`scripts/_faulthandler_bootstrap.py`) — all-threads Python traceback on SIGSEGV.
- **py-spy rotating dumps** — last ~30 min of Python stacks (60 dumps × 30s).
- **dmesg** — kernel-side segfault records from WSL's CaptureCrash (also writes minidumps to WSL's dump dir).

If it crashes again we'll have: faulthandler traceback in `chess_train.log`, debug-malloc abort with corruption site, and py-spy stacks for the last minute of activity.

## Recovery after WSL crash

```bash
# 1. Re-lower ptrace (sysctl doesn't persist across boots)
sudo sysctl kernel.yama.ptrace_scope=0

# 2. Relaunch training from the latest checkpoint
ls -t checkpoints/chess/2026_04_23_0008/checkpoint_*.pt | head -1
# pass that path as --resume in the launch command above

# 3. Restart the py-spy loop in a separate tmux session (see above)
```

## Ruled out

- Rehydration-time `board.copy()` SIGILL — rehydration eliminated this path for shard reads. Training is consuming rehydrated legacy shards (`isinstance(first, list)` branch of `_iter_shard_games`).
- Compact-shard replay path — not invoked for rehydrated legacy shards.
- Self-play / reanalyze `board.copy()` — gated off until injection pool exhausts (~step 30k).
- Eval-loop `game.step()` — `eval_interval=5000`, crash was at step 2400.

## Options considered (pick next if A+B don't pinpoint it)

- **A (active):** `PYTHONMALLOC=debug` + faulthandler.
- **B (active):** py-spy rotating dumps.
- **C:** `strace -f -o trace.log -e trace=all` — last syscall + originating lib.
- **D:** Stream shards instead of materializing (`_advance_injection_shard` calls `list(_iter_shard_games(...))` — loads whole 1.7 GB shard at once). Convert to generator → 5× lower peak memory. May incidentally fix the bug if corruption is allocator-stress-correlated.
- **E:** Subprocess-isolate shard loads (mirror the rehydration pattern — parent never unpickles). Heavier refactor; decisively quarantines any unpickle-side corruption.
- **F:** Run under valgrind memcheck or ASAN. 20–50× slowdown, but bug reproduces in 9 min → ~5 hr total. Definitive.
- **G:** Pin/bump `python-chess`. Cheap to try; noisy if it "fixes" by timing change.

## Next actions after the next crash

1. `tail -200 chess_train.log` — look for faulthandler traceback and debug-malloc abort message.
2. `ls -t py-spy-dumps/*.txt | head -5` — inspect the last few stacks pre-crash.
3. `dmesg -T | tail -50` — confirm the new crash signature (SIGSEGV vs. abort from debug malloc).
4. Based on where corruption is happening, pick from D/E/F.

## Culprit identified (2026-04-23, post-faulthandler)

Faulthandler caught it: crash is in **torch's legacy tensor unpickler** (`torch/serialization.py:1630 UnpicklerWrapper` → `torch/storage.py:535 _load_from_bytes`) during `pickle.load` of a rehydrated shard in `_iter_shard_games`. It's *not* python-chess — it's torch's pickle reducer for tensor storages corrupting the heap on large shards.

Immediate fix (applied): the affected pool was converted from `observations: list[torch.Tensor]` → `list[np.ndarray]` in-subprocess and re-pickled (one-time salvage script, since deleted). `_iter_shard_games` wraps numpy obs back to `torch.Tensor` via `torch.from_numpy` (zero-copy) on yield, so downstream code is unchanged. The durable fix is `generate_stockfish_games.py --format-version=1` emitting numpy-obs directly — see Resolution below.

## Resolution (2026-04-24, settled)

**Canonical format: v1 list with `np.ndarray` observations.** This is what `data/stockfish_asymmetric_v1_numpy/` holds (51 GB, 64 shards, 32k games across 4 buckets).

Three formats coexist in code; only one is production-safe for chess:

| Format | Disk | Training-safe? | Notes |
|---|---|---|---|
| v1 list + `np.ndarray` obs | ~2.8 MB/game | **Yes — canonical** | `_iter_shard_games` rewraps to `torch.Tensor` on yield. |
| v1 list + `torch.Tensor` obs | ~3.5 MB/game | No — SIGSEGV | Torch pickle reducer corrupts heap on large shards. |
| v2 compact streaming | ~10–50 KB/game | No — SIGILL | Training-time `from_compact_dict` replays actions through `game.step()`, hitting python-chess `board.copy()` SIGILL ~1-in-7500 games. |

### What the codebase now enforces

- `scripts/generate_stockfish_games.py` defaults to `--format-version=1`. The v1 path calls `.numpy()` on obs before pickling, so new pools are canonical by construction. `--format-version=2` prints a warning on startup.
- The historical salvage scripts (`prehydrate_stockfish_shards.py` and `np_convert_stockfish_shards.py`) were deleted 2026-04-24 — their job is done and the producer now emits canonical format directly. If a torch-obs or v2 pool ever reappears, the recipe is in this file's git history; mirror the prehydrate pattern (subprocess-isolated decode) and the np-convert pattern (`.numpy()` rewrap before re-pickle).

### What we did *not* do, and why

- **v2 compact migration of the current pool** — considered, rejected. Would shrink disk ~1000× but re-expose training to the `board.copy()` SIGILL. Not worth fighting today; disk pressure isn't binding (623 GB free) and supervisor+checkpointing handles the crash loop well enough.
- **Root-cause the `board.copy()` SIGILL** — would unlock the v2 path but high-variance (could be a day). Deferred indefinitely.

### What's left as follow-up

- Training supervisor (`scripts/supervise_train.sh`) — done 2026-04-24.
- WSL crash-dump flag flipped to `=1` (takes effect after next `wsl --shutdown`) — done 2026-04-24.
- Heartbeat + watchdog for hangs (supervisor only reacts to exit) — deferred Tier 2.
- Structural `GameHistory.observations` = numpy natively so no save site can regress — deferred Tier 2; the current guards cover the known producers.
