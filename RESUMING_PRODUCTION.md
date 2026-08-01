# Resuming the production run — `2026_07_08_prod_xl_ft_s800`

Handoff written 2026-08-01 at step ~142k. Read top to bottom before relaunching.

## Current state
- Preset `chess_hybrid_xl`, 600k-step schedule, at **step 142000** (~24% done).
- Latest measured strength: **ckpt 141000 vs 60000 = −115 Elo** (8W/16D/26L→11W/12D/27L
  over the last two reads; up from the −182 trough). The opening deficit that
  defined the trough is fixed — the model plays even-to-ahead through ~move 20.
- **Warmstart restored to 40%** and self-renewing (the injection pool now cycles).
  This is the intervention the late-July work was about; keep it at 40% until a
  graduation test says the model is near the SF-depth-8 teacher.
- Remaining deficit is in the **middlegame→endgame** (see "Next work").

## What's in the export tarball (`chesszero_export_2026_08_01.tar`, 13G)
Extract it at the repo root; paths land in place:
- `checkpoints/chess/2026_07_08_prod_xl_ft_s800/` — checkpoints 10k–140k + 142000
  (`.pt` weights + `.buf` replay buffers).
- `data/stockfish_injection/` — the SF warmstart pool (REQUIRED for warmstart).
- `data/tb_anchor/`, `data/endgame_seeds_train_v2.txt` (+ `.meta.json`),
  `data/endgame_seeds_train.txt` — anchor + seed archives.
- `runs/chess/2026_07_08_prod_xl_ft_s800/` — TensorBoard history.
- `chesszero_main.bundle` — the git history (see "Getting the code").
- logs, h2h dumps, launch scripts, and the full Claude conversation history.

## What you MUST re-obtain (NOT in the tarball — public/regenerable)
1. **Tablebases — REQUIRED (the trainer probes them every step).** No script
   fetches these; the tarball skips them (6.6G+939M of public data).
   - Syzygy WDL+DTZ (3-4-5-man) → `data/syzygy/`. Standard source: the Lichess
     mirror `https://tablebase.lichess.ovh/tables/standard/{3,4,5}-piece/`.
   - Gaviota DTM (≤5-man) → `data/gaviota/`. Standard Gaviota `.gtb.cp4` set.
   Without these, launch fails at the first TB probe.
2. **Stockfish binary** → `tools/stockfish/`. Run `bash scripts/setup_stockfish.sh`
   (or `apt-get install stockfish` and point to it).
3. **Python env.** `bash scripts/runpod_setup.sh` (assumes a PyTorch base image
   with torch+CUDA already matched — it installs only the small deps + Stockfish).
   - If instead you `uv sync` from `uv.lock`: **GOTCHA** — the lock pins a `+cu130`
     torch build; on an older-driver box CUDA will silently be unavailable.
     Reinstall the matching build (e.g. cu128) if `torch.cuda.is_available()` is False.

## Getting the code onto the new machine
The commit history is NOT yet on GitHub (push was credential-blocked from the
Fly box). Two options:
- From the bundle in the tarball: `git clone chesszero_main.bundle chessZero`
  then `git -C chessZero remote set-url origin https://github.com/jackylu97/chessZero.git`
- Once you can auth, push: `git push origin main` (9 commits, clean fast-forward).

## Resume steps
1. Clone the repo (bundle or GitHub) and `cd` in.
2. Extract the tarball at the repo root.
3. Re-obtain tablebases + Stockfish + env (above).
4. Launch: `bash train_logs/launch_ft2_reanalyze.sh`
   - It auto-resolves the NEWEST `checkpoint_*.pt` that has a matching `.buf`.
   - Pin one explicitly with `RESUME_OVERRIDE=checkpoints/chess/2026_07_08_prod_xl_ft_s800/checkpoint_142000.pt bash train_logs/launch_ft2_reanalyze.sh`.
5. **VERIFY the restart** — it is NOT confirmed until you see BOTH lines:
   `Resuming from: …/checkpoint_142000.pt` and `Loaded replay buffer (9940 games)`.
   A missing/truncated `.buf` silently bootstraps an EMPTY buffer (symptom:
   `buf=<small>` and a negative loss) — kill and fix if you don't see the load line.
6. Confirm warmstart is flowing: `train/batch_warmstart_frac` should climb to
   ~0.40 within a few hundred steps (look for `injection pool CYCLED` in the log
   the first time the pool wraps).

## Operational gotchas (all learned the hard way this run)
- **`.buf` writes are not atomic.** Never SIGTERM/SIGKILL the trainer at a %1000
  checkpoint step — you can truncate the buffer. Check the step parity first.
- **Compile caches must live on a roomy volume.** The launch script sets
  `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` to `/home/...`; on Fly the root `/`
  (which holds `/tmp`) is tiny and shared — it filled and crashed the run once.
- **GPU eval probes vs the live trainer.** An uncapped h2h/eval alongside the
  trainer OOMs it (contention). For a trustworthy Elo, STOP the trainer, run the
  h2h full-GPU, restart. A memory-capped probe run concurrently is UNRELIABLE
  (it gave −315 vs a clean −164 on identical inputs).
- **Clean stop.** SIGTERM the python; its shutdown can DEADLOCK — SIGKILL after
  ~100s, then wait for the dead process's CUDA context to release (nvidia-smi
  shows the pid draining) before relaunching.
- **The janitor** keeps 10k milestones + the most-recent checkpoint; intermediate
  ones (e.g. 141000) get pruned as newer ones save. Copy aside any exact
  checkpoint you want to keep for the record.

## Next planned work (single-variable, in order)
1. **`endgame-seed-frac` 0.30 → ~0.50.** Restoring warmstart to 40% halved the
   won-endgame seed share of each batch (seeds ride the self-play channel), which
   caused a conversion dip (self-corrected to ~0.63 but below the ~0.70 pre-change
   level). Bumping the seed frac restores conversion practice without cutting
   warmstart. Do this AFTER confirming the −115 keeps climbing, so attribution stays clean.
2. **Watch drawn-endgame value over-confidence** — the health probe found TB-drawn
   `|V|>0.5` doubled (9%→20%); the model increasingly can't tell a drawn K+P from
   a won one. This and (1) are the middlegame→endgame frontier.
3. **Background diagnostic:** representation effective-rank keeps compressing
   (29.8→22.7 PR), though win/loss is still perfectly decodable (AUC 1.0).

## Where the detailed history lives
Committed docs: `veto_target_fix_2026_07_22.md`,
`resignation_relabel_policy_2026_07_20.md`, `design.md`, `bug_hunt_*.md`,
`strategy_2026_07_02.md`. Full investigation notes: the Claude transcripts +
`.claude/.../memory/` in the tarball.
