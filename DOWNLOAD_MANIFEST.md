# Download Manifest — before machine access is lost (2026-08-01)

Paths are relative to `/home/user/chessZero/`. Sizes approximate.

## Git / code — PUSH IS BLOCKED (do this first)
The commit `220f3ca` is made locally, but `git push` failed: this machine's SSH
key authenticates as **phulin**, who lacks push access to `jackylu97/chessZero`,
and there is no HTTPS token. **9 commits are unpushed** (mine + 8 pre-existing;
clean fast-forward onto remote `f572436`).

Two ways to get the code to GitHub:
- **Bundle (machine-independent):** download `chesszero_main.bundle` (1.6M). On
  any machine: `git clone chesszero_main.bundle chesszero && cd chesszero &&
  git remote set-url origin https://github.com/jackylu97/chessZero.git &&
  git push origin main`.
- **Push now with your own creds** (you still have this machine today):
  `git push https://<YOUR_PAT>@github.com/jackylu97/chessZero.git main`

## TIER 1 — ESSENTIAL (~1.5 G, irreplaceable)
- `chesszero_main.bundle` (1.6M) — the code (all 9 unpushed commits).
- `checkpoints/chess/2026_07_08_prod_xl_ft_s800/checkpoint_141000.pt` + `.buf` (705M)
  — latest measured model (−115 Elo vs 60k) + its replay buffer, to resume/eval.
  NOTE: training is LIVE and still writing newer checkpoints; grab the newest
  `checkpoint_*.pt`+`.buf` pair at download time for the most-trained weights.
- `checkpoints/chess/2026_07_08_prod_xl_ft_s800/checkpoint_60000.pt` + `.buf` (~700M)
  — the FIXED head-to-head eval reference (every Elo number is vs this).
- `runs/chess/2026_07_08_prod_xl_ft_s800/` (51M) — full TensorBoard metrics history.
- `data/endgame_seeds_train_v2.txt` + `.meta.json`, `data/endgame_seeds_train.txt` (~12M)
  — endgame seed archives (v2 = current, with ~13% drawn seeds).
- `selfplay_production.log`, `train_logs/h2h_vs60k_series.log`,
  `train_logs/h2h_games_*.jsonl`, `train_logs/dashboard_*.png` (~1M) — run +
  measurement records.

## TIER 2 — IMPORTANT if resuming TRAINING (~3 G)
- `data/stockfish_injection/` (2.7G) — the Stockfish warmstart/injection pool
  (generated; needed to keep warmstart alive on resume — this is the channel
  whose exhaustion caused the whole late-July saga).
- `data/tb_anchor/` (132M) — TB anchor demonstration archive.
  (Compressed copy already exists: `chesszero_prod_data.tgz`, 22M.)
- Remaining checkpoints (`checkpoint_{10000..140000}.pt`+`.buf`, ~12G) — only if
  you want every resume/eval point; otherwise latest + 60000 suffice.

## TIER 3 — RECONSTRUCTABLE (public, re-downloadable — skip unless convenient)
- `data/gaviota/` (6.6G) — public Gaviota DTM tablebase (≤5-man).
- `data/syzygy/` (939M) — public Syzygy WDL/DTZ tablebase.
- `tools/stockfish` — public binary; restore via `scripts/setup_stockfish.sh`.

## OPTIONAL — investigation notes / probe tooling
- `~/.claude/projects/-home-user-chessZero/memory/` — detailed saga notes
  (supplementary; the load-bearing findings are in committed repo docs:
  `veto_target_fix_2026_07_22.md`, `resignation_relabel_policy_2026_07_20.md`,
  `design.md`, `bug_hunt_*.md`, `strategy_2026_07_02.md`).
- Session scratchpad probe harnesses (value/policy/world-model health,
  h2h + phase-attribution scripts, no-warmstart baseline JSON) live under the
  session temp dir and will be lost with the machine; regenerable but handy if
  you want to re-run the analyses.

## Totals
- Tier 1: ~1.5 G   | Tier 1+2: ~16.5 G (incl. all checkpoints)
- Everything incl. public tablebases: ~24 G
