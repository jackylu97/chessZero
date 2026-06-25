# Plan — Root tablebase probing in MuZero self-play MCTS (2026-06-25)

## Goal
Break the conversion ceiling (model reaches +3 material in ~78% of games, converts
6–8%) by injecting Syzygy ground truth **at the root of self-play MCTS**, so the
search plays the winning conversion move. The corrected visit distribution becomes
the policy target → the policy head *learns* the technique (the thing value-only
rescoring can't teach). See `decisive_signal_plan_2026_06_23.md` and the
conversion analysis for the motivation.

## Why root-only (MuZero constraint)
Leela probes WDL at internal search nodes because it searches the true tree.
MuZero searches in **latent space** — internal nodes have no board, so we can't
probe them. But the root always has a real board, and **self-play steps through
every position as a root**, so root-only probing covers the entire conversion
(each ply of a KQ-K mate is its own root). This is the only place we *can* probe,
and it's enough.

## What gets overwritten (mirrors the existing repetition penalty)
In `tensor_mcts._select`, the PUCT score is `prior_score + value_score`, where
`value_score` is the normalized mover-POV Q of each child. The root-terminal-draws
feature already overwrites `value_score` for repeating root children. We add the
identical hook for TB: for root children (`cur == 0`) of games whose root is a
≤N-piece position, replace `value_score` with the **tablebase verdict of that move**:
- winning move → ≈ +1, minus a small DTZ penalty so the **shortest-DTZ** (progress)
  move scores highest (this is the within-won-region gradient flat-WDL lacks)
- drawing / win-throwing move → draw_score / −1

Visits then flow to the progress move → it's played (argmax visits) AND becomes the
policy target. **Soft bias on `value_score` (the Q term), NOT a hard policy-prior
boost** — Lc0 disabled a direct DTZ *policy* boost over KLD issues, so we steer via
value and keep the visit distribution smooth.

## Components
1. **Tablebase files** — python-chess ships a real small Syzygy set (KQvK, KRvK,
   KPvK, KBNvK, + 4/5-man) covering exactly the basin endgames. Copy to
   `data/syzygy/` (stable, gitignored). Production: download full 3-4-5-man (~1 GB)
   to the same dir.
2. **`chess_gpu.state_to_board(state, i)`** — decode game i's GPU batched state
   (pieces bitboards + side + castling + ep + halfmove) into a `chess.Board`.
3. **`src/games/syzygy_probe.py : SyzygyRootProber.root_move_values(state, legal_mask)`**
   → `[N, A]` float tensor, mover-POV TB value per legal move, NaN where not
   classifiable (game not in TB range / move not in TB). Piece-count gate on GPU
   (cheap); only ≤N-piece games hit the CPU probe. probe_wdl for all child moves,
   probe_dtz only for winners (ordering). 50-move rule: cursed-win/blessed-loss → draw.
4. **`tensor_mcts`** — `run_batch_gpu(root_tb_value=...)` gathers `[N,A]→[N,K]`
   (same as `forced_draw_mask`) into `self._root_tb_value`; `_select` override;
   force non-triton backend when on (like terminal-draws).
5. **`self_play.play_games_parallel_gpu_resident`** — build `root_tb_value` each
   ply, pass to `run_batch_gpu`. GPU-resident sim loop untouched; only a selective
   root excursion for in-TB games.
6. **config + CLI** — `tb_root_probe` (bool), `tb_path` (str, default data/syzygy),
   `tb_max_pieces` (int, 5), `tb_dtz_weight` (float, 0.05).

## GPU-resident property
The 800-sim simulation loop stays 100% on GPU. Per ply we add ONE selective
CPU excursion: GPU popcount → for ≤N-piece games only, copy boards to CPU, probe,
copy a `[N,A]` value tensor back. Middlegame plies have zero in-TB games → zero
overhead. Cache by FEN; probe_dtz only for winning moves.

## Verification
- `state_to_board` round-trip: reset_batch(FENs) → state_to_board → FEN matches.
- `root_move_values` on KQvK: all queen-keeping moves get +1, shortest-DTZ highest,
  king-blunder (stalemate/queen-hang) gets ≤0.
- `_select` TB override: synthetic `_root_tb_value` pins value_score (mirror the
  terminal-draws test).
- Smoke self-play arm: turn on, confirm threefold↓ / win_natural↑ on bare endgames,
  no throughput collapse, GPU-resident loop intact.

## Risks / open
- **Throughput**: many simultaneous in-TB games late in self-play → many probes.
  Mitigated by wdl-first/dtz-for-winners + FEN cache; monitor games/sec.
- **Generalization**: teaches ≤N-piece conversions; whether it generalizes to
  "+3 in a middlegame" is the open question (broader TB / seed needed if not).
- **KLD**: keep the override a bounded value bias, not a one-hot policy spike.
- **Square/ep/castling conventions** must match python-chess (covered by round-trip test).
