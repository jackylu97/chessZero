# ChessZero — Claude Context

## Project Overview
MuZero multi-game implementation in PyTorch. Three phases:
1. Single-game baseline (tic-tac-toe → connect4 → chess)
2. Multi-game joint training (shared backbone)
3. Latent state visualization/analysis

**GitHub:** `https://github.com/jackylu97/chessZero.git`

## Key Files
- `src/model/muzero_net.py` — All networks (Representation, Dynamics, Prediction, MuZero, MultiGameMuZero)
- `src/model/utils.py` — ResidualBlock, scalar_transform, scalar_to_support
- `src/mcts/mcts.py` — MCTS with MinMaxStats, PUCT; BatchedMCTS for parallel self-play and reanalyze
- `src/training/trainer.py` — Single-game training loop with reanalyze, PER, LR scheduling
- `src/training/multi_game_trainer.py` — Multi-game round-robin training
- `src/training/self_play.py` — Single and parallel (batched) self-play
- `src/training/replay_buffer.py` — Replay buffer with PER and reanalyze sampling
- `src/config.py` — Per-game hyperparameters
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/train_multi.py`
- `COMMANDS.md` — Quick-reference for training, eval, TensorBoard, and interactive play
- `design.md` — Full record of design decisions and deferred improvements

## Known Issues / Deferred Work

### Draw-basin ROOT CAUSE — V^π ≈ draw (CONFIRMED 2026-06-17, `scripts/probe_value_target_audit.py`)
The deepest issue is NOT a code bug: the self-play value target is the game
OUTCOME under the current (weak) policy. The policy can't convert wins, so
objectively-winning positions are played out to draws and labeled ~draw. Audit
of the 76k buffer: positions Stockfish rates >+3 for the side to move get mean
target z=+0.18 with **67% outright draws** (root_value only +0.26). So the value
head is trained to call winning positions ≈draw → can't rank positions/moves →
MCTS gets no signal → policy never improves → self-reinforcing. Sim-scaling
probe (`probe_sim_scaling.py`) confirms: more MCTS search HELPS at the warmstart
checkpoint but HURTS at self-play checkpoints (search amplifies the miscalibrated
value). A bigger value head does NOT fix this (target has no resolution); the fix
must INJECT decisive signal — win adjudication (don't play won positions out to
draws), persistent Stockfish value supervision, and/or TD bootstrap targets
(root_value tracks SF better, corr 0.47 vs z 0.18). See [[value-sibling-ranking]].

### Repetition-draw mechanism + MCTS audit (2026-06-17, `mcts_audit_2026_06_17.md`)
Traced 11 threefold self-play games (`scripts/trace_threefold.py`) + audited BOTH search
engines (`mcts.py` numpy, `tensor_mcts.py` live). Findings: repetition draws are won K+P
endgames the policy can't convert; the search has NO repetition awareness (value from the
latent net only — no terminal/draw branch in select/expand/backprop, both engines), so the
shuffle keeps full winning value (root_value RISES into the repetition) and is the top-Q
move → MCTS banks a phantom win into the draw. **MCTS math is CORRECT** (PUCT/backup/MinMax
identical numpy↔tensor to ~1e-16; no divergence). Three exploration issues, none a math bug:
(1) `dirichlet_alpha=0.1` too peaked → a prior-0.06 converting move starved ~half the
searches; **raise to 0.3**. (2) leaf nodes expand the FULL unmasked action space, both
engines (`mcts.py:146-147,640`, `tensor_mcts.py:1195-1197`) — known deferred issue #1;
**add leaf legal masking**. (3) no in-search repetition penalty (structural MuZero limit) —
**add MCTS-level repetition penalty + win adjudication**. More sims HURT (800 sims → 75%
threefold vs 200 sims → 46%). See [[repetition-draw-mechanism]], [[value-sibling-ranking]].

### Draw-basin pipeline bugs — 2026-06-13 (see `bug_hunt_2026_06_13.md`), now mostly FIXED
- **(FIXED) `select_action_gpu` / policy targets** (`src/mcts/tensor_mcts.py`,
  `src/mcts/mcts.py`): targets are now deduped raw visit fractions, temperature-
  independent (bug_hunt §A/§B). Plus the root now enumerates all legal moves when
  `num_legal <= sample_k` instead of always sampling (commit a9906eb).
- **(FIXED) GPU insufficient-material draw** (`src/games/chess_gpu.py:1864-1866`,
  commit `0da507f`, `tests/test_chess_gpu_insufficient_material.py`): wired into
  the `done` mask. NOTE: this only handles DEAD draws (KvK/KBvK/KNvK); it does NOT
  address winning-but-unconverted positions (that's the V^π issue above).
- **(MEDIUM, unverified) Mutating terminal observation** (`chess_gpu.py:1636-1722`):
  board fields may not be frozen for done games, possibly corrupting end-of-batch
  `final_obs`. Status not re-verified.

### 1. Illegal moves at MCTS leaf nodes (`src/mcts/mcts.py:127`)
At the root, legal actions from the game engine are correctly masked (legal_actions_list is now stored in GameHistory and used during reanalyze). At leaf nodes (latent space), all `action_space_size` actions are still treated as legal. Two consequences:
- Dynamics network trained only on legal transitions but queried on illegal ones (distribution mismatch)
- Policy targets (MCTS visit counts) include illegal move exploration

**Partial fix landed:** `GameHistory.legal_actions_list` now stores root legal actions per step; reanalyze uses them correctly. Leaf node masking during training unroll is still deferred.

### 2. Per-game prediction residual blocks not implemented
muzero-general runs additional residual blocks inside `PredictionNetwork` before the heads. Skipped for now — backbone depth is sufficient for single-game. Revisit for multi-game: per-game prediction blocks would give each game's heads dedicated processing capacity on top of the shared backbone. See `design.md` for full reasoning.

### 3. Game ID embeddings not wired up (`src/model/muzero_net.py`)
`self.game_embeddings` is defined in `MultiGameMuZeroNetwork` but not concatenated
to the hidden state. Wire it up before the shared backbone if multi-game training is unstable.

### 3. Action encoding collision (multi-game)
Action index N in tic-tac-toe and action index N in chess produce the same scalar.
The shared dynamics backbone can't distinguish them from the action signal alone.

**Deferred fix:** Per-game `nn.Embedding(action_space_size, embed_dim)` tables, broadcast
spatially. For small-action games (tictactoe/connect4) a one-hot spatial plane is even
better (directly encodes *where* the action was on the board). Keeping scalar broadcast for now.

## Architecture Summary

### Normalization — LayerNorm (not BatchNorm)
All norm layers in the network are `nn.LayerNorm([C, H, W])` — see `norm_layer()` in `src/model/utils.py`. `ResidualBlock` takes a required `spatial_shape` arg because LN's affine params are per-(C,H,W).

**Why not BN:** MuZero dynamics is applied recurrently during training unroll (K=5 applications per sample). BN's `running_var` EMA assumes IID batch statistics; unrolled hidden states violate this. An observed failure mode: a single outlier batch pulled dynamics BN `running_var` by 30–80×, losses spiked across all three heads simultaneously, and the net never recovered before crashing on NaN ~400 steps later. LayerNorm has no running stats → failure mode disappears. Matches LightZero's default and DeepMind's updated architecture. Full diagnosis in `design.md`.

**Residual block order:** post-activation (`Conv → Norm → ReLU → Conv → Norm → add → ReLU`), matching both muzero-general and LightZero.

### Training stability monitoring (trainer.py)
- `train/grad_norm`: pre-clip gradient norm (divergence early-warning).
- `train/amp_scale`: `GradScaler` scale; staircase-down during a loss spike indicates AMP overflow.
- Non-finite loss → skip optimizer step and priority update (prevents NaN propagation into weights).
- `ReplayBuffer` sanitizes non-finite priorities on both write and read.

### Multi-game spatial alignment
Games padded to 8×8 via `_pad_to_size()` (zero-padding). Bilinear interpolation
was rejected — produces fractional values that distort discrete board semantics.
See `design.md` for full list of alternatives considered.

### Value/reward representation
Categorical distribution over support `{-K,...,K}` (`K=10` value, `K=1` reward).
Targets transformed via `h(x) = sign(x)(sqrt(|x|+1)-1) + 0.001x`. Cross-entropy loss.
Value loss weighted at `0.25`.

### MCTS
- MinMaxStats normalizes Q-values to `[0,1]` per search
- PUCT: `pb_c = log((N + 19652 + 1) / 19652) + 1.25`
- Value score includes reward: `reward + discount * child.value`
- Dynamics hidden state gradients scaled by `0.5` via hook during training unroll
- `BatchedMCTS.run_batch()` runs N games in parallel with one batched forward pass per simulation step

### Reanalyze
Re-runs MCTS on stored positions with the current network to freshen policy and value targets before training. Implemented synchronously in the training loop.
- Uniform sampling from buffer (not PER) — avoids double-biasing; matches muzero-general `force_uniform=True`
- No Dirichlet noise — clean policy estimates, not exploratory ones (lightzero `reanalyze_noise=False` default)
- Updates both `policies` and `root_values` in GameHistory in-place; value TD bootstrapping in `make_target` then uses fresher estimates automatically
- Configured via `reanalyze_interval` (steps between calls) and `reanalyze_batch_size` (games per call)
- Tictactoe default: `interval=100, batch=100` — matches self-play cadence of 100 games/100 steps (1:1 refresh ratio)

### PER (Prioritized Experience Replay)
- Priorities initialized to max current priority for new games
- `sample_batch` returns IS weights; losses are per-sample and weighted before `.mean()`
- Priorities updated from root value TD errors after each train step
- Controlled by `per_alpha` (priority exponent), `per_beta_init` (IS weight, anneals to 1.0), `per_epsilon` (floor)

### LR Scheduling
Piecewise constant decay via `MultiStepLR`. Milestones specified as fractions of `training_steps` in `lr_decay_milestones`. Default: decay at 50% and 75% of training by factor `lr_decay_factor=0.1`.

### Temperature Scheduling
`temperature_schedule` in config: list of `(step_fraction, temperature)` pairs. Applied at the start of each self-play batch to set `temperature_init` for that batch.

## User Preferences
- Concise responses, no emojis
- Ask before pushing to GitHub
