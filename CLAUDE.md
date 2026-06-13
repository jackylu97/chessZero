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

### Draw-basin pipeline bugs — CONFIRMED 2026-06-13 (see `bug_hunt_2026_06_13.md`)
Three verified self-play data-pipeline defects, all draw-basin contributors,
none yet fixed (pending review — no code changed). Full evidence in the doc.
- **(HIGH) `select_action_gpu` temperature per-slot, not per-action** (`src/mcts/tensor_mcts.py:88-121`).
  Duplicate root-sample slots are softmax'd BEFORE being summed, so at T≠1 the
  GPU production path picks wrong moves (56/64 games) and writes de-sharpened /
  near-uniform policy training targets. Numpy reference path is correct; tests
  miss it.
- **(HIGH) Policy targets use the sampling temperature, not raw visit counts**
  (`src/mcts/mcts.py:678`, all `self_play.py` loops). At `temperature_final=0.1`
  every target past ply 30 is ~one-hot(argmax). Docstring + reanalyze (T=1.0)
  prove intent was `N(a)/ΣN(a)`. Shares a fix with the above: build the target
  from deduplicated raw visit fractions, decoupled from sampling temperature.
- **(HIGH) GPU engine never detects insufficient material** (`src/games/chess_gpu.py:1708`).
  K-vs-K / KB-vs-K / KN-vs-K run to the 75-move clock (~150-230 extra plies)
  instead of drawing immediately as CPU/python-chess does, flooding the buffer
  with `[0,1,0]` draw targets and desyncing self-play vs warmstart/eval labels.
- **(MEDIUM) Mutating terminal observation** (`chess_gpu.py:1636-1722`): board
  fields aren't frozen for done games, so the end-of-batch `final_obs` is corrupt
  for every game that finished before the batch's last — poisoning the terminal
  (decisive-labeled) value sample of those games.

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
