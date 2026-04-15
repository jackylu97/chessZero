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
- `src/mcts/mcts.py` — MCTS with MinMaxStats, PUCT
- `src/training/trainer.py` — Single-game training loop
- `src/training/multi_game_trainer.py` — Multi-game round-robin training
- `src/config.py` — Per-game hyperparameters
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/train_multi.py`
- `design.md` — Full record of design decisions and deferred improvements

## Known Issues / Deferred Work

### 1. Illegal moves at MCTS leaf nodes (`src/mcts/mcts.py:127`)
At the root, legal actions from the game engine are correctly masked. At leaf nodes
(latent space), all `action_space_size` actions are treated as legal. Two consequences:
- Dynamics network trained only on legal transitions but queried on illegal ones (distribution mismatch)
- Policy targets (MCTS visit counts) include illegal move exploration

**Fix:** Store per-step legal action masks in `GameHistory` during self-play; apply during training unroll. Currently accepted approximation consistent with MuZero literature.

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

## User Preferences
- Concise responses, no emojis
- Ask before pushing to GitHub
