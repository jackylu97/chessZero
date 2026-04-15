# Design Decisions

A running record of key implementation choices, the reasoning behind them, and known deferred improvements.

---

## Observation Encoding

### Multi-plane board representation
Each game encodes the board as multiple binary planes rather than a single plane with values `{-1, 0, 1}`.

**Example (tic-tac-toe):** 3 planes — player 1 pieces, player 2 pieces, current player (all-0 or all-1).

**Why:**
- A single plane conflates three distinct concepts: piece presence, ownership, and turn
- `0` is ambiguous — indistinguishable from zero-padding added by the network
- Separate binary planes give the CNN clean, unambiguous signals

---

## Action Encoding

### Scalar broadcast (current)
Actions are encoded as `action / action_space_size`, broadcast to a `(B, 1, H, W)` constant plane. Matches muzero-general CNN convention.

**Known limitation (multi-game):** Action index 2 in tic-tac-toe and action index 2 in chess produce identical scalar values. The shared dynamics backbone cannot distinguish them from the action signal alone — it must rely on the incoming hidden state for game context.

### Deferred: per-game action embeddings
`nn.Embedding(action_space_size, embed_dim)` per game, broadcast spatially. Would solve the cross-game collision problem and give actions learnable structure (similar moves cluster in embedding space). Chess embedding table costs ~300K params.

**Better alternative for small-action games (tictactoe, connect4):** one-hot spatial plane — a single channel with `1.0` at the action's board cell. Directly communicates *where* the action was to the convolutional backbone.

Revisit if multi-game dynamics training is unstable.

### Chess action space (4672 = 8×8×73)
Follows the AlphaZero encoding. Each source square has 73 possible move planes:
- 56 queen-style moves (7 distances × 8 compass directions)
- 8 knight moves
- 9 underpromotions (3 pieces × 3 columns)

Illegal moves are masked to −∞ before softmax in MCTS `_expand`, so only ~30-40 legal moves per position carry non-zero probability.

---

## Multi-game Spatial Alignment

**Problem:** Games have different board sizes (chess 8×8, connect4 6×7, tic-tac-toe 3×3) but share a common latent spatial size (8×8).

**Decision: zero-padding via `_pad_to_size()` (`src/model/muzero_net.py`)**

Bilinear interpolation was rejected — it produces fractional blended values that distort the discrete piece-identity semantics of board games (e.g. a Connect4 cell becomes a weighted average of neighbouring cells). Zero-padding preserves every original cell exactly; the network learns to ignore the constant-zero border.

**Alternatives considered:**
1. **Per-task encoder → flat latent vector** (ScaleZero 2025, current SOTA): each game's encoder maps its board to a fixed-size 1D vector; shared dynamics operates on vectors rather than spatial tensors. Avoids alignment entirely but sacrifices spatial inductive bias.
2. **Native 8×8 for all games**: extend connect4 to 8 columns, play tic-tac-toe on 8×8. Cleanest for the shared backbone but changes game semantics and requires rewriting game logic.
3. **ViT / GNN backbone**: patch tokenisation or graph message-passing handles variable board sizes naturally. Best choice if cross-size generalisation within a game family is a goal.

Revisit if multi-game training underperforms single-game baselines.

---

## Value and Reward Representation

**Decision: categorical distribution over a discrete support, not scalar regression.**

Values and rewards are predicted as distributions over `{-K, ..., 0, ..., K}` (default `K=10` for value, `K=1` for reward). Targets are transformed via `h(x) = sign(x)(sqrt(|x|+1) - 1) + 0.001x` before being projected onto the support. Loss is cross-entropy.

**Why:**
- More numerically stable than MSE for large value ranges
- Matches the MuZero paper and reference implementations
- Value loss is downweighted by `0.25` (following the Reanalyze paper) to prevent value targets from dominating policy learning

---

## Multi-game Architecture

### Shared backbone + per-game heads
```
per-game input projection → shared ResNet backbone → per-game policy/value/reward heads
```

Each game has its own Conv2d projection (maps game-specific channels to `hidden_planes`) and its own output heads (different action space sizes). The residual backbone is fully shared — this is where cross-game knowledge transfer is intended to happen.

### Game ID embeddings (defined, not yet wired up)
`self.game_embeddings` exists in `MultiGameMuZeroNetwork` but is not concatenated to the hidden state. Wiring it up would give the shared backbone an explicit "which game am I playing" signal, which may help if training is unstable or if the action encoding collision causes interference.

### Deferred: per-game prediction residual blocks
muzero-general's `PredictionNetwork` runs additional residual blocks on the hidden state before the policy/value heads, giving each head its own private processing capacity on top of the shared representation. Our implementation feeds the hidden state directly into the heads.

This is intentionally skipped for single-game training — the backbone's 8 blocks are deep enough that the heads don't need extra capacity. However, **revisit for multi-game training**: when the shared backbone must serve all games simultaneously, per-game prediction blocks would let each game's heads do additional game-specific computation rather than reading directly off the shared representation. This could meaningfully improve per-game prediction quality without affecting the shared backbone.

---

## MCTS

### PUCT formula
Dynamic exploration constant following the MuZero paper:
```
pb_c = log((N + 19652 + 1) / 19652) + 1.25
UCB  = pb_c * prior * sqrt(parent_visits) / (1 + child_visits)  +  normalize(reward + discount * value)
```
Q-values are normalized to `[0, 1]` per search via `MinMaxStats`.

### Known issue: illegal moves at leaf nodes
At the root, legal actions come from the game engine and are correctly masked. At leaf nodes, the game engine is not available (search is in latent space), so all `action_space_size` actions are treated as legal. This causes:
1. The dynamics network is trained only on legal transitions but queried on illegal ones at inference (distribution mismatch)
2. MCTS visit counts used as policy targets include illegal move exploration, so the policy loss inadvertently encourages probability mass on illegal actions

**Principled fix:** store per-step legal action masks in `GameHistory` during self-play and apply them during the training unroll. Currently accepted as a known approximation consistent with the MuZero literature.

### Gradient scaling on dynamics hidden state
A `register_hook` scales gradients by `0.5` on the hidden state output of the dynamics network during training unroll. This prevents gradients from K unroll steps from overwhelming gradients through the representation network.
