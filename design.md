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

### Observation perspective (chess)
`to_tensor` encodes from white's fixed perspective (row 0 = rank 1, white pieces in planes 0–5, black in 6–11). The turn plane (plane 17) tells the network whose move it is.

**Why not perspective-flip like AlphaZero?**
Proper perspective flipping requires flipping both the observation AND the action indices — `legal_actions()` would return indices in the flipped coordinate system, and `_action_to_move()` would un-flip before applying. Flipping only the observation while keeping action indices unflipped creates a spatial mismatch: CNN features computed on a flipped board don't align with the unflipped action space, making the mapping harder to learn than the consistent-but-asymmetric alternative.

**Deferred:** Implement full perspective flipping (observation + action index flip) as a future improvement. The network will need more capacity to represent both colors independently, but for a 256-plane, 16-block network this is not a meaningful cost.

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

---

## Reanalyze

**Decision: synchronous reanalyze in the training loop, uniform buffer sampling.**

Re-runs MCTS on stored game positions using the current network to replace stale self-play targets (policies and root values) with fresher estimates. Based on the MuZero Reanalyze paper (Schrittwieser et al. 2021).

**Sampling: uniform, not PER-weighted.**
PER already concentrates training on high-error positions. Applying PER to reanalyze sampling too would over-concentrate refreshes on those same games, leaving most of the buffer stale. muzero-general explicitly uses `force_uniform=True` for this reason.

**No Dirichlet noise during reanalyze.**
Self-play uses noise for exploration; reanalyze is trying to produce clean, accurate policy estimates from the current network. lightzero's `reanalyze_noise` flag defaults to False.

**Updates both policies and root_values.**
muzero-general stores `reanalysed_predicted_root_values` separately alongside child visit counts. We update in-place since `make_target`'s TD bootstrap reads `root_values` directly — updating them means future training samples from the same game automatically use the fresher value estimates without any additional bookkeeping.

**Synchronous vs async.**
muzero-general runs reanalyze as a continuous Ray background worker racing against self-play. lightzero uses a `reanalyze_ratio` to trigger it on a fraction of training steps. We use a fixed `reanalyze_interval` (simpler, no Ray dependency). The key metric is the refresh ratio: reanalyzed positions per step vs new positions added per step. Tictactoe is configured at 1:1 (`interval=100, batch_size=100` matches `self_play_interval=100, num_self_play_games=100`).

**Known gap vs reference repos.**
Neither muzero-general nor lightzero uses a separate target network for reanalyze in their default configs (target networks are listed as optional/experimental). We follow the same approach — reanalyze uses the live network weights. If training becomes unstable (reanalyze targets chasing the training targets in a feedback loop), a lagging target network (EMA of weights) would be the fix.

---

## BatchedMCTS

**Decision: batch N games' MCTS simulations into a single forward pass per simulation step.**

`BatchedMCTS.run_batch()` takes N observations and runs all simulations in lockstep. At each simulation step: select phase is serial (pure CPU, per game), then all N parent hidden states are stacked into one batch for a single `recurrent_inference` call. For small models on GPU, kernel launch overhead dominates at batch size 1 — batching amortizes this.

Used by both parallel self-play (`play_games_parallel`) and reanalyze (`_reanalyze`).

**Limitation:** games in a batch must all run the same number of simulations. If games end mid-batch (in parallel self-play), they drop out of the active list; games that finish earlier don't waste compute. For reanalyze, all positions always run the full simulation count.

---

## Prioritized Experience Replay (PER)

**Decision: PER with IS correction, alpha=0.6, beta annealing from 0.4→1.0.**

Standard PER (Schaul et al. 2015). Priorities initialized to max current priority for new games (ensures each game is sampled at least once before being de-prioritized). Updated from root value TD errors after each training step.

Loss functions return per-sample tensors (not reduced means). IS weights are applied before `.mean()` so that high-priority samples don't dominate gradients uncorrected.

**alpha=0, beta=1** degrades gracefully to uniform sampling — useful for ablations.

---

## LR Scheduling and Temperature Scheduling

**LR:** Piecewise constant decay via `MultiStepLR`. Milestones as fractions of `training_steps` (default: 0.5, 0.75) with `lr_decay_factor=0.1`. This matches the schedule used in muzero-general for longer runs.

**Temperature:** `temperature_schedule` is a list of `(step_fraction, temperature)` pairs that lower `temperature_init` as training matures. Applied at the start of each self-play batch. Within a game, `temperature_drop_step` still switches to `temperature_final` after a fixed move count (unchanged).

---

## Scaling to Chess — Training Speed Strategies

Chess has two properties that make naive MuZero slow: a 4672-action space and games that are 40-80 moves long. The following approaches are candidates for Phase 1 chess training, ordered roughly by expected impact.

### 1. Sampled MuZero (highest priority for chess)
Instead of expanding all 4672 actions at each MCTS node, sample a small subset (20-50 actions) from the policy prior. Since only ~30 legal moves exist per chess position, most of the 4672 actions are illegal anyway — full expansion wastes compute evaluating them. Sampled MuZero makes each simulation dramatically cheaper and is effectively mandatory for large action spaces.

**Paper:** "Learning and Planning in Complex Action Spaces" (Hubert et al. 2021)

### 2. EfficientZero consistency loss
Adds a self-supervised auxiliary loss: the latent state produced by rolling the dynamics network forward (`g(h_t, a_t)`) should match the latent state produced by running the representation network on the actual next observation (`h(o_{t+1})`). This keeps the dynamics network grounded in reality rather than drifting in latent space, dramatically improving sample efficiency.

EfficientZero achieved AlphaZero-level Atari performance with ~200x less data than MuZero. The consistency loss is the core contribution and is largely orthogonal to other improvements — it can be added on top of the existing architecture.

**Paper:** "Mastering Atari Games with Limited Data" (Ye et al. 2021)

### 3. Curriculum transfer via multi-game training (Phase 2 overlap)
Train tictactoe → connect4 → chess sequentially, transferring the shared backbone. The representation and dynamics networks learn general game reasoning on simpler games before chess, giving a much better initialization than random weights. This is directly enabled by the existing `MultiGameMuZeroNetwork` architecture and is essentially free to implement given Phase 2 is already planned.

### 4. Higher train/self-play ratio
Currently 1:1 for tictactoe (100 training steps per 100 self-play games). Increasing to 5:1 or 10:1 means the network improves more before generating new self-play games, so each batch is higher quality. Cheaper than generating more games and easy to tune via `self_play_interval` and `num_self_play_games`.

### 5. Gumbel MuZero
Replaces PUCT + visit-count policy targets with a statistically more efficient action selection scheme based on Gumbel-top-k sampling. Gets equivalent policy quality with ~10x fewer simulations, which directly reduces the cost of both self-play and reanalyze.

**Paper:** "Policy Improvement by Planning with Gumbel" (Danihelka et al. 2022)

### Recommended combination for chess
Sampled MuZero + EfficientZero consistency loss + curriculum transfer from connect4. Sampled MuZero is close to required given the action space; the consistency loss is a relatively contained addition to the training loop; curriculum transfer is free given the existing architecture.
