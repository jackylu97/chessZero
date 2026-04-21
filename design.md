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

### Known issue: EfficientZero consistency loss not wired into `MultiGameMuZeroNetwork`
`MuZeroNetwork` has a `project()` method (projection + predictor) used by the EfficientZero consistency loss. `MultiGameMuZeroNetwork` does not — it was missed when the SimSiam heads were added. Trainer gates the consistency branch on `config.use_consistency_loss`, so setting it with a multi-game network will blow up on `self.network.project(...)`.

**Options when we revisit multi-game:**
- Shared projection/predictor operating on the shared-backbone latent (cheapest; matches current shared-dynamics philosophy).
- Per-game projection/predictor (better if per-game prediction blocks land too — lets each game learn its own world-model similarity metric).

**Status:** Deferred. Chess (the cold-start motivation for EZ) is single-game, and there's no path through `MuZeroTrainer` that can combine them today.

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

### Deferred: decisive-game prioritization for MuZero cold start

When most self-play games end in zero-reward draws (e.g. chess early in training, or any environment with sparse terminal reward), the value/reward heads are starved of non-zero labels. Current PER priorities are per-game from TD error, but within a game positions are sampled uniformly — so a long draw still contributes many zero-label samples for every decisive sample.

**Candidate fix:** in `ReplayBuffer.save_game`, weight the initial priority by `|game_outcome|` (and possibly `1/game_length`) so decisive games — especially short mates — are over-sampled during the bootstrap phase. Phase the weighting out as training matures to avoid biasing toward tactical-only positions.

Related lineage: Hindsight Experience Replay (Andrychowicz 2017) and "success prioritization" techniques in sparse-reward RL. Not standard in the AlphaZero/MuZero line, but a natural extension to PER when the value head is cold-starting. Kept in reserve as a Plan C fix for the chess draw basin — deferred because root-cause fixes (ply cap, MC returns) address the starvation more directly and should be tried first.

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

---

## Chess Self-Play — Catastrophic Leaf Expansion (and Fix)

### Symptom
Initial chess self-play stalled indefinitely. Overnight run showed 0 games completed across 9+ hours of wall-clock, with 103% CPU on one core, 12% GPU utilization, and ~10GB of RSS memory accumulating. `ctrl-C` landed inside `MCTS._expand` creating child nodes.

### Root cause
The Python MCTS expands every leaf with **all 4672 chess actions** (`mcts.py:272`). Two consequences:

1. **Expansion cost**: per simulation across 64 parallel games, ~300K `MCTSNode` objects are created. At 50 simulations per move, that's ~15M Python object allocations per move advancement. With Python's per-object overhead, the first move of the first batch never completed.
2. **Selection cost**: `_select_child` iterates `node.children.items()` to find the max UCB. With 4672 children, every selection step within the MCTS tree is also O(4672) in Python.

This was **known issue #1** in `CLAUDE.md` — tolerable for tictactoe (9 actions) and connect4 (7 actions), catastrophic at chess's 4672.

### How AlphaZero/MuZero avoid it
**AlphaZero does not have this problem** — it uses the real chess engine at every tree node, so it only ever creates ~35 children (the legal moves). MuZero introduces the problem because leaves live in learned latent space with no access to game rules, so legality can't be known.

**DeepMind's MuZero paper handles 4672-way expansion by (a) implementing MCTS in C++ with preallocated node pools, making expansion cost nanoseconds per node, (b) vectorizing UCB selection as a single AVX `argmax` over a flat prior array, and (c) relying on the network learning to put near-zero prior on illegal moves after early training — in practice >99% of the 4672 children get ignored in selection because the prior is negligible.** None of these translate directly to Python.

The "proper" Python-compatible fix is **Sampled MuZero** (Hubert et al. 2021): sample K actions from the prior distribution at each node and use an importance-weighted MCTS target. Listed in the "Scaling to Chess" section above as a deferred improvement.

### Implemented fix: top-K leaf expansion
Added `leaf_top_k` config field. When set, `_expand` keeps only the top-K children by prior using `torch.topk` instead of materializing all `action_space_size` children.

- **At the root**: unchanged. Legal actions come from `python-chess`, all expanded (~35 children). Visit-count policy target remains faithful.
- **At latent leaves**: only the top-K priors become children. Chess config uses `leaf_top_k=50`, roughly 50× the average number of legal chess moves.
- **Other games unaffected**: tictactoe/connect4 configs leave `leaf_top_k=None`, behavior identical.

Expected speedup: ~90× on both expansion and per-node UCB selection. Chess self-play should now complete an initial batch in ~5-15 minutes instead of hanging forever.

### Why top-K is a defensible baseline (not just a hack)
- Chess's ~35 legal moves almost always fall inside the network's top-50 priors after a few thousand training steps
- Actions outside top-K still get their prior computed (since softmax happens pre-topk) and contribute to the statistics via the Dirichlet root noise; they just can't be selected past a leaf
- MCTS literature (muzero-general, similar Python reimplementations) uses equivalent heuristics for large action spaces — this is the standard baseline

### Why top-K is still wrong in principle
- Bottom-ranked actions can never be explored at a leaf — if the network's prior is badly miscalibrated for a particular position, MCTS can't correct it via visitation. Sampled MuZero fixes this by giving every action positive sampling probability.
- Visit counts at the root are only over top-K (at leaves), so the policy target is mildly biased toward concentrated priors.

### Secondary fix: move cap
Added `max_plies = 400` to `ChessGame`. Forces draw (`winner=0`, `reward=0.0`) on games that run past 400 plies. With `leaf_top_k` in place, games should terminate via normal chess rules long before the cap; this stays as a safety net against pathological repetition loops that evade python-chess's 5-fold repetition check (which requires exact position matches including castling rights, en passant, and side-to-move).

AlphaZero used a 512-ply cap for the same reason. 400 is a conservative choice — longest recorded master game is ~540 plies, but the longest meaningful MuZero self-play game is much shorter.

---

## Normalization: BatchNorm → LayerNorm (training stability)

### Symptom
Chess training run diverged at step 2650 of a 25k-step run: policy loss jumped from ~4.2 → 6.2, value and reward losses spiked simultaneously. Loss never fully recovered over the remaining 400 steps before the run crashed on NaN at step 3048.

### Diagnosis via checkpoint diff
Compared per-parameter weight deltas across `checkpoint_2500 → checkpoint_3000` (contains the spike) against `checkpoint_2000 → checkpoint_2500` (a baseline 500-step window with no spike). The 15 largest-ratio parameters were all BatchNorm `running_mean` / `running_var`, not trainable weights:

- `dynamics.bn_in.running_var`: 0.22 → 16.94 (76× baseline delta)
- `dynamics.blocks.0.bn1.running_var`: 0.31 → 18.35 (60×)
- `dynamics.blocks.1.bn1.running_var`: 0.58 → 26.75 (46×)
- `prediction.value_head.1.running_mean`: 5.23 → 52.95 (10×)

Trainable weights in the spike window moved only 3–5× baseline — consistent with normal gradient descent adapting to shifted inputs. Median layer delta ratio was 0.95 (most layers looked normal). The anomaly was strictly in BN running stats, concentrated in the dynamics stack.

### Root cause
MuZero's dynamics network is applied **recurrently** during training unroll (K=5 times per sample, each time consuming its own output). BatchNorm's `running_var` / `running_mean` are EMAs that assume IID batch statistics, but unrolled hidden states at different timesteps are causally correlated — not IID. One outlier batch pulled the running_var on dynamics BN layers by 30–80×, and the corrupted running stats then distorted every subsequent forward pass. Because the net keeps ingesting hidden states through corrupted BN at each unroll step, the stats can't self-correct within a few iterations — hence the 400-step slow-recovery-never-completes pattern observed in the run.

### Fix: replace BatchNorm with LayerNorm throughout
- `src/model/utils.py`: added `norm_layer(num_channels, spatial_shape) → nn.LayerNorm([C, H, W])`. `ResidualBlock` now takes a required `spatial_shape` argument (needed because LayerNorm's affine params are per-(C,H,W), not per-channel as in BN/GN).
- `src/model/muzero_net.py`: every `nn.BatchNorm2d(C)` call replaced with `norm_layer(C, (H, W))` in `MuZeroNetwork` (representation projection, dynamics bn_in, prediction policy/value heads) and `MultiGameMuZeroNetwork`.

LayerNorm has no running statistics, so the recurrent-drift failure mode disappears entirely.

### Reference check
- **muzero-general**: uses BatchNorm2d throughout (susceptible to same failure).
- **LightZero**: `norm_type` is configurable (`'BN'` or `'LN'`); **`'LN'` is the default**. Uses `nn.LayerNorm([C, H, W])` — we match exactly.
- **DeepMind's updated MuZero architecture** (post-2020 papers): uses LayerNorm.

Residual block ordering is **post-activation** (`Conv → Norm → ReLU → Conv → Norm → add → ReLU`), matching both muzero-general and LightZero's ResBlock. An earlier attempt to also adopt pre-activation (ResNet v2) was reverted — the claim came from an unverified web-search snippet, and the two actively-maintained reference implementations both use post-activation.

### Param count
LayerNorm's per-(C,H,W) affine adds parameters relative to BN's per-C. Chess net: 39.06M → 41.19M (+5%). Well within budget.

### Monitoring added to catch future divergences early
- `train/grad_norm`: pre-clip gradient norm. Sharp spike = bad gradient event.
- `train/amp_scale`: `GradScaler` current scale. Staircase-down pattern during a loss spike = AMP overflow. Stable = overflow is not the cause.
- `trainer._train_step` skips the optimizer step and priority update on non-finite loss.
- `ReplayBuffer.sample_batch` / `update_priorities` sanitize non-finite priorities defensively.

### Breaking change
The norm-layer swap changes parameter names and shapes, so checkpoints trained under the BN architecture cannot be loaded into the new one. A fresh training run is the only path forward.
