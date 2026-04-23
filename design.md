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

---

## Deferred: Compact GameHistory Encoding

### Problem
Stockfish warmstart shards are ~3.3 MB/game → a 32k-game pool weighs ~110 GB. Chess checkpoints + buffers already dominate disk; multi-hundred-GB teacher pools will bottleneck future runs.

### Where the bytes go
Per game (~80 plies):
- `observations` (80× tensors of 20+ planes × 8×8 × fp32): **~400 KB** — mostly zeros, fully reconstructable from the action sequence.
- `policies` (80× 4672-wide fp32 arrays): **~1.4 MB** — Stockfish one-hots are 99.98% zeros; only the action index matters.
- `legal_actions_list` (80× lists of ~35 ints): ~28 KB — reconstructable from position.
- Everything else (`actions`, `rewards`, `root_values`, `external_values`, `game_outcome`): <2 KB — keep.

Roughly **95% of a shard's bytes are redundant.**

### Three compaction levels

**Level 1 — zstd-compressed pickle (10-line change, ~5-10× smaller).**
Wrap `open()` in `zstandard.ZstdCompressor().stream_writer(...)` on save, mirror on load. Observation tensors and one-hot policies compress extremely well. `110 GB → ~15-25 GB` with zero format changes. Useful as an interim knob.

**Level 2 — drop redundant fields, reconstruct on load (recommended).**
- Don't store `observations` — replay actions through `ChessGame` during shard load.
- Don't store `legal_actions_list` — recompute from the reconstructed position.
- Store `policies` as a single int (action index) per Stockfish position; expand to one-hot at load time. For self-play soft-target policies, use a sparse `(index, weight)` list.

Final size: ~2-4 KB/game. `110 GB → ~100 MB`. Load cost ~10 ms/game via python-chess replay; negligible vs training time and easily hidden in lazy-decode per shard during injection.

**Level 3 — custom binary format (overkill for current scale).**
Fixed-width records, `uint8`-packed planes if obs are kept, FEN strings as alternative. Gets to ~1-2 KB/game. Skip unless we hit a scale where Level 2 isn't enough.

### Format versioning
Use the same pattern `.buf` save already uses (v1 legacy → v2 streamed header). Shard header carries `format_version`; load path dispatches. Old shards remain loadable.

### Decision
**Plan: land Level 2 before the next major run.** Skip Level 1; the marginal engineering to add a format-version dispatcher is minimal and the payoff is ~40-100× smaller pools vs ~5-10× for compression-only. Don't regenerate the current 32k-game Stockfish pool for this — that pool is known-good under the existing format and stays usable via the legacy load path.

### Urgency bump (2026-04-23)
Run `2026_04_22_0002` OOM-killed ~step 36.3k of 150k when self-play games displaced shorter Stockfish games in the 5000-slot buffer. Self-play games average 277 plies × 25 KB/ply ≈ **6.7 MB/game** in RAM; Stockfish games average 152 plies ≈ **3.0 MB/game**. Full self-play turnover projected 5000 × 6.7 = 33.6 GB buffer — larger than the 32 GB host. Short-term fix: `replay_buffer_size` dropped to 2500 (peak ~17 GB).

### Landed 2026-04-23 (Steps 1, 3, 5, 6 + generator; all tests passing)
Compact on-disk encoding shipped. Measured on a 200-game slice of an existing Stockfish shard:
- Legacy: 728 MB (3.5 MB/game). Compact: **0.69 MB** (3.4 KB/game). **1054× smaller.**
- Decode cost: 26 ms/game; 256-game injection batch decodes in <7 s.
- 16 new tests in `tests/test_compact_encoding.py` cover roundtrips (one-hot + sparse policies; tictactoe + chess), pickle, version dispatch (v1/v2/v3 buffers, legacy-list vs compact shards), and legal/obs reconstruction.

**Caveat on RAM:** in-memory buffer still holds full GameHistory. The `replay_buffer_size=2500` cap stays in place. Compact storage only helps disk/shard size today. Lazy in-memory compact storage (decode in `sample_batch`) is a follow-up — needs a decoded-game LRU cache to be tractable (naive decode on sample = 256 × 26 ms ≈ 7 s/batch).

### Implementation plan (concrete steps)

**Step 1 — new compact serialization (in-memory object unchanged)**
Add to `src/training/replay_buffer.py`:
- `GameHistory.to_compact_dict(game_cls) -> dict` — emits `{version: 2, game_name, actions, rewards, root_values, game_outcome, policies_compact, legal_actions_mode, external_values}` where `policies_compact` is either `(N,) int32` of argmax indices (when every entry is one-hot) or a sparse `[(idx, weight), ...]` list per step. `legal_actions_mode`: "recompute" (default) or "stored" (explicit list, for non-chess games).
- `GameHistory.from_compact_dict(d, game) -> GameHistory` — reconstructs `observations` by replaying `actions` through `game.reset()`/`game.step()`; reconstructs `legal_actions_list` via `game.legal_actions(state)` at each ply; expands `policies_compact` to full-width fp32 arrays.
- Dispatcher on load: peek `version` from header; v1 → existing path, v2 → compact decode.

**Step 2 — emit compact shards from generation**
`scripts/generate_stockfish_games.py`: switch `pickle.dump(shard, ...)` to write a header dict then one compact-dict per game. Stockfish policies are always one-hot, so `policies_compact` becomes a single `np.int32[N]`. Add a `--format-version` flag defaulting to 2 so the old path is one flag-flip away.

**Step 3 — compact `.buf` save/load in `ReplayBuffer`**
`ReplayBuffer.save`: emit header `{version: 3, game_name, n_records, total_games}` then one `(compact_dict, priority)` per game. `ReplayBuffer.load`: dispatch on header version — v2 → legacy full GameHistory path, v3 → compact path with decoder. Lazy decode optional (decode on `sample_batch` rather than on load) if we want to push in-memory footprint even lower; skip initially to keep the diff small.

**Step 4 — self-play path**
`src/training/self_play.py` already builds full GameHistory objects; no change needed to that path. Compaction happens at save time. (If we later want compact in-memory storage to fit 10k+ games in the buffer, we revisit — but Step 3 already gets us comfortably within the RAM budget.)

**Step 5 — viewer / debug tools**
`scripts/view_buffer_web.py` (and any other buffer-inspecting scripts): wrap load in the version dispatcher. Inspection tooling needs the full GameHistory, so it decodes on read.

**Step 6 — tests**
`tests/test_compact_encoding.py`:
- Roundtrip: original GameHistory → compact → decoded → assert every field used by `make_target` matches (observations via allclose, policies/rewards/values/actions/legal_actions bit-exact).
- Version dispatch: load v1 and v2 shards from the same API, both work.
- Chess-specific: replay-from-actions reconstructs the same observation tensor as was originally stored.

### Migration path
1. Keep the current `data/stockfish_injection/` pool untouched — it remains loadable via the v1 path for any future re-runs of the 2026_04_22_0002 config.
2. Land Steps 1, 3, 6 first (in-memory compaction + `.buf` compaction + tests). That alone kills the RAM OOM without requiring new Stockfish generation.
3. Land Step 2 when the next Stockfish pool is actually needed — bundles with the asymmetric-teacher generation plan.

### Affected code (scope preview)
- `src/training/replay_buffer.py` — `GameHistory` compact serialization + version dispatcher in `ReplayBuffer.save` / `.load` / `load_warmstart_games`.
- `scripts/generate_stockfish_games.py` — emit compact shards going forward.
- `scripts/view_buffer_web.py` — load path must handle both formats.
- `src/training/trainer.py` — no change (calls ReplayBuffer API; dispatcher is internal).
- `tests/test_compact_encoding.py` — new file with roundtrip + version-dispatch tests.

### Secondary: cap replay buffer by bytes, not game count
Even after compaction, a cleaner invariant is "buffer holds ≤ N GB" rather than "≤ K games". Games have 10× variance in length; game-count cap is a poor proxy. One-line change after compaction lands: track a running byte total in `save_game`/`pop`, evict until under cap.

---

## Deferred: Tensor-Native MCTS on GPU

### Problem
`BatchedMCTS.run_batch` is *partially* GPU-accelerated: network forwards (dynamics, prediction) are batched across games per simulation step, but tree structure — selection walk, PUCT scoring, expansion, backup, MinMaxStats — lives in Python lists/dicts on CPU. Between each network call, the GPU idles while CPU picks the next leaf.

At current batch_size=256 parallel games, this is fine (~79% GPU util). The ceiling comes into view when scaling parallelism further: Python tree-ops overhead grows super-linearly with B, so we can't easily push past ~512-1024 parallel games without diminishing returns.

### Where the time actually goes (rough breakdown per simulation step)
- Network forward (dynamics + prediction, batched): ~75-85%
- Tree traversal + PUCT + backup (Python): ~10-20%
- Host↔device sync + misc: ~5%

So moving tree ops to GPU saves ~10-20% of MCTS wall time at our current batch size — modest on its own. The real win is unlocking **much larger parallel self-play batches**, where GPU tensor ops scale sub-linearly with B while Python scales super-linearly.

### Design sketch (full-tensor MCTS)

Pre-allocate a fixed-shape per-batch tree buffer. With Sampled MuZero limiting branching to `sample_k=50`, memory is manageable even for chess:

- `hidden_states: (B, max_nodes, C_hid, H, W)`
- `visit_counts: (B, max_nodes, K)` where K = sample_k = 50
- `q_values, priors, rewards: (B, max_nodes, K)`
- `children_idx: (B, max_nodes, K)` — -1 for unexpanded slots
- `parent_idx, parent_action: (B, max_nodes)` — for backup path tracking
- `min_q, max_q: (B,)` — MinMaxStats per game

For B=256, max_nodes=200, K=50: ~10 MB per quantity × ~8 quantities ≈ 100 MB total. Fits trivially.

**Per-simulation ops, all vectorized across B:**
1. **Selection walk** — fixed-depth loop (up to `max_depth`): at each depth, gather children's N, Q, priors; compute PUCT; argmax → next node. Masked when a game's walk hits a leaf before max_depth.
2. **Expansion** — scatter the new `(hidden, reward, policy_logits, value)` into the next free slot; bump per-batch `next_free_slot` counter.
3. **Backup** — scatter-add the backed-up value along the walked-path-indices tensor.
4. **MinMaxStats** — elementwise min/max update on per-batch Q range.

No Python tree manipulation. No host↔device sync between simulations (single stream of GPU work from step 1 to step `num_simulations`).

### Reference implementations
- **[Mctx](https://github.com/google-deepmind/mctx)** (DeepMind, JAX) — reference for Gumbel MuZero paper. Native TPU, scales to 4096+ parallel sims per step. Closest in spirit to what we'd want.
- **[LightZero's `gumbel_muzero_mcts.py`](https://github.com/opendilab/LightZero)** (PyTorch + C++ tree) — closer to our stack but uses a C++ extension for the sparse tree ops, since PyTorch lacks JAX's `while_loop` primitive at equivalent performance.

### Gains and costs
**Wins:**
- ~10-20% MCTS speedup at current batch sizes.
- Unlocks 4-16× larger parallel self-play batches → proportional data-generation speedup.
- Reanalyze and eval get the same acceleration for free (both are just batched MCTS).
- Removes CPU-side MCTS as a forever-bottleneck for multi-GPU scaling.

**Costs:**
- **Major rewrite.** ~2-4 weeks of focused work with careful test parity against existing `BatchedMCTS` output (bit-for-bit visit counts and Q values on fixed seeds).
- **Asymmetric root vs non-root logic** — Gumbel MuZero's Sequential Halving at root + PUCT at non-root needs two well-isolated code paths inside the tensor MCTS.
- **Harder debugging** — "wrong visit count in row 247 of batch slot 31" vs inspecting a Python tree.
- **Framework choice.** JAX (via Mctx) is proven but brings a new dep. PyTorch-native needs either custom CUDA kernels or a C++ extension for the while-loop-heavy selection walk.

### Decision
**Defer until we actually need to scale parallel self-play past ~1024 games** or hit multi-GPU training. Current config (B=256, 200 sims, ~79% GPU util) doesn't justify the rewrite. Bundle the planning pass with the compact GameHistory rework (both are "pre-next-major-run" optimizations) and evaluate jointly: compact encoding cheap/high-value; tensor MCTS expensive/situational.

### When re-evaluating, answer these first
1. What's the new target batch size, and what does current BatchedMCTS look like at that batch (is it actually slow)?
2. JAX (bring Mctx directly) vs PyTorch-native (LightZero-style)? JAX is faster to land but adds a framework.
3. Is the `(B, max_nodes, K)` memory budget still comfortable with the new scale, or do we need sparse tree storage?

### Affected code (scope preview)
- `src/mcts/mcts.py` — complete rewrite of `BatchedMCTS` (serial `MCTS` can stay for debugging).
- `src/training/self_play.py`, `src/training/trainer.py` (`_reanalyze`, `_play_vs_random_batched`), `scripts/evaluate.py`, `scripts/play_web.py` — all call sites need to route through the new interface.
- `tests/` — new parity tests asserting tensor MCTS matches current `BatchedMCTS` on fixed seeds; existing `test_gumbel_*.py` continue to exercise the correctness invariants.

---

## Deferred: Phase-Dependent `value_loss_weight` (Warmstart vs Self-Play)

### Problem
`value_loss_weight = 0.25` is inherited from the MuZero paper, where it was calibrated for **noisy MCTS-bootstrapped** value targets during self-play. During our Stockfish warmstart phase, value targets come from `external_values` — deterministic depth-8 Stockfish evals — which deserve *more* supervision weight, not less.

The problem compounds with our 5-bin value encoding: the CE loss scales with `log(num_bins)`, so moving from the paper's 21-bin to our 5-bin support dropped the value head's gradient share by ~half (log(21)/log(5) ≈ 1.89). Net effect of both issues: value is contributing ~4-5% of total loss during Stockfish warmstart vs ~14-15% under the paper's defaults. Likely a contributor to the value-head mean-collapse plateau before step ~3750 and the subsequent slow convergence to MAE 0.16.

### Why 0.25 exists in the paper
It's a deliberately conservative weight for *self-play* targets, which are:
- **Bootstrapped** (TD or MC): depend on the network's own value/root estimates. Strong gradient → feedback loop: bad value → bad MCTS → bad targets → worse value.
- **Single-trajectory** (MC with `td_steps=-1`): ±1 or 0 per position based on one game outcome. High variance for positions far from terminal.
- **Policy-quality-dependent**: early-training self-play games produce wildly different outcomes for similar positions.

Chasing these noisy targets aggressively (high weight) destabilizes training. Damping with 0.25 is the paper's fix.

### Why warmstart deserves a different weight
Warmstart value targets have none of those properties:
- **Deterministic** (same position → same Stockfish eval).
- **Low variance** even across diverse duplicates in the pool.
- **Teacher >> current net** (depth-8 Stockfish is strictly better than an untrained MuZero).
- **No feedback loop** (targets don't depend on our network).

In this regime we *want* strong supervision. 1.0 is appropriate; arguably higher.

### Three implementation options

**A. Phase-switch on `pool_alive` (recommended).** Piggyback on the existing injection-pool gate:
```python
# inside _train_step
value_weight = (
    self.config.value_loss_weight_warmstart if self._injection_shards
    else self.config.value_loss_weight_selfplay
)
```
Hard switch at pool exhaustion. Minimal code, matches the rest of the gating structure. Downside: during the ~5-10k-step transition window after pool dies, the buffer still contains some Stockfish games that now get the lower self-play weight. Minor inefficiency; FIFO replacement will resolve it.

**B. Gradual anneal over a transition window.** Linearly interpolate from warmstart weight → self-play weight over N steps post-exhaustion. Adds one hyperparameter; smoother but more complexity.

**C. Per-sample weighting by `external_values` presence.** Cleanest handling of the buffer-mix phase:
```python
has_ext = torch.tensor([bool(g.external_values) for g in sampled_games])
per_sample_value_weight = torch.where(has_ext, 1.0, 0.25)
```
Correctly weights each sample regardless of buffer composition. Downside: mixes gradient scales within a batch (slight AMP stability risk), needs plumbing `has_external_values` into the batch dict.

### Related: the encoding-normalization question
An orthogonal fix would be to normalize CE by `log(num_bins)` inside `_value_loss` and `_reward_loss`, making loss magnitudes encoding-invariant. Pro: future encoding changes don't silently shift gradient allocation. Con: loss numbers would no longer be comparable to the published MuZero literature (everyone reports raw CE). If we ever switch encodings again, consider this as a one-time fix paired with option A/B/C.

### Decision
Recommended for the next major run:
- Add `value_loss_weight_warmstart: float = 1.0` and `value_loss_weight_selfplay: float = 0.25` to `MuZeroConfig`, deprecate the single `value_loss_weight` (or keep as a fallback).
- Wire Option A (`pool_alive`-gated switch) into `trainer._train_step`.
- Leave this run alone — mid-run loss-weight changes destabilize training, and the current run's value head has already made most of its achievable progress at MAE ~0.16.

### Affected code (scope preview)
- `src/config.py` — two new config fields.
- `src/training/trainer.py::_train_step` — replace `self.config.value_loss_weight` usage with the gated variant.
- `tests/test_stockfish_injection.py` — add a test asserting the weight switches at pool exhaustion.

---

## Deferred: Asymmetric-Teacher Stockfish Generation (next major run)

### Problem
Our current Stockfish pool (32k games, depth-8 vs depth-8 with MultiPV=3 softmax) produces **decisive games** (~92% W/L rate) but the per-position Stockfish evals skew strongly toward **zero** — most positions in a balanced depth-8-vs-depth-8 game are slightly unbalanced at best. This feeds the value head with mostly-near-zero targets, which contributes to the "predict 0 everywhere" mean-collapse regime we've observed. The user's live test of the trained model (step ~10k) confirmed the symptom: model hangs queens and misses mates — it hasn't been taught tactical cause-and-effect because neither side in its training data *was* making the mistakes that produce tactical consequences.

### The redesign: asymmetric teacher with strong labeler
Decouple **trajectory generation** from **label quality**:
- **Play** with mixed-depth engines (e.g. depth-8 vs depth-5). Creates decisive asymmetric games where the weak side makes principled tactical mistakes and the strong side converts them.
- **Label** every position at depth-8 (the "evaluator"). `external_values = depth-8 STM eval`; `policies = one-hot on depth-8's best move`.
- **Actions** (`actions[i]`) record the *actually played move*, which may be a depth-5 blunder. The unroll trajectory preserves the mistake → bad-position causality.

This is the classical off-policy RL "behavior policy ≠ target policy" pattern: data is generated by weaker play, supervision is from stronger analysis. `policies[i]` (target) and `actions[i]` (behavior) can diverge for weak-side positions — this is intended.

### What the model learns from an off-policy position
- **Value target** (before a weak-side blunder): "this position is +0.5 per depth-8"
- **Policy target** (at that position): "depth-8 would play move Y"
- **Actual trajectory**: depth-5 played blunder M ≠ Y, leading to position X+1 with depth-8-eval -0.8
- **Dynamics unroll learns**: "if action=M at X, you land in bad position X+1"
- **Net signal**: tactical cause-and-effect — "this move is a blunder, and here's what happens when you play it"

The current symmetric-depth pool is missing this exact signal, because neither side *makes* mistakes worth capitalizing on. Asymmetric-teacher generation reintroduces it deliberately.

### Why MultiPV is still required (even with asymmetric depth)
Fixed-depth Stockfish is deterministic. Asymmetric depth picks *different* moves per side but still produces *identical game sequences* across runs from the same starting position unless we inject randomness. We keep the current MultiPV=K softmax (τ=0.15) sampling mechanism for both sides' play; only the **labeling** analyse uses MultiPV=1 (single top line).

### Composition: 4 equal buckets across depth pairs

| Bucket | Pair | Share | What it teaches |
|---|---|---|---|
| 1 | depth-8 vs depth-5 | 25% | Tactical conversion of clear blunders |
| 2 | depth-8 vs depth-6 | 25% | Strategic-error punishment |
| 3 | depth-8 vs depth-7 | 25% | Decisive-but-close games |
| 4 | depth-8 vs depth-8 | 25% | Balanced-position precision (baseline, drawn-ish) |

Within each bucket, alternate colors 50/50 so depth-8 plays white half the time and black half. Total pool ~32k games (unchanged); 8k per bucket.

### Higher MultiPV-K: K=10 (if budget allows) or K=5 (pragmatic)
Softmax with τ=0.15 already zero-weights blunders regardless of K (a move 0.5 worse in eval gets weight `e^(-3.3) ≈ 0.04`). With depth-8 labels everywhere, *played moves* can safely wander without corrupting the training signal. So: more K = more trajectory diversity at zero label-quality cost.

- **K=10** (recommended if compute allows): ~3× current generation time (~17h for 32k games). Maximum trajectory novelty.
- **K=5** (pragmatic): ~1.7× current (~9h). Meaningful diversity gain over current K=3, much cheaper.

Pick K=10 if the overnight generation slot is available, otherwise K=5.

### Cost estimate
Per-ply analyse cost in "depth-8 MPV=1 equivalents" (Stockfish analyse cost scales ~exponentially with depth, ~linearly with MultiPV-K):

| Bucket | Per-ply cost at K=10 | At K=5 |
|---|---|---|
| 8v5 | ~6.1 | ~3.3 |
| 8v6 | ~6.8 | ~3.6 |
| 8v7 | ~8.0 | ~4.3 |
| 8v8 | ~10.0 | ~5.0 |
| **Average** | **7.7** | **4.0** |

Current symmetric-depth K=3 pipeline: ~3 units/ply. So average slowdown: K=10 → 2.6×, K=5 → 1.35×.

32k games currently takes ~5.5h on 4 workers → proposed timelines:
- K=10: ~14h
- K=5: ~7.4h

### Implementation sketch
Extensions to `scripts/generate_stockfish_games.py`:
- Flag `--asymmetric` turns on dual-engine mode.
- Flag `--play-depth-white N --play-depth-black M` for bucket-specific depth pairs (or a `--bucket` shortcut).
- Flag `--label-depth` (default 8) for the evaluator engine's depth.
- Flag `--multipv` bumped default to 10 (or 5) on the play side.
- Bucket rotation logic to alternate colors and sweep through `[(8,5), (8,6), (8,7), (8,8)]`.
- Two engine processes inside `generate_one_game`: one at `label_depth` (used every ply for labels), one at the current-side's `play_depth` (used for action selection). For the 8v8 bucket these collapse into a single MPV=K analyse since the play engine *is* the label engine.

Inside the per-ply loop:
1. `label_info = label_engine.analyse(board, Limit(depth=8), multipv=1)` → depth-8 top-1 move + value.
2. `play_info = play_engine.analyse(board, Limit(depth=play_depth), multipv=K)` → K candidate moves.
3. `played_move = softmax_sample_from_multipv(play_info, board.turn, τ, rng)`.
4. Record: `actions += played_move`, `policies += one_hot(label_top_move)`, `external_values += label_value_stm`, `root_values += label_value_stm`.

For the 8v8 bucket: skip step 1, reuse `play_info[0]` for labels (single analyse call).

### Why this fits naturally with existing infrastructure
- `GameHistory.external_values` field already exists and already bypasses `td_steps`/`game_outcome` value paths when populated.
- `ReplayBuffer.load_warmstart_games` / injection pipeline consumes the shards uniformly — doesn't care whether they're asymmetric or symmetric.
- The `policies[i] ≠ argmax(action-path)` case is already the default (policies are distributions, actions are integers); no trainer changes needed.

### Decision
**Land for the next major run.** Bundle with the other pre-next-run plans: compact GameHistory encoding (so the bigger/more-varied pool doesn't balloon disk further), phase-dependent value_loss_weight (so the clean teacher signal gets weighted correctly). Start K=5 as a first pass if compute is tight, bump to K=10 if the first pass shows the diversity is still the limiting factor.

Don't touch the current 32k-game symmetric pool — it's the baseline against which we compare. The new asymmetric pool is a separate artifact.

### Affected code (scope preview)
- `scripts/generate_stockfish_games.py` — add `--asymmetric`, `--play-depth-white/black`, `--label-depth`, bucket-rotation flags; dual-engine management inside `generate_one_game`; off-policy label recording.
- `scripts/run_sf_gen.sh` — parameterize bucket sweep; emit balanced game counts per bucket.
- `tests/test_stockfish_diversity.py` — sanity-check that asymmetric mode produces (a) decisive-rate >90%, (b) `policies ≠ one_hot(actions)` on >0% of weak-side positions (confirms off-policy labeling), (c) `external_values` spread confirms bimodal distribution (not clustered near 0).

---

## Deferred: Drop Gumbel MuZero, Use Classic PUCT for Chess

### Problem: the m=16 truncation structurally excludes moves chess needs
Plain Gumbel MuZero at root selects actions via Sequential Halving over the top-`gumbel_num_considered=16` actions by prior logit. At **inference** (`add_noise=False` → `use_gumbel_noise=False`), this is a *deterministic* truncation by raw policy logits (`src/mcts/mcts.py:494-497`). If the best move for a position ranks 17+ in the policy prior, **it's structurally unreachable** — no amount of search budget finds it because it never enters the tree.

### Chess branching-factor context makes m=16 too tight
| Phase | Typical legal moves |
|---|---|
| Opening (moves 1-8) | 20-30 |
| Middlegame (moves 20-40) | **30-40 (peak)** |
| Complex endgame | 30-45 |

Average across a game: **~35 legal moves**. Top-16 covers ~45% of typical middlegame legal moves. Tactical rescues and defensive-resource moves (quiet queen retreats, fianchetto-escapes, mid-ply captures of loose pieces) commonly rank 17-30 in an undertrained policy head.

### Why this is different from Go (the paper's test domain)
Gumbel MuZero's m=16 default was calibrated for Go, which has ~200 legal moves per position but **sharp policy priors** — the policy head learns hot-zone patterns quickly and the best move is almost always in the top-16. Chess has **narrower action spaces but broader priors**: a queen can attack the whole board, a rook can threaten mate from a far file, and "the right move" often doesn't pattern-match common openings. The m=16 default ports poorly.

### Inference-vs-training mismatch when you flip only one
If you keep Gumbel in training but switch to PUCT for inference (e.g. via `--no-gumbel` in `play_web.py` / `evaluate.py`), the policy head was trained toward π' = softmax(logits + σ(completedQ)) targets — which PUCT's visit-count-based selection doesn't directly consume. The prior's calibration may be sub-optimal for PUCT. Not catastrophic but a known alignment mismatch.

### Four options
| Option | What it does | Cost | Risk |
|---|---|---|---|
| **A** | Bump `num_simulations=400` + `gumbel_num_considered=32` together | 2× inference time everywhere | Same Gumbel bet, more coverage |
| **B** | Bump `gumbel_num_considered=32` at fixed `num_simulations=200` | None (just thinner per-candidate visits) | Could be *worse* — thin Q estimates lead Sequential Halving to cut real best candidates |
| **C** | Train Gumbel, infer PUCT (use the `--no-gumbel` override) | None | Train/infer target mismatch; PUCT reads poorly-calibrated priors |
| **D** | **Drop Gumbel entirely, use classic PUCT + Dirichlet** | Config flip, no compute delta | Lose Gumbel's theoretical low-sim-budget advantage; match the MuZero paper's own chess setup |

### Decision (pragmatic, given compute constraints)
**Default Option D for next major chess run: drop Gumbel, use classic PUCT.**

Reasoning:
- DeepMind's MuZero paper used PUCT for chess/shogi/go — proven setup for this domain.
- Gumbel's claimed advantage ("better at small sim budgets") was validated on Go with sharp priors. We're in the opposite regime: narrower actions, broader priors.
- m=16 truncation is a structural bottleneck for tactical positions that can't be worked around with more training.
- **At our compute scale, the noise floor on experiments is high enough that modest algorithmic deltas are practically indistinguishable.** Algorithmic A/B's would need multiple full training runs to reach statistical significance — we can afford maybe 2-3 of those in total. Better to spend those budget slots on interventions with higher a-priori expected value (asymmetric-teacher data, compact encoding, phase-dependent loss weights) than on PUCT-vs-Gumbel comparison.
- Classic PUCT at the same num_simulations=200 has no structural action truncation and matches the proven board-game baseline — lower-variance choice.

### What to do with the Gumbel code
Keep it. It's tested (52 tests passing), lives cleanly behind the `use_gumbel` config flag, and costs nothing to retain. Set `use_gumbel=False` in the chess preset; keep the flag-gated path for potential future use when we have the compute to run a real A/B.

### When to revisit this decision
- If we accumulate enough compute to run multiple full chess training runs (say, 4+ runs fit within an experiment budget), then a proper Gumbel-vs-PUCT A/B at matched settings is worth doing.
- If the broader AI-for-chess literature shifts toward one or the other on similar scale.
- If we port to a domain with sharper priors (Go, some RL environments) where Gumbel's m-based truncation is well-suited.

### Affected code (scope preview)
- `src/config.py` — flip chess preset `use_gumbel=True → False` (via CLI `--use-gumbel` override or direct config edit for next run).
- No actual code removal needed — the Gumbel path is already cleanly gated.
- `COMMANDS.md` — update training-launch recipe for next run (drop `--use-gumbel` flag).

### One-line summary
**For chess at our compute scale, classic PUCT is the safer default. Gumbel's advantages don't clearly port from Go and the m=16 truncation is a structural problem. Defer A/B until we have enough compute to run it properly.**

---

## Deferred: Double Base Learning Rate for Chess (1e-3 → 2e-3)

### Problem
Current chess preset uses `lr=1e-3` with Adam, which is conservative by MuZero-family-reimplementation standards. Observational evidence from the current run indicates the optimizer has headroom for larger updates but isn't using it, meaning training converges more slowly than it has to.

### Evidence from current run that LR=1e-3 is under-using headroom

- **Grad norm range: 0.1-0.6.** Clipping threshold is 1.0 (`torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)` in `_train_step`). We're consistently well below the clip — the optimizer never has its updates truncated, meaning we could safely take larger steps per parameter.
- **AMP scaler is confident, not strained.** `train/amp_scale` climbed from 65536 → 131072 between steps 500-2100. That's the `GradScaler` *increasing* FP16 precision because gradients aren't overflowing. The opposite failure mode (staircase-down) would indicate pushing too hard.
- **Policy loss is slow-monotone-decreasing** (8.3 → 4.5 across 10k steps). Slow-monotone descent without oscillation is the signature of undertrained weights absorbing small gradient updates. Bigger updates would accelerate this without obvious destabilization risk.
- **Warmstart targets are clean.** Training against deterministic Stockfish evals, not noisy MCTS bootstraps. Clean supervised targets tolerate higher LR — classical image-classification CNNs use LR=1e-2 with SGD on comparable target cleanliness.

### Reference points
- MuZero paper: LR varies by domain (Atari vs board games), typically 5e-5 to 2e-2.
- muzero-general default: ranges 2e-3 to 1e-2 across games.
- LightZero chess defaults: typically 2e-3.
- Our 1e-3 is in the conservative third of the distribution.

### Decision: bump chess LR 1e-3 → 2e-3 for next major run
Keep everything else:
- `lr_decay_milestones=[0.5, 0.75]` unchanged
- `lr_decay_factor=0.1` unchanged
- `weight_decay=1e-4` unchanged
- Same `MultiStepLR` scheduler

Effective schedule under new LR:
- Steps 0 → 75k: LR = 2e-3
- Steps 75k → 112.5k: LR = 2e-4 (first decay)
- Steps 112.5k → 150k: LR = 2e-5 (second decay)

### Why this is a low-risk experiment at our compute scale
- **Instant divergence signal if wrong.** NaN loss, AMP staircase-down, or grad-norm blow-up shows up within the first 1-2k steps. Roll back and lose ~1 hour.
- **Large expected-value upside.** A 2× LR with the same data gets to any given loss level ~1.5-2× faster. Translates directly to either faster experiments or deeper convergence in the same wall-clock.
- **Unlike algorithmic choices (e.g. Gumbel-vs-PUCT), LR effect is not buried in compute-budget noise.** Convergence rate differences from LR are observable within a few thousand steps, not tens of thousands.

### Optional belt-and-suspenders: 500-step linear warmup
Add LR warmup from 1e-5 → 2e-3 over the first 500 steps. Protects against early-training gradient explosion when weights are near init and gradient statistics haven't stabilized. ~5 lines of scheduler code:
```python
def lr_lambda(step):
    if step < 500:
        return step / 500 * 2.0  # ramps 0 → 2× base over 500 steps, where base is config.lr=1e-3
    return 2.0  # then flat at 2× base until the decay milestones kick in
```
Or, more cleanly, just set `config.lr = 2e-3` and wrap the scheduler in a `torch.optim.lr_scheduler.SequentialLR([warmup_scheduler, multistep_scheduler], milestones=[500])`. Slightly more code but cleaner semantics.

Skip warmup if you want to minimize code changes. The risk profile without warmup is still low — if early steps diverge, you'll know within minutes.

### Deferred follow-up: phase-dependent LR (Option C from discussion)
Once Option B (simple 2× bump) is validated as stable, consider phase-dependent LR: `lr_warmstart=2e-3`, `lr_selfplay=1e-3`. Switch on `pool_alive` just like the phase-dependent `value_loss_weight` plan. Mirrors the "clean supervision → aggressive, noisy bootstrap → conservative" pattern. Small code addition (~5 lines in `_train_step`); defer until Option B has one run's worth of data.

### Affected code (scope preview)
- `src/config.py` — change chess preset `lr: float = 1e-3` → `lr: float = 2e-3`.
- No other code changes for Option B (optional warmup scheduler is the only follow-up work).
