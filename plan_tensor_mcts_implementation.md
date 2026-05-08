# Plan: Hand-Rolled Tensor MCTS on GPU

## Research result: mctx is incompatible

mctx (DeepMind's MCTS library) **does not support Sampled MuZero** (Hubert et al. 2021, "Learning and Planning in Complex Action Spaces", arXiv:2104.06303). Specifically:

- Public API exposes `muzero_policy`, `gumbel_muzero_policy`, `stochastic_muzero_policy` — all three operate over the **full discrete action space** at every node.
- No top-K, sampled-with-replacement, or sampled-without-replacement scheme is exposed.
- No reference to Hubert 2021 / arXiv:2104.06303 in `mctx/_src/policies.py` or the README.
- Sampled MuZero is an open feature request across the major MCTS libs (mctx, muzero-general, LightZero) and **has not shipped anywhere** in a drop-in form.
- mctx is JAX-native — using it from PyTorch needs a JAX↔PyTorch bridge around `recurrent_fn`.

Confirmed by reading `mctx/_src/policies.py`, the README, the policy-improvement demo, and tracking issues #66 / #81 / #93.

mctx **does** support Gumbel MuZero (Danihelka 2022) — relevant only if we re-enable the Gumbel root path for chess (currently disabled per `plan_drop_gumbel_for_chess.md`).

**Decision: hand-roll a PyTorch tensor-native MCTS that preserves the Sampled MuZero §5.1 Proposed Modification.**

Sources:
- [google-deepmind/mctx README](https://github.com/google-deepmind/mctx/blob/main/README.md)
- [mctx/_src/policies.py](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py)
- [Hubert et al. 2021 (Sampled MuZero), arXiv:2104.06303](https://arxiv.org/abs/2104.06303)
- [Danihelka et al. 2022 (Gumbel MuZero, ICLR)](https://openreview.net/forum?id=bERaNdoegnO)

---

## Goal

Move MCTS tree storage and the per-simulation selection/expansion/backprop loop from Python+numpy onto GPU PyTorch tensors. **Eliminate per-simulation CPU↔GPU sync barriers.**

Current per-sim sync count in `BatchedMCTS.run_batch`: **3 syncs × num_simulations × num_plies** (rewards `.tolist()`, values `.tolist()`, probs `.cpu()`). At chess-preset scale (sims=200, plies≈100), that's ~60k sync points per batch — each ~5–50 µs.

After the rewrite: **0 syncs per simulation; 1 sync per ply** (only when committing the chosen action).

## Required correctness invariants

The new implementation **must** preserve:

1. **Sampled MuZero §5.1 Proposed Modification**: at each node, sample K=`config.sample_k` actions i.i.d. *with replacement* from β=π over legal/all actions; deduplicate to K_unique unique actions; use β̂(a) = count(a)/K as the PUCT prior over the sampled subset. Training target = raw N(a)/ΣN(a) over sampled children.
2. **Mover-POV value/reward sign convention**: matches the just-fixed convention in `src/mcts/mcts.py` — `Q(parent_POV) = reward + γ · V(child)_parent_POV = reward − γ · child.value(child_POV)`.
3. **Root-only Dirichlet noise** on the legal-action prior, applied after the K-action sample.
4. **MinMaxStats** Q-normalization per game per search (range tracked over `node.reward − γ · node.value`).
5. **Backwards compat with the rest of the pipeline**: outer interface returns the same per-game `MCTSNode`-equivalent with `value`, `child_visits`, `child_actions` so `select_action`, replay-buffer policy targets, and reanalyze work unchanged.

## Phase 0 — Sync collapse (do first regardless, ~1 day)

Independent of the bigger rewrite, the existing `BatchedMCTS.run_batch` can be cheap-tuned:

- Combine the 3 per-sim transfers (`rewards.view(-1).tolist()`, `value_batch.view(-1).tolist()`, `probs_gpu.cpu()`) into one `.cpu(non_blocking=True)` pinned-memory copy of a single combined buffer, then sync once.
- Move the multinomial sampling for Sampled MuZero leaves to GPU, transfer only the resulting unique-action indices and counts (small payload) instead of the full `[N, 4672]` softmax.

**Expected gain: ~1.5–2× on the per-sim path.** ~50 LOC. Land before Phase 1, both because it's a free win and because it sets up the cleaner data-flow that Phase 1 then GPU-residencies.

## Phase 1 — Tensor data structures

Pre-allocate the search tree as fixed-shape tensors on GPU. **Per game:**

```
N         = num_parallel_games
M         = num_simulations + 1          # max nodes per tree
K         = config.sample_k              # max children per node (50 for chess)
H_d       = hidden_dim, H, W              # latent shape per node
```

Tensor layout (all GPU-resident):

| Name | Shape | Dtype | Purpose |
|---|---|---|---|
| `node_count` | `[N]` | int32 | # nodes allocated per game (starts at 1: root) |
| `parent_idx` | `[N, M]` | int32 | parent node index, −1 for root |
| `parent_child_slot` | `[N, M]` | int32 | which slot in parent's child arrays this node fills |
| `node_visits` | `[N, M]` | int32 | total visits at node |
| `node_value_sum` | `[N, M]` | float32 | accumulated value (in node-POV) |
| `node_reward` | `[N, M]` | float32 | predicted reward of transition into node |
| `node_hidden` | `[N, M, C, H, W]` | float16 | latent state |
| `node_terminal` | `[N, M]` | bool | reserved (we don't track tree-side terminals; absorbing handled by clipped unroll) |
| `child_actions` | `[N, M, K]` | int32 | sampled action indices (−1 = empty slot) |
| `child_priors` | `[N, M, K]` | float32 | β̂ from i.i.d. sample (or full softmax at root) |
| `child_visits` | `[N, M, K]` | int32 | mirrored from node_visits[child_idx] for fast PUCT |
| `child_value_sum` | `[N, M, K]` | float32 | mirrored from node_value_sum[child_idx] |
| `child_rewards` | `[N, M, K]` | float32 | mirrored from node_reward[child_idx] |
| `child_node_idx` | `[N, M, K]` | int32 | which node index this slot points to (−1 = unmaterialized) |
| `mm_min`, `mm_max` | `[N]` | float32 | MinMaxStats per game |

Padding/sentinel rules: empty child slots have `action=-1, prior=0, visits=0, child_node_idx=-1`. Selection masks these out via `prior == 0` or `action == -1`.

Trees are reset (zeroed/sentineled) at each ply via `node_count.fill_(1)` + targeted writes to root entries. Lazy reset: only touch slots up to `max(node_count)` from the previous ply.

## Phase 2 — Vectorized PUCT selection (single-sim path)

Each simulation walks one path from each game's root to its current leaf. Different games sit at different depths; selection therefore runs as a fixed-iteration loop over `max_depth = M` steps, with a `still_walking[N]` mask that gates updates.

Per step:

```
current_node[N] int32                        # initialized to 0 (root) at sim start
still_walking[N] bool                        # initialized to True

for step in range(max_depth):
    # Gather priors/visits/rewards/value_sums for current_node across games
    # Shape [N, K] each.
    # PUCT: prior_score = pb_c * priors * sqrt(node_visits) / (1 + child_visits)
    #       q_raw       = child_rewards - γ * (child_value_sum / max(child_visits, 1))
    #       value_score = (q_raw - mm_min) / max(mm_max - mm_min, ε)  (where range exists)
    #       score       = prior_score + value_score
    # Mask invalid slots: where action == -1 or still_walking is False.
    # argmax along K → next_slot[N]
    
    # Look up child_node_idx[gi, current_node[gi], next_slot[gi]] → child_idx
    # If child_idx == -1, we've hit an unexpanded leaf for that game → still_walking[gi] = False
    # Otherwise current_node[gi] = child_idx; record path[gi, step] = (current_node, next_slot)
    
    if not still_walking.any(): break
```

Output: `path_node_idx[N, max_depth] int32`, `path_slot[N, max_depth] int32`, `leaf_node[N] int32`, `leaf_slot[N] int32`, `leaf_action[N] int32`. Padded with −1 past each game's actual leaf depth.

**Notes:**
- The `gather` over `[N, M, K]` to get the current node's child arrays is one tensor op per stat → 4 small gathers per step.
- argmax along K with a `score_invalid = -inf` mask is a single op.
- `max_depth = M` is a worst-case bound; in practice early-stopping via `still_walking.any()` keeps the average loop short.
- The Python `for step` is fine — it runs ~depth times per sim, not N×depth. Tracing it under `torch.compile` or capturing via CUDA graph is feasible if needed.

## Phase 3 — Batched expansion at leaves

After selection, the network forwards one batch:

```
parent_hidden = node_hidden.gather(... leaf_node ...)            # [N, C, H, W]
next_hidden, reward, policy_logits, value = recurrent_inference(parent_hidden, leaf_action)
```

Then expand each leaf:

1. Allocate a new node slot per game: `new_node_idx[gi] = node_count[gi]; node_count[gi] += 1`.
2. Write `node_hidden[gi, new_node_idx]`, `node_reward[gi, new_node_idx]`, etc.
3. Hook up parent: `child_node_idx[gi, leaf_node, leaf_slot] = new_node_idx`.
4. Sample K actions for the new node:
   ```
   probs = softmax(policy_logits)                                  # [N, action_space]
   sampled_idx = torch.multinomial(probs, K, replacement=True)     # [N, K]
   ```
   Compute β̂ via per-row dedup. **The dedup is the awkward bit**: `unique` per row isn't natively batched. Two options:
   - **(a)** Use `torch.bincount` per row (vmap or Python loop over N — but each is a single GPU op so launches overlap).
   - **(b)** Keep duplicates and let PUCT see K slots with smaller β̂ each. Equivalent in the limit; loses some K_unique cap discipline. **Recommended for v1** because it's bit-clean.
5. Initialize `child_actions[gi, new_node_idx, :] = sampled_idx[gi]`, `child_priors[gi, new_node_idx, :] = 1/K` (under option b) or `count/K` (under option a).

**Memory trick**: `node_hidden` is the dominant storage (`[N, M, C, H, W]` fp16). At chess-preset scale (N=256, M=201, C=64, H=W=6), that's ~240 MB. Workable.

## Phase 4 — Vectorized backprop

Walk path leaf→root, summing reward + γ·value with sign-flip per ply. Static-depth scatter-add over `path_node_idx`.

```
value_per_game = value_batch                                     # [N], leaf-POV
for d in reversed(range(max_depth)):
    nodes_at_d = path_node_idx[:, d]                             # [N]
    valid = nodes_at_d != -1
    # Add value_per_game to node_value_sum at nodes_at_d, masked
    # Add 1 to node_visits at nodes_at_d, masked
    # Mirror to parent's child arrays (parent_child_slot tells us where)
    # Update mm_min, mm_max using node.reward - γ * node.value
    
    # For next iteration, flip POV
    value_per_game = node_reward[nodes_at_d] - γ * value_per_game
```

`scatter_add_` along `[N, M]` with `nodes_at_d` indices, masked with `valid`. Three ops per step (visit, value_sum, mm update).

## Phase 5 — Root handling

Root is a special-case node. At each ply:

1. `obs_batch` from `gpu_game.to_tensor_batch(state)` — already GPU-resident.
2. `legal_mask` from `gpu_game.legal_mask(state)` — `[N, action_space]` bool, GPU-resident.
3. `hidden, policy_logits, value = initial_inference(obs_batch)` — all GPU.
4. Apply legal-mask: set illegal logits to -inf before softmax.
5. Sample K actions from `softmax(masked_logits)` → root's `child_actions` and `child_priors`.
6. Apply Dirichlet noise to root's `child_priors`: `priors = (1-ε) * priors + ε * Dirichlet(α)`. Dirichlet sampling on GPU via `torch.distributions.Dirichlet`.
7. Initialize root: `node_hidden[:, 0] = hidden`, `node_count.fill_(1)`, etc.
8. Run `num_simulations` sims (Phases 2-4).
9. **Single sync at end of ply**: `.cpu()` on `child_actions[:, 0]`, `child_visits[:, 0]`, `node_value_sum[:, 0] / node_visits[:, 0]` — extract per-game action, visit-count policy target, and root value for the game history.

## Phase 6 — Validation

Three layers of testing:

**a) Bit-equivalence to existing `BatchedMCTS`** for fixed network + fixed seed:
- Same network weights, same observation, same legal mask, same sample seed.
- Compare visit counts at root after `num_simulations` simulations. Should match exactly under deterministic CUDA (`torch.use_deterministic_algorithms(True)`) and seeded `torch.multinomial`.
- If exact match isn't achievable due to different reduction orders, target `‖Δvisits‖_∞ / Σvisits < 1%`.

**b) Self-play replay equivalence:**
- Run a self-play game with both `BatchedMCTS` and `TensorMCTS` (same seed, same network).
- Per-ply: same selected action, same visit-count policy, same root value (within tolerance).
- 4 games × ~80 plies each.

**c) Integration smoke test:**
- Full `play_games_parallel_gpu` self-play run with new MCTS, 4 games, sims=4.
- Replay through `ChessGame` and verify obs/legals/actions/rewards/outcome match (mirrors existing `tests/test_chess_gpu_self_play.py::test_history_replay_equivalence`).

## Memory budget

At chess-preset scale (N=256, sims=200, K=50, C=64, H=W=6, fp16 hidden):

| Tensor | Bytes | MB |
|---|---|---|
| node_hidden | 256·201·64·6·6·2 | **~240 MB** |
| 4× child stats (priors, visits, value_sum, rewards) at fp32/int32 | 256·201·50·4·4 | ~41 MB |
| node_visits, node_value_sum, node_reward, parent_idx etc. | 256·201·~24 bytes | ~1.2 MB |
| **Total tree** | | **~280 MB** |

At N=1024: ~1.1 GB. Fits on a 24 GB card but eats into training-batch budget. May need to drop hidden precision to int8 or shrink M (cap simulations).

## Performance estimate

Current `BatchedMCTS` at N=256, sims=20 burns ~13 ms/ply (per the integrated bench), of which ~7 ms is sync overhead and per-game Python tree work. Tensor MCTS targets:

- Sync overhead: → ~0 (1 sync per ply instead of 3·sims).
- Selection + backprop: → ~1-2 ms (vectorized PUCT for [N, K] is one cuBLAS op).
- Network forward: unchanged, dominant once tree ops are gone.

**Expected end-to-end ~2-3× per-ply throughput** at N=256, scaling to ~5-8× at N=1024 (where current Python tree ops grow super-linearly).

## Out of scope for v1

- **Gumbel root** (sequential halving + improved-policy target). Currently disabled for chess (`plan_drop_gumbel_for_chess.md`). Add later if re-enabled — the structure is comparable to mctx's `gumbel_muzero_policy` and we can crib from that as a reference implementation, but in PyTorch.
- **Multi-GPU**. Tensor MCTS naturally shards across devices but data-parallel of self-play has its own complications.
- **Adaptive tree depth / pruning**. Static M is fine until N > 1024.
- **Move tree to int8 for hidden states**. Memory optimization; defer until N > 1024.

## Risks & open questions

1. **Per-row dedup for sampled actions** — option (a) above isn't natively batched in PyTorch. Loop over N rows works but adds N-launches/sim. Option (b) (keep duplicates, β̂ = 1/K uniform over the K slots even when some are dupes) is bit-clean but mathematically equivalent only in the limit. Worth a small experiment to confirm option (b) doesn't shift training dynamics measurably.
2. **Reanalyze compatibility** — reanalyze calls `BatchedMCTS.run_batch` synchronously in the training loop. As long as `TensorMCTS.run_batch` exposes the same outer signature (returns N nodes with `value`, `child_visits`, `child_actions`), swapping is a one-line change. **Build a thin compat shim that returns numpy-backed `MCTSNode` objects** so callers don't change.
3. **Determinism** — `torch.multinomial` and CUDA reductions are nondeterministic by default. For the bit-equivalence test (Phase 6a), enable `torch.use_deterministic_algorithms(True)` in the test path; production self-play doesn't need it.
4. **Memory pressure during training** — if MCTS owns 280 MB and training also holds a large batch, might bump into OOM. Solution: keep tree on the same CUDA stream as training; PyTorch's allocator reuses freed blocks within a stream.

## Effort estimate & sequencing

| Phase | Effort | Risk |
|---|---|---|
| 0. Sync collapse | 1 day | Low — small, isolated |
| 1. Data structures + reset | 2-3 days | Low — pure plumbing |
| 2. Vectorized selection | 3-4 days | Medium — masking edge cases |
| 3. Batched expansion | 2-3 days | Medium — dedup choice |
| 4. Vectorized backprop | 2 days | Low — straightforward scatter |
| 5. Root + glue | 2 days | Low |
| 6. Validation | 2-3 days | Medium — bit-equivalence may need tuning |

**Total: 2-3 weeks of focused work.** Phase 0 is a standalone 1-day win; ship it whether or not we proceed to Phase 1+.

## Decision criteria for actually starting

Per the existing `plan_tensor_mcts.md` memory and confirmed by the integrated bench at N=256 (GPU env still 1.7× slower than CPU env in the full self-play loop), the trigger for starting this work is:

- Self-play throughput becomes the training bottleneck (not network forward, not training step).
- Or N parallel games scaling past 1024.
- Or multi-GPU training plan.

At current scale (N=256, 4090, ~4 it/s training), **none of those triggers are firing**. Phase 0 is worth doing on its own merits; Phases 1-6 should wait for one of the triggers above.
