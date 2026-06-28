# Attention Architecture for ChessZero MuZero — Design & Leela Comparison

**Date:** 2026-06-28
**Status:** Design proposal (attention deferred from the #1/#2 endgame run; this is the spec for the next architecture run)
**Context:** Follows the stalemate diagnosis ([[endgame-stalemate-diagnosis]]) and the #1 moves-left-head + #2 terminal-aware-search fixes, which dented but did not break the endgame-conversion basin.

---

## 0. TL;DR / recommendation

The endgame work this session isolated the binding wall as **mating-procedure execution** — a long-range, high-precision *geometric* skill (confine the king, count its escape squares, hold the opposition) that a small-receptive-field CNN with a lossy latent acquires poorly. This is the exact problem Leela hit and solved by moving from SE-ResNet to a **transformer body with smolgen**. 

Recommendation: add a **self-attention encoder to the Representation network first** (cheap — runs once per position, highest geometry payoff), with **learned 8×8 positional embeddings** and, in a second phase, **smolgen-style data-dependent attention biases** (the single most relevant Lc0 idea for our problem). Keep the dynamics conv-based initially (it runs ~200× per MCTS move — attention there is the throughput risk). Validate on the from-scratch TB-endgame vehicle (`train_tb_endgame.py`) against this session's conv baseline, using the same per-config stalemate/conversion diagnostic.

**Do not copy BT4.** Leela's attention nets are 100–240M params; our `chess_small` inference backbone is 1.4M. We need a *small* attention body and must respect the MCTS throughput budget that Leela (which has no learned dynamics) doesn't pay.

---

## 1. Motivation: why attention, and why now

The per-config diagnostic (greedy, on the supervised TB checkpoint) reframed the conversion failure:

- **79.3% of won ≤5-man positions are converted to a DRAW, dominated by STALEMATE.** Even **KQvK converts only 35%.** It is *not* shuffling (8.7%) and *not* blundering (0.1%) — the model drives toward mate (~9–11 plies when it converts) then fumbles the final move into stalemate.
- The #1 (moves-left/DTM head) + #2 (terminal-aware search) fixes did their narrow jobs — terminal-mask ON cut stalemate **71.8% → 62.0%** — but conversion stayed ~5% because the freed games **shuffle** instead of converting. The failure shifted from stalemate-draw to shuffle-draw.
- **Conclusion: the binding constraint is the representation/policy's inability to *execute* the mating procedure**, which is the architecture lever — not more target/search tweaks.

**Empirical confirmation (2026-06-28, `train_tb_endgame.py`):** a from-scratch 30k-step TB-endgame run with BOTH #1+#2 on did NOT learn to convert over time. value_acc (~0.89), policy_acc (plateaus at **0.76**), and the DTM head's won/draw separation (~12 vs ~93 plies) all converge early — but **MCTS conversion stays flat at ~4% the entire run** (range 2–6%, no upward trend). Terminal-mask ON held stalemate to ~53–59% (vs ~72% OFF), but the trade is more shuffling (cap 30%→40%). The heads converge and the *outcome does not*: more training does not break the basin → the lever is architecture.

This is precisely the regime where CNNs are known to struggle, and exactly why Leela moved to attention. From the Lc0 transformer blog: *"For the a1 square to learn about what piece is on h8, the information must make at least 7 trips from square to square."* Endgame mating nets are about long-range relations between distant pieces — the diagonal between two kings, a rook cutting off a file across the board — which a 3×3-conv tower propagates slowly and lossily.

### Why the CNN underperforms in the endgame regime (mechanism)

| | Mid-game (model learns it) | Endgame mate (model fails) |
|---|---|---|
| Signal | **Local** (fork, pin, pawn chain) — CNN-native | **Long-range geometry** (king opposition, confining box across the board) |
| Value surface | **Smooth** (up a pawn ≈ +1 everywhere) | **Jagged** (one square = win ↔ stalemate) |
| Margin | **Forgiving** (many moves keep the edge) | **Zero** (one imprecise move = stalemate draw) |
| Latent | lossy compression is fine | exact position matters; lossy latent destroys the win/draw distinction |

Attention attacks the top row directly: every square attends to every other square in one layer (global receptive field), and smolgen makes those long-range biases *data-dependent on the whole board* (open vs closed position).

---

## 2. Our current architecture (baseline)

(From the codebase survey — `src/model/muzero_net.py`, `src/model/utils.py`.)

- **Representation** (`RepresentationNetwork`, muzero_net.py:17-55): obs `(B, 22×history, 8, 8)` → Conv3×3 stem + LayerNorm + ReLU → `num_residual_blocks` × `ResidualBlock` → min-max-normalized latent `(B, C, 8, 8)`. `C = hidden_planes` (64 for chess_small, 128 for chess).
- **Dynamics** (`DynamicsNetwork`, muzero_net.py:58-152): latent `(B, C, 8, 8)` + action → next latent + reward. Action is a **16-dim learned embedding broadcast to a spatial plane and concatenated** `(B, C+16, 8, 8)` → conv_in → ResidualBlocks → next latent. **Applied K=5 times in training, ~`num_simulations` (200) times per MCTS move.**
- **Prediction** (`PredictionNetwork`, muzero_net.py:208-310): **conv policy head** (`ConvPolicyHead`, 8×8×73 = 4672 logits), **WDL value head** (3 logits), **moves-left head** (21-bin support).
- **Building block:** `ResidualBlock` = Conv→LN→ReLU→Conv→LN→add→ReLU (post-activation). **LayerNorm, not BatchNorm** (deliberate — BN's running stats blow up under the K-step unroll).
- **Sizes:** `chess_small` ≈ 1.4M inference backbone (5.9M for `chess`). Latent **stays spatial 8×8 throughout rep and dynamics → 64 tokens, attention-ready.**

**Hard contract to preserve:** the conv policy head indexes `logit = from_sq*73 + move_type`, so the latent must remain `(B, C, 8, 8)`. Any attention encoder must reshape `(B,64,C) → (B,C,8,8)` on exit.

---

## 3. How Leela does it (accurate, cited)

### 3.1 Evolution
SE-ResNet (squeeze-excitation residual tower, e.g. 10×128, 20×256, 24×320) → attention bodies BT1–BT4. Per the Lc0 "Transformer Progress" comparison (raw-policy Elo vs T78, their strongest conv net): BT1 +13, BT2 +123 (added **smolgen** + more heads), BT3 +179, **BT4 +270 Elo, with fewer params/FLOPs than T78.** Motivation explicitly stated as long-range board relations the conv body propagates slowly.

### 3.2 Encoder body
- **Tokenization:** 64 tokens, one per square (board oriented to side-to-move). Each square's input feature is **112-dim** before embedding: 8 history plies × (12 piece one-hots + 1 repetition) + en-passant/castling/rule50/etc.
- **Concrete (BT4):** 15 encoder layers, **embedding 1024, 32 heads** (head dim 32), **feed-forward d_ff = 4096** (paper convention; the blog's "1536" column is a different convention), ~191M (blog) / 243M (paper, incl. aux heads).
- **Norm/init:** **Post-LayerNorm, encoder-only, DeepNet (DeepNorm) initialization** (not pre-norm). Activation **Mish**. QKV biases omitted (~10% throughput).

### 3.3 Smolgen — the standout idea (most relevant to us)
Dot-product attention captures *content* relations but the positional bias was static. **Smolgen makes the positional attention bias data-dependent on the whole board** ("in a closed position, far-apart squares should be constrained; in an open position, amplified"). Mechanism:
1. **Compress (shared):** linear-project the 64 tokens to 64 vectors of length 32, flatten → 2048, dense → **256-dim bottleneck**.
2. **Per-head generate:** from the 256 bottleneck, a per-head 256-dim vector → a **shared 256×4096** projection → reshape to a **64×64 bias matrix per head** (BT4: 24×64×64).
3. **Add** these logits to QKᵀ **before softmax.**

Effect: *"play as if 50% larger with only 10% throughput reduction."* It factorizes what would otherwise be a 64×64×heads learned table per layer through a tiny shared bottleneck — **exactly the long-range, board-dependent geometry our stalemate problem needs.**

### 3.4 Heads
- **Attention policy head:** project tokens to from-square *queries* and to-square *keys*; **Q·K over squares → a 64×64 move-logit matrix** → re-indexed to Lc0's **1858** legal-move vector (+ a small promotion bias). Contrast: AlphaZero's **8×8×73 = 4672** conv policy learns each (square, move-type) plane independently; Lc0 reasons about a move as a *relationship between two squares*.
- **WDL value head** (3-way) + **Moves-Left Head (MLH)** — note Lc0 puts distance/progress in a **separate head, not the value**, directly corroborating our #1 (moves-left/DTM head).

### 3.5 Comparison table

| | AlphaZero (2017) | **Lc0 BT4 (2024)** | DeepMind Searchless (2024) | **Ours (chess_small)** |
|---|---|---|---|---|
| Body | ResNet ~20×256 conv | 15-layer transformer, embed 1024, 32 heads, d_ff 4096 | 16-layer transformer, embed 1024, 8 heads (270M) | 6× ResidualBlock, 64ch, 8×8 latent |
| Params | ~23M | ~190–243M | 9/136/270M | **~1.4M** |
| Norm | BatchNorm | Post-LN + DeepNorm | post-norm + SwiGLU | **LayerNorm** |
| Tokens | 8×8 conv grid | 64 square-tokens (112-dim) | FEN chars (77 tokens) | 64 latent tokens (post-rep) |
| Attn bias | — | **smolgen (data-dependent 24×64×64)** | — | — |
| Policy | 8×8×73 = 4672 conv | attention from→to, 1858 | argmax over 1968 | 8×8×73 = 4672 conv |
| Value | scalar tanh | WDL + MLH | 128-bin action-value | WDL + MLH |
| Model? | real board | real board | real board (FEN) | **learned latent + dynamics** |
| Search | MCTS | MCTS | none | MCTS |

---

## 4. The crucial difference: MuZero latent + dynamics vs Lc0's direct board

This is the part a naïve "make it a transformer like Lc0" misses:

1. **Lc0 has no learned model.** Attention runs on the *real board*, **once per evaluation**, no recurrence. We have a **Representation** (board→latent, once) **and a Dynamics** net (latent+action→latent) applied **recurrently**: K=5 in training, **~200× per MCTS move.** So:
   - Attention in the **Representation is cheap** (one call/position) and is where long-range board geometry is *first encoded* → **do this first.**
   - Attention in the **Dynamics is the expensive, stability-critical part** (×200 in search, ×5 unrolled in training with a 0.5 grad scale). It is also where the consistency loss must keep the unrolled latent aligned. **Treat as optional / phase 3.**
2. **The latent must survive K dynamics steps.** Geometry encoded by an attention-equipped representation has to be *preserved* by the dynamics. If the dynamics is conv, it may blur the long-range structure the representation built. The EfficientZero **consistency loss** (already in the codebase) is the lever that keeps the unrolled latent on the representation's manifold — it becomes *more* important with an attention rep.
3. **We are ~100× smaller than Lc0.** BT4 is 190M+; `chess_small` is 1.4M. We cannot copy the BT4 hyperparameters. We need embed ≈ 64–128, 3–4 layers, 4–8 heads — and we must watch the param/throughput budget that Lc0 (no dynamics) never pays.

---

## 5. Proposed design for ChessZero (phased)

### Phase A — Attention in the Representation only (lowest risk, do first)
Replace (or interleave with) the representation's `ResidualBlock`s with a small transformer encoder over the 64 latent tokens.

- **Keep the conv stem** (Conv3×3→LN→ReLU) for local feature fusion, then flatten `(B,C,8,8)→(B,64,C)`.
- **Add learned 8×8 positional embeddings.** Chess is **not** translation-invariant (a1 ≠ e4; the corner matters for mating) — positional embeddings are mandatory, unlike a vanilla ViT on natural images.
- **Encoder layer:** multi-head self-attention + FFN, **post-LN + DeepNorm init** (Lc0's recipe — stable for encoder-only depth), Mish or ReLU FFN.
- **Reshape out** `(B,64,C)→(B,C,8,8)` to preserve the conv-policy-head contract.
- **Concrete sizing (chess_small scale):** embed = `hidden_planes` (64), or bump to 128; tokens = 64; **layers = 3–4; heads = 4–8** (head dim 16); d_ff = 2–4× embed. Param budget ≈ 0.5–1.5M — keep the total backbone in the few-million range.
- **Leave dynamics + heads unchanged.**

### Phase B — Smolgen in the representation attention
Add a scaled-down smolgen module: compress 64 tokens → 32 → ~128 bottleneck → per-head → 64×64 bias added to QKᵀ pre-softmax. **This is the highest-value Lc0 idea for our exact (long-range geometry) failure** — it lets the model down-weight/amplify square-pair relations conditioned on the whole position (king-confinement geometry is board-specific). Cheap relative to its effect ("+50% effective size, 10% throughput").

### Phase C — Attention in the Dynamics (optional, expensive — benchmark first)
Either light attention (1–2 layers) in the dynamics, or keep it conv. **Benchmark MCTS throughput** (dynamics+prediction × `num_simulations`) before committing — this is the only place attention threatens self-play wall-clock. The consistency loss must hold the unrolled latent aligned; watch `consistency_loss` and the per-step value/policy CE over the unroll.

### Phase D — Attention policy head (optional, later)
Replace the conv 8×8×73 head with Lc0's from→to Q·K head over the latent tokens (move = relation between two squares). More natural move geometry, but our conv head works and this changes the action interface; defer until A–C show attention helps.

---

## 6. Risks & open questions

- **Throughput (the real risk):** MCTS calls dynamics+prediction ~200×/move. Attention is O(64²·d) per layer — fine for the *representation* (1 call), the concern is *dynamics* attention ×200. Benchmark before Phase C.
- **Data/training:** attention has a weaker inductive bias and usually needs more data. For the **TB-supervised endgame setting data is infinite**, so validate there first (`train_tb_endgame.py`); self-play data volume is the separate bottleneck.
- **Positional embeddings are mandatory** (chess is not translation-invariant).
- **Does the dynamics need attention, or does an attention-encoded latent + conv dynamics suffice?** Open — Phase A/C ablation answers it.
- **Smolgen at our scale:** worth the complexity, or do learned pos-emb + plain attention capture enough of the geometry? Phase A vs B answers it.
- **Param/throughput ceiling for `chess_small`:** how big can the attention body go before self-play wall-clock regresses?

---

## 7. Concrete first experiment

1. Fork `scripts/train_tb_endgame.py` → swap the Representation's `ResidualBlock`s for a Phase-A attention encoder (+ 8×8 pos-emb), keep dynamics/heads/#1/#2 identical.
2. Re-run the same from-scratch TB-endgame curve + the per-config MCTS diagnostic (`diag_perconfig_mcts.py`).
3. **Success metric:** at equal-or-smaller params, does the attention representation **raise conversion / cut (stalemate + shuffle)** beyond this session's conv baseline (conversion ~5%, stalemate 62% with terminal-ON)? Specifically watch whether KQvK / KRvK conversion climbs — those are the pure long-range-geometry mates.
4. If yes, add Phase B (smolgen) and re-measure; then decide on C/D.

---

## 8. Positional embeddings — mechanics & where they apply

**What they are.** A learned positional embedding is just a parameter table: 64 vectors (one per square), each of dimension = the attention embed dim. In PyTorch, `nn.Parameter(torch.zeros(1, 64, embed_dim))` (added to the flattened tokens) or equivalently `nn.Embedding(64, embed_dim)` indexed by square id. Tiny — 64×embed_dim params (≈8k at embed 128), negligible vs the backbone.

**Are they trained in our loop?** Yes — they are ordinary parameters, trained jointly by the *same* Adam optimizer and the *same* loss (value/policy/moves-left CE), via the *same* backprop. There is no separate stage or pre-training. They sit in the forward path (added to each square's token before the attention layers), so gradients flow into them from the heads like any other weight. They start random (small init) and the network learns, e.g., "the a1 vector should encode corner-ness."

**Do they work with our CURRENT (CNN) setup, or do they need attention?** **They are paired with attention — they are essentially meaningless in the current pure-conv network.** The reason: self-attention is *permutation-equivariant* — it flattens the 64 squares into a set and has no inherent notion of which token is which square, so you must re-inject position via the embedding. A **convolution does not have this problem**: the feature map stays spatially laid out (the activation at grid cell (r,c) *is* square (r,c)), so a CNN already "knows where things are" structurally. Adding token-style positional embeddings to a conv net is a no-op (there are no tokens). So: **introduce positional embeddings exactly when you introduce attention, as one package.**

**The CNN analogue (if you want absolute position WITHOUT attention).** Convolutions are *translation-equivariant* — they slide the same kernel everywhere, so they get only weak absolute-position information (from zero-padding at the board edges). If we suspect the current CNN's failure is partly "it can't tell a1 from e4," the cheap fix is **CoordConv**: append 2 constant coordinate planes (normalized row, col) — and optionally a 3rd "is-edge/corner" plane — to the input. This hands the conv net absolute position for ~0 params and **no attention required**. It's worth running as a standalone cheap probe (does absolute-position awareness alone move conversion?) before committing to the attention rewrite.

---

## 9. Full menu of architectural improvements (ranked by what we're seeing)

What the evidence points at: the heads converge (value 0.89, policy 0.76, DTM clean) but the model **cannot execute the precise long-range mating procedure** — it avoids stalemate then shuffles. That implicates four distinct targets; here is the full menu, grouped by target, with cost and expected leverage.

### Group 1 — Long-range geometry (the primary diagnosis)
| # | Improvement | Mechanism | Cost | Leverage |
|---|---|---|---|---|
| 1.1 | **Self-attention encoder in the Representation + learned pos-emb** | global receptive field over the 64 latent tokens; a1↔h8 in one hop | med (0.5–1.5M params; rep runs once/position) | **HIGH** — directly targets the long-range geometry the CNN propagates slowly |
| 1.2 | **Static learned relative-position bias** | dedicated additive logit per (i,j) relation (diagonal/file/rank/distance) — geometry decoupled from content | low (~tens of k) | **HIGH per cost** — hands the model board geometry instead of making it rediscover it through QKᵀ |
| 1.3 | **Smolgen (scaled down)** | data-dependent version of 1.2 — bias changes per board (this diagonal is hot in this position) | med (shrink bottleneck→~64, ~4 heads, low-rank the 64×64 output → ~50–250k) | MED — adds position-specificity; only if 1.2 plateaus |
| 1.4 | **CoordConv coordinate channels** | 2–3 constant row/col/edge planes → absolute position for the *current CNN*, no attention | ~0 | **probe** — cheapest test of whether absolute-position awareness alone helps |

### Group 2 — Capacity & extraction (the jagged value + the policy ceiling)
| # | Improvement | Mechanism | Cost | Leverage |
|---|---|---|---|---|
| 2.1 | **Wider/deeper backbone** (`hidden_planes` 64→128, more blocks) | more capacity to fit the high-frequency endgame value/policy surface | med | MED — the jagged value needs capacity; but KQvK already fails, so capacity alone is not the whole story |
| 2.2 | **Deeper value head** (`value_head_planes/blocks` > 1/0, as we did for MLH) | the `endgame-latent-entangled` finding: the latent *has* the DTZ signal but the 1-plane head can't extract it | low | MED — pairs with the distance targets; cheap |
| 2.3 | **Less-lossy latent** (more latent channels / higher fidelity) | win vs draw = one square; the lossy 8×8×C latent discards the exact-position info that decides it | med | MED — addresses the compression that destroys the win/draw distinction |

### Group 3 — Search reaching deeper (the shuffle)
| # | Improvement | Mechanism | Cost | Leverage |
|---|---|---|---|---|
| 3.1 | **Learned terminal/"done" head** → interior terminal-awareness | #2 is root-only (real board); a learned done-head lets MCTS see stalemate/draw at *latent* interior nodes, so search avoids the cliff several plies deep, not just at the root | med | MED — converts "avoid stalemate at the root" into "steer the whole line away from no-progress" |

### Group 4 — Policy strength (the proximate compounding bottleneck)
| # | Improvement | Mechanism | Cost | Leverage |
|---|---|---|---|---|
| 4.1 | **Attention policy head** (Lc0 from→to Q·K → moves) | a move as a relation between two squares; better move geometry than the 8×8×73 conv head | med | MED — but our conv head works; defer until 1.x shows attention helps |

### Recommended ordering (cheap probes → main bet → residual-driven)
1. **CoordConv probe (1.4)** — ~0 cost, tests whether absolute position alone helps the current CNN. A quick read on "is it geometry-awareness or genuinely capacity/attention."
2. **Deeper value head (2.2)** — cheap, pairs with the DTM head we already added; rules in/out the extraction bottleneck.
3. **Main bet: attention rep + pos-emb + static relative bias (1.1 + 1.2)** on `train_tb_endgame.py`, vs this session's conv baseline (conversion ~4%, stalemate ~55%). Watch KQvK/KRvK specifically — the pure long-range-geometry mates.
4. **Then, residual-driven:** smolgen (1.3) if geometry is still the gap; wider backbone (2.1) / less-lossy latent (2.3) if it looks capacity-bound; learned terminal head (3.1) if the failure is still deep-tree shuffling; attention policy (4.1) last.

Each step re-runs the same instrument: the from-scratch TB-endgame learning curve + the per-config MCTS diagnostic (stalemate / conversion / shuffle), so every change is measured against the identical baseline.

---

## Appendix — sources
- Lc0: lczero.org/blog/2024/02/transformer-progress/ (encoder body, smolgen, BT-series Elo); Chessformer paper arXiv:2409.12272 (BT4 = 15 layers / embed 1024 / 32 heads / d_ff 4096; attention policy; WDL d_value=32); lc0 `src/neural/encoder.{h,cc}` (112-plane input); lczero.org/dev/backend/nn (1858 policy, WDL, MLH).
- DeepMind "Grandmaster-Level Chess Without Search" arXiv:2402.04494 (9/136/270M decoder-only, FEN-char tokens, 128-bin Stockfish action-value, no search, 2895 blitz vs humans).
- AlphaZero arXiv:1712.01815 (20×256 ResNet, 8×8×73 policy, scalar value, 119-plane input).
- Our codebase: `src/model/muzero_net.py` (RepresentationNetwork:17, DynamicsNetwork:58, PredictionNetwork:208, ConvPolicyHead:155, MovesLeftHead:390), `src/model/utils.py` (ResidualBlock:29, norm_layer:7).
