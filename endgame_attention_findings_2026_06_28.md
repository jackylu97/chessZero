# Endgame Conversion — Attention Findings & Autonomous Experiment Log

**Date:** 2026-06-28
**For:** read when you're back. Live results are appended at the bottom by the autonomous pipeline as each experiment finishes; I write the final synthesis + the attention-policy-head determination once all are done.

---

## TL;DR

- The endgame-conversion wall is a **representation problem, not a capacity problem**. Swapping the conv representation tower for a **smolgen self-attention encoder** (1.72M vs the conv 1.54M — *nearly identical param count*) took MCTS conversion from a **flat ~4%** (never improved with training) to **rising 1%→16%** over 30k steps. The gain concentrates in the pure long-range-geometry mates (**KQvK 24%** vs KRvK/KPvK ~7%), exactly as the theory predicts.
- Three autonomous experiments are queued to refine this (no input needed from you):
  1. **(running)** attention + **consistency (SimSiam) + inverse-dynamics** losses — does conditioning the conv dynamics to preserve the geometry through the MCTS rollout beat attention-alone's 16%?
  2. **Smolgen ablation** — attention *without* smolgen, to isolate how much of the gain is the attention body vs. the data-dependent bias.
  3. **Shared attention prediction body** — attention layers shared across the policy/value heads (your suggestion), to get geometry to every search node regardless of dynamics quality.
- After #3 I make a call on whether an **attention policy head** (Lc0 from→to) is still needed for the 0.78 policy-accuracy ceiling.

---

## The arc that got us here

**1. The failure is STALEMATE, not shuffling or blundering.** Per-config diagnostic on the supervised TB checkpoint: of won ≤5-man positions, 79% are converted to a **draw, dominated by stalemate** (even **KQvK only 35%**), 0.1% blundered, 8.7% shuffled. The model drives toward mate (~9-11 plies) then fumbles the final move into stalemate — a long-range, zero-margin *geometric* precision failure.

**2. The targeted fixes (#1 DTM head + #2 terminal-aware search) dented but didn't break it.** Terminal-mask ON cut stalemate 72%→62%, but conversion stayed ~5% — the freed games *shuffle* instead of converting. The failure shifted from stalemate-draw to shuffle-draw. Binding constraint = mating-procedure *execution* = the representation.

**3. More training does NOT break the basin (conv).** A from-scratch 30k-step run with #1+#2 on plateaued: value 0.89, policy 0.76, **MCTS conversion flat ~4% with no trend**. The heads converge; the outcome doesn't.

**4. Attention breaks the plateau.** Same vehicle, only conv→smolgen-attention in the representation:

| metric | conv baseline | **attention (smolgen)** |
|---|---|---|
| params | 1.54M | 1.72M |
| value_acc | 0.86–0.90 (flat) | **0.91–0.93** |
| policy_acc | 0.76 (plateau) | **0.78, rising** |
| MCTS conversion | flat ~4% (no trend) | **rising 1% → 16%** |

Conversion trajectory (every 3k steps):
```
            3k    6k    9k   12k   15k   18k   21k   24k   27k
conv      .027  .043  .050  .047  .040  .047  .040  .020  .060   ← flat
attention .010  .060  .077  .060  .100  .093  .127  .160  .117   ← climbing
```
Per-config (attention, terminal-OFF): **KQvK 24%**, KPRvK 18%, KRvK 7%, KPvK 7% — gain concentrated in the long-range-geometry mates. This is a *representation* win (params held ~constant), not capacity.

---

## Why attention helps (mechanism)

- **Global receptive field:** every square attends to every other in one layer — a1↔h8 directly, vs the conv's ~7-hop propagation. Endgame mating is about long-range relations (king opposition, the confining box, escape-square counting).
- **Learned positional embeddings:** attention is permutation-equivariant (no innate sense of *which* square is which); a per-square learned vector re-injects position. Mandatory because chess is not translation-invariant (the corner/edge matter for mating). Trained jointly in the loop.
- **Smolgen:** a data-dependent additive bias on the attention logits — "*this* diagonal is hot *in this position*". Scaled down for our budget (small bottleneck, 4 heads, shared near-zero-init final projection; starts as a no-op and ramps up).

---

## What's queued (autonomous — runs back-to-back on the GPU, results appended below)

| # | Experiment | Toggle | Tests |
|---|---|---|---|
| 0 | attn + SimSiam consistency + inverse-dynamics (running) | `attn_ssl_inv`, 30k | does a geometry-preserving dynamics beat attention-alone 16%? |
| A | smolgen ablation (attention, **no** smolgen) | `attn_nosmol_ssl_inv`, 24k | how much of the gain is smolgen vs. the attention body |
| B | shared attention prediction body | `attn_predattn_ssl_inv`, 24k | re-attend at every search node (covers dynamics blur), shared across heads |

Each finishes with the per-config MCTS diagnostic (stalemate / conversion / shuffle, per material config). Comparison anchor: the running `attn_ssl_inv` run and the attention-only 16% peak.

**Design notes:** the consistency/inverse/projection heads are *training-only* — the MCTS inference backbone stays small (attn 1.72M; attn+pred-body 2.15M). The pred-body runs in the ~200×/move hot path (free for these supervised experiments, a self-play throughput cost to weigh later). Policy accuracy (0.78) is partly **unroll dilution** (root policy gets only ~1/6 of the gradient) — if it doesn't lift, the targeted fix is the attention policy head and/or up-weighting the k=0 policy loss.

---

## RESULTS (appended live by the pipeline)
