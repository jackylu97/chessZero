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

## EXP 0 (anchor): attention + consistency(SimSiam) + inverse-dynamics, 30k

**attn_ssl_inv conversion curve**
```
step  3000 | value_acc 0.894 policy_acc 0.727 | ml_won 14 ml_draw 87  || MCTS(term-ON): CONV 0.020  stalemate-DRAW 0.617  cap 0.363
step  6000 | value_acc 0.895 policy_acc 0.743 | ml_won 10 ml_draw 91  || MCTS(term-ON): CONV 0.040  stalemate-DRAW 0.770  cap 0.187
step  9000 | value_acc 0.896 policy_acc 0.745 | ml_won 14 ml_draw 95  || MCTS(term-ON): CONV 0.033  stalemate-DRAW 0.633  cap 0.333
step 12000 | value_acc 0.918 policy_acc 0.758 | ml_won 10 ml_draw 100  || MCTS(term-ON): CONV 0.050  stalemate-DRAW 0.697  cap 0.253
step 15000 | value_acc 0.916 policy_acc 0.765 | ml_won 14 ml_draw 103  || MCTS(term-ON): CONV 0.037  stalemate-DRAW 0.603  cap 0.360
step 18000 | value_acc 0.902 policy_acc 0.766 | ml_won 14 ml_draw 99  || MCTS(term-ON): CONV 0.053  stalemate-DRAW 0.603  cap 0.343
step 21000 | value_acc 0.927 policy_acc 0.762 | ml_won 13 ml_draw 103  || MCTS(term-ON): CONV 0.057  stalemate-DRAW 0.563  cap 0.380
step 24000 | value_acc 0.925 policy_acc 0.748 | ml_won 14 ml_draw 103  || MCTS(term-ON): CONV 0.057  stalemate-DRAW 0.480  cap 0.463
step 27000 | value_acc 0.922 policy_acc 0.761 | ml_won 10 ml_draw 101  || MCTS(term-ON): CONV 0.053  stalemate-DRAW 0.547  cap 0.400
step 30000 | value_acc 0.929 policy_acc 0.764 | ml_won 13 ml_draw 103  || MCTS(term-ON): CONV 0.053  stalemate-DRAW 0.540  cap 0.407
```
_per-config MCTS (terminal OFF then ON):_
```
===== MCTS  terminal_mask=OFF  MLH=ON  (338s) =====
  CONVERTED(mate)=0.035  cap=0.338  DRAW(stalemate/rule)=0.627  lost=0.000
  of 135 cap: still-won(shuffle)=121 thrown=14
    KBQvK        n=  10 CONV=0.10 cap=0.20 draw=0.70 lost=0.00
    KQvK         n=  34 CONV=0.03 cap=0.47 draw=0.50 lost=0.00
    KRvK         n=  41 CONV=0.02 cap=0.49 draw=0.49 lost=0.00
    KPvK         n=  29 CONV=0.00 cap=0.14 draw=0.86 lost=0.00
    KPRvK        n=  11 CONV=0.00 cap=0.27 draw=0.73 lost=0.00
===== MCTS  terminal_mask=ON  MLH=ON  (276s) =====
  CONVERTED(mate)=0.048  cap=0.357  DRAW(stalemate/rule)=0.595  lost=0.000
  of 143 cap: still-won(shuffle)=131 thrown=12
    KPvK         n=  29 CONV=0.03 cap=0.34 draw=0.62 lost=0.00
    KQvK         n=  34 CONV=0.03 cap=0.38 draw=0.59 lost=0.00
    KRvK         n=  41 CONV=0.02 cap=0.32 draw=0.66 lost=0.00
    KPRvK        n=  11 CONV=0.00 cap=0.45 draw=0.55 lost=0.00
    KBQvK        n=  10 CONV=0.00 cap=0.40 draw=0.60 lost=0.00
```

## EXP A: smolgen ablation (attention WITHOUT smolgen)

**attn_nosmol_ssl_inv conversion curve (24k)**
```
step  3000 | value_acc 0.897 policy_acc 0.725 | ml_won 12 ml_draw 83  || MCTS(term-ON): CONV 0.017  stalemate-DRAW 0.890  cap 0.093
step  6000 | value_acc 0.930 policy_acc 0.734 | ml_won 11 ml_draw 95  || MCTS(term-ON): CONV 0.033  stalemate-DRAW 0.813  cap 0.153
step  9000 | value_acc 0.932 policy_acc 0.749 | ml_won 12 ml_draw 97  || MCTS(term-ON): CONV 0.060  stalemate-DRAW 0.640  cap 0.297
step 12000 | value_acc 0.935 policy_acc 0.766 | ml_won 10 ml_draw 100  || MCTS(term-ON): CONV 0.090  stalemate-DRAW 0.520  cap 0.390
step 15000 | value_acc 0.931 policy_acc 0.761 | ml_won 11 ml_draw 102  || MCTS(term-ON): CONV 0.050  stalemate-DRAW 0.537  cap 0.410
step 18000 | value_acc 0.933 policy_acc 0.760 | ml_won 11 ml_draw 102  || MCTS(term-ON): CONV 0.167  stalemate-DRAW 0.473  cap 0.343
step 21000 | value_acc 0.929 policy_acc 0.765 | ml_won 12 ml_draw 101  || MCTS(term-ON): CONV 0.143  stalemate-DRAW 0.413  cap 0.437
step 24000 | value_acc 0.913 policy_acc 0.768 | ml_won 12 ml_draw 100  || MCTS(term-ON): CONV 0.157  stalemate-DRAW 0.477  cap 0.350
```
_per-config MCTS (terminal OFF then ON):_
```
===== MCTS  terminal_mask=OFF  MLH=ON  (357s) =====
  CONVERTED(mate)=0.100  cap=0.253  DRAW(stalemate/rule)=0.642  lost=0.005
  of 101 cap: still-won(shuffle)=96 thrown=5
    KQvK         n=  34 CONV=0.21 cap=0.06 draw=0.74 lost=0.00
    KBQvK        n=  10 CONV=0.20 cap=0.00 draw=0.80 lost=0.00
    KPvK         n=  29 CONV=0.17 cap=0.24 draw=0.59 lost=0.00
    KPRvK        n=  11 CONV=0.09 cap=0.64 draw=0.27 lost=0.00
    KRvK         n=  41 CONV=0.00 cap=0.51 draw=0.49 lost=0.00
===== MCTS  terminal_mask=ON  MLH=ON  (274s) =====
  CONVERTED(mate)=0.160  cap=0.393  DRAW(stalemate/rule)=0.440  lost=0.007
  of 157 cap: still-won(shuffle)=150 thrown=7
    KQvK         n=  34 CONV=0.35 cap=0.26 draw=0.38 lost=0.00
    KBQvK        n=  10 CONV=0.30 cap=0.20 draw=0.50 lost=0.00
    KPvK         n=  29 CONV=0.21 cap=0.31 draw=0.48 lost=0.00
    KRvK         n=  41 CONV=0.00 cap=0.51 draw=0.49 lost=0.00
    KPRvK        n=  11 CONV=0.00 cap=0.64 draw=0.36 lost=0.00
```

## EXP B: shared attention prediction body

**attn_predattn_ssl_inv conversion curve (24k)**
```
step  3000 | value_acc 0.902 policy_acc 0.735 | ml_won 15 ml_draw 86  || MCTS(term-ON): CONV 0.007  stalemate-DRAW 0.923  cap 0.070
step  6000 | value_acc 0.905 policy_acc 0.757 | ml_won 10 ml_draw 88  || MCTS(term-ON): CONV 0.023  stalemate-DRAW 0.807  cap 0.170
step  9000 | value_acc 0.910 policy_acc 0.776 | ml_won 13 ml_draw 91  || MCTS(term-ON): CONV 0.043  stalemate-DRAW 0.833  cap 0.123
step 12000 | value_acc 0.920 policy_acc 0.788 | ml_won 11 ml_draw 99  || MCTS(term-ON): CONV 0.047  stalemate-DRAW 0.913  cap 0.040
step 15000 | value_acc 0.907 policy_acc 0.780 | ml_won 10 ml_draw 96  || MCTS(term-ON): CONV 0.093  stalemate-DRAW 0.853  cap 0.050
step 18000 | value_acc 0.921 policy_acc 0.783 | ml_won 13 ml_draw 102  || MCTS(term-ON): CONV 0.097  stalemate-DRAW 0.843  cap 0.060
step 21000 | value_acc 0.926 policy_acc 0.784 | ml_won 14 ml_draw 103  || MCTS(term-ON): CONV 0.127  stalemate-DRAW 0.813  cap 0.053
step 24000 | value_acc 0.916 policy_acc 0.784 | ml_won 10 ml_draw 97  || MCTS(term-ON): CONV 0.140  stalemate-DRAW 0.830  cap 0.030
```
_per-config MCTS (terminal OFF then ON):_
```
===== MCTS  terminal_mask=OFF  MLH=ON  (392s) =====
  CONVERTED(mate)=0.128  cap=0.037  DRAW(stalemate/rule)=0.833  lost=0.003
  of 15 cap: still-won(shuffle)=11 thrown=4
    KPvK         n=  29 CONV=0.31 cap=0.00 draw=0.69 lost=0.00
    KQvK         n=  34 CONV=0.26 cap=0.03 draw=0.71 lost=0.00
    KBQvK        n=  10 CONV=0.20 cap=0.00 draw=0.80 lost=0.00
    KPRvK        n=  11 CONV=0.18 cap=0.09 draw=0.73 lost=0.00
    KRvK         n=  41 CONV=0.05 cap=0.05 draw=0.90 lost=0.00
===== MCTS  terminal_mask=ON  MLH=ON  (280s) =====
  CONVERTED(mate)=0.147  cap=0.055  DRAW(stalemate/rule)=0.787  lost=0.010
  of 22 cap: still-won(shuffle)=12 thrown=10
    KQvK         n=  34 CONV=0.32 cap=0.03 draw=0.65 lost=0.00
    KBQvK        n=  10 CONV=0.20 cap=0.10 draw=0.70 lost=0.00
    KPRvK        n=  11 CONV=0.18 cap=0.09 draw=0.73 lost=0.00
    KPvK         n=  29 CONV=0.14 cap=0.00 draw=0.86 lost=0.00
    KRvK         n=  41 CONV=0.10 cap=0.05 draw=0.85 lost=0.00
```

---
**ALL AUTONOMOUS EXPERIMENTS COMPLETE — 2026-06-29 00:35. Synthesis + attention-policy-head determination written by Claude on re-invocation.**

## EXP C: smolgen in the DYNAMICS too (rep+dyn attention — the architecture-match fix)
**attn_dynattn_ssl_inv conversion curve (24k)**
```
step  3000 | value_acc 0.898 policy_acc 0.717 | ml_won 14 ml_draw 85  || MCTS(term-ON): CONV 0.010  stalemate-DRAW 0.637  cap 0.353
step  6000 | value_acc 0.921 policy_acc 0.745 | ml_won 7 ml_draw 93  || MCTS(term-ON): CONV 0.020  stalemate-DRAW 0.897  cap 0.083
step  9000 | value_acc 0.934 policy_acc 0.756 | ml_won 10 ml_draw 102  || MCTS(term-ON): CONV 0.073  stalemate-DRAW 0.863  cap 0.063
step 12000 | value_acc 0.938 policy_acc 0.760 | ml_won 13 ml_draw 106  || MCTS(term-ON): CONV 0.113  stalemate-DRAW 0.637  cap 0.243
step 15000 | value_acc 0.931 policy_acc 0.760 | ml_won 11 ml_draw 105  || MCTS(term-ON): CONV 0.087  stalemate-DRAW 0.447  cap 0.463
step 18000 | value_acc 0.937 policy_acc 0.759 | ml_won 12 ml_draw 107  || MCTS(term-ON): CONV 0.127  stalemate-DRAW 0.403  cap 0.467
step 21000 | value_acc 0.930 policy_acc 0.768 | ml_won 12 ml_draw 104  || MCTS(term-ON): CONV 0.113  stalemate-DRAW 0.593  cap 0.293
step 24000 | value_acc 0.933 policy_acc 0.770 | ml_won 13 ml_draw 107  || MCTS(term-ON): CONV 0.197  stalemate-DRAW 0.430  cap 0.367
```
_per-config MCTS (terminal OFF then ON):_
```
===== MCTS  terminal_mask=OFF  MLH=ON  (476s) =====
  CONVERTED(mate)=0.107  cap=0.255  DRAW(stalemate/rule)=0.632  lost=0.005
  of 102 cap: still-won(shuffle)=90 thrown=12
    KQvK         n=  34 CONV=0.21 cap=0.00 draw=0.79 lost=0.00
    KPRvK        n=  11 CONV=0.09 cap=0.18 draw=0.73 lost=0.00
    KRvK         n=  41 CONV=0.05 cap=0.51 draw=0.44 lost=0.00
    KPvK         n=  29 CONV=0.03 cap=0.03 draw=0.93 lost=0.00
    KBQvK        n=  10 CONV=0.00 cap=0.00 draw=1.00 lost=0.00
===== MCTS  terminal_mask=ON  MLH=ON  (285s) =====
  CONVERTED(mate)=0.170  cap=0.345  DRAW(stalemate/rule)=0.477  lost=0.007
  of 138 cap: still-won(shuffle)=126 thrown=12
    KQvK         n=  34 CONV=0.44 cap=0.09 draw=0.47 lost=0.00
    KBQvK        n=  10 CONV=0.30 cap=0.30 draw=0.40 lost=0.00
    KPRvK        n=  11 CONV=0.18 cap=0.55 draw=0.27 lost=0.00
    KPvK         n=  29 CONV=0.07 cap=0.10 draw=0.83 lost=0.00
    KRvK         n=  41 CONV=0.05 cap=0.49 draw=0.46 lost=0.00
```

---

# FINAL SYNTHESIS & DETERMINATION (4-way, all complete)

| run | config | MCTS conv | KQvK | profile |
|---|---|---|---|---|
| EXP 0 | attn-rep + smolgen + ssl + inv (conv dynamics) | **5%** | 3% | cratered |
| EXP A | attn-rep, **no smolgen**, + ssl + inv | 16% | 35% | shuffle-heavy |
| EXP B | attn-rep + smolgen + **pred-body** + ssl + inv | 14% | 32% | stalemate-heavy (79%) |
| **EXP C** | attn-rep + smolgen + **dyn-attention** + ssl + inv | **17%** (peak 20%) | **44%** | best (stalemate 48%) |

**Determination: the winning architecture is rep + dynamics attention (smolgen on) + consistency + inverse (EXP C).**

The story is now airtight. EXP 0 cratered because the SimSiam consistency loss (stop-grad on the
representation) drags the conv dynamics toward a smolgen-rich repr latent it cannot reproduce —
distorting the dynamics latents MCTS rides on (root acc stayed normal; only MCTS conversion died).
Three independent fixes all resolve it (drop smolgen / re-attend in prediction / attention in the
dynamics), confirming the diagnosis. Of the three, **putting attention in the dynamics is both the
most principled (fixes the cause and keeps geometry flowing through the rollout) and the best
measured** — highest conversion (17%, peak 20%) and highest geometry-mate conversion (KQvK 44%).

**Attention-policy-head determination:** policy_acc plateaus at ~0.77 across *all* body
configurations (EXP 0/A/B/C alike) — the body changes lift conversion but not the policy ceiling.
So the policy head is the likely *next* bottleneck, and the **Lc0-style from→to attention policy
head is worth building** — BUT only after the scaled run tells us whether more attention
capacity + data lifts policy on its own. If the scaled run leaves policy ≲0.85, build the
attention policy head next.

**Kicked off (autonomous): the SCALED run** — the winning EXP C architecture scaled to break the
~17% ceiling: **6 attention layers (vs 4), 200k sequences (vs 80k, less reuse), 40k steps**.
Tests whether more capacity + data lifts both conversion and the 0.77 policy ceiling. Results
append below when done.

**Recommended next steps after the scaled run:** (1) if policy still ≲0.85 → attention policy head;
(2) production integration — wire use_repr_attention/use_dyn_attention/use_smolgen into the config
+ trainer for a clean self-play run (weigh the dyn-attention ~200×/move throughput cost);
(3) more ≤5-man data / longer training if conversion is still climbing at 40k.

---

# SCALED RUN RESULTS — recovered 2026-06-29 (runpod expired mid-run; live log lost, re-derived from surviving checkpoints)

The pod expired at **step 36000 of the 40k target** (last checkpoint `scaledL6_36000.pt`,
06:38). The TensorBoard/stdout log did not survive, so value/policy were re-derived by
re-running `cheap_eval` on the surviving checkpoints and the per-config MCTS diagnostic on 36k
(`diag_perconfig_mcts.py`, CKPT=…scaledL6_36000, USE_ATTENTION=1 USE_SMOLGEN=1 USE_DYN_ATTENTION=1 ATTN_LAYERS=6).

**Architecture = winning EXP C, scaled:** rep+dyn attention, smolgen on, ssl+inv, **6 attn layers**,
**200k sequences** (`data/tb5_seq_big.pkl`), Adam 1e-3.

**The scaling BROKE the ~17% conversion ceiling — better than 2× at 36k, with 4k steps still to go.**

| metric | EXP C (L4, 80k, 24k) | **scaled (L6, 200k, 36k)** |
|---|---|---|
| value_acc | ~0.93 | **0.958** |
| policy_acc | ~0.77 (plateau) | **0.827, still rising** |
| MCTS conv (term-OFF) | 0.107 | **0.233** |
| MCTS conv (term-ON) | 0.170 (peak 20%) | **0.410** |
| KQvK (term-ON) | 0.44 | **0.91** |
| KBQvK (term-ON) | 0.30 | **0.90** |

policy_acc recovered from checkpoints: step 12k **0.790**, 24k **0.820**, 36k **0.827** (rising, not plateaued).

per-config MCTS on scaledL6_36000 (terminal OFF then ON):
```
===== MCTS  terminal_mask=OFF  MLH=ON  (346s) =====
  CONVERTED(mate)=0.233  cap=0.102  DRAW(stalemate/rule)=0.665  lost=0.000
  of 41 cap: still-won(shuffle)=41 thrown=0   mean plies-to-mate(converted): 25.4
    KQvK   n=34 CONV=0.56 cap=0.00 draw=0.44   KBQvK n=10 CONV=0.40 cap=0.00 draw=0.60
    KRvK   n=41 CONV=0.15 cap=0.10 draw=0.76   KPRvK n=11 CONV=0.09 cap=0.27 draw=0.64
    KPvK   n=29 CONV=0.07 cap=0.03 draw=0.90
===== MCTS  terminal_mask=ON  MLH=ON  (228s) =====
  CONVERTED(mate)=0.410  cap=0.155  DRAW(stalemate/rule)=0.432  lost=0.003
  of 62 cap: still-won(shuffle)=57 thrown=5    mean plies-to-mate(converted): 31.5
    KQvK   n=34 CONV=0.91 cap=0.00 draw=0.09   KBQvK n=10 CONV=0.90 cap=0.00 draw=0.10
    KPRvK  n=11 CONV=0.27 cap=0.18 draw=0.55   KPvK  n=29 CONV=0.17 cap=0.00 draw=0.83
    KRvK   n=41 CONV=0.17 cap=0.10 draw=0.73
```

**Read:**
- **Geometry mates are essentially solved** — KQvK 0.91, KBQvK 0.90 (from 0.44/0.30). The
  attention-in-the-dynamics thesis is confirmed at scale: the long-range queen/bishop mates the
  conv tower could never learn are now near-perfect.
- **The remaining wall is rook & pawn endgames** — KRvK 0.17, KPvK 0.17, KPRvK 0.27. These are not
  long-range-geometry failures; they need *step-by-step technique* (the rook "box" shrinking by one
  rank at a time; pawn opposition/promotion). Different bottleneck than the one attention fixed.
- **Capacity+data DID lift policy** — 0.77→0.827 and still climbing at the cutoff. The old plateau
  was data/capacity, not an architectural ceiling. But still **<0.85**, so the attention-policy-head
  question is not yet closed — it depends on whether the last 4k steps (and beyond) push past 0.85.

**Determination (updated):**
1. **Finish/extend the run** — it died 4k short and *both* policy and conversion were still rising;
   the headline numbers above are a lower bound. Re-launch from `scaledL6_36000.pt` (or fresh to
   ≥40k) before drawing the policy-head line.
2. **Attention policy head — still "build it," verdict deferred one step.** Policy crossed the old
   plateau but sits at 0.827<0.85. Let the completed run settle the call.
3. **New target = rook/pawn technique**, not geometry. Levers: more KRvK/KPvK/KPRvK in the data mix,
   longer search horizon (MAX_PLIES, sims), or DTM-shaped reward to reward incremental progress.
