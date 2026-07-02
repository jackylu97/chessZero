# Run Setup Audit — `2026_06_30_attn_warmstart_fix`

Complete enumeration of everything active in the current self-play run, so we can see the
whole picture in one place. **Effective value** = chess_small preset default *after* the CLI
overrides in `scripts/_launch_attn_warmstart.sh`. Where the CLI changes a default, both are shown.

> TL;DR of what makes this run distinct from a vanilla MuZero run: **(a)** smolgen
> self-attention backbone (rep+dyn), **(b)** a Stockfish warmstart anchor, **(c)** the full
> tablebase decisive-signal stack (root probe + value/DTM relabel), **(d)** endgame seeding with
> a held-out split, **(e)** material-margin value shaping + resignation, **(f)** moves-left head +
> search utility, **(g)** terminal-aware root search. Reanalyze is OFF.

---

## 0. Identity & launch

| | |
|---|---|
| run-id | `2026_06_30_attn_warmstart_fix` |
| preset | `chess_small` (= `chess` preset, model shrunk ~5×; `replace(configs["chess"], hidden_planes=64, num_residual_blocks=6, fc_hidden=64, ...)`) |
| game | chess (ChessGame / GpuChessGame), 8×8, action space 4672 |
| device | cuda, AMP fp16 (`use_amp=True`, `tensor_mcts_hidden_dtype=float16`) |
| training_steps | 150,000 |
| launcher | `scripts/_launch_attn_warmstart.sh` (tmux `selfplay`, tee → `selfplay_attn_warmstart_fix.log`) |
| net params | 5,877,462 (~5.88M) |
| checkpoint/TB | `checkpoints/chess/<run>` , `runs/chess/<run>` (note: under **`chess`** the game, not `chess_small`) |

---

## 1. Model architecture (the network) — `src/model/muzero_net.py`

| component | effective | notes |
|---|---|---|
| hidden_planes (latent C) | **64** | latent is `64×8×8` |
| latent_h × latent_w | 8 × 8 | one token per board square |
| history_frames | 8 | obs = `22 planes × 8 frames = 176` input channels, STM-canonical encoding (+plane 17 = absolute turn color) |
| **representation body** | **smolgen self-attention** (`use_repr_attention=True`) | `BoardAttentionEncoder`, **4 layers**, 4 heads, smolgen ON. **Replaces** the conv residual tower (a conv stem remains) |
| **dynamics body** | **smolgen self-attention** (`use_dyn_attention=True`) | same encoder (4L/4H, smolgen ON). Action via learned `nn.Embedding(4672, 16)` broadcast spatially + conv_in + LightZero residual skip |
| prediction body | **conv** (`use_pred_attention=False`) | pred-attention tested in supervised proxy, did NOT help → off |
| attn_layers / attn_heads | 4 / 4 | supervised proxy best was L6; L4 chosen for throughput |
| use_smolgen | True | data-dependent additive attention bias, shared near-zero-init final proj |
| policy head | **conv** (AlphaZero 73-plane, `policy_head_type="conv"`) | 64×73 = 4672, weight-shared across from-squares |
| value head | **WDL** (`value_head_type="wdl"`) | 3 logits (W,D,L), `V = P(W)−P(L)+draw_score·P(D)`; `value_head_planes=1`, `blocks=0`; `value_head_init_std=0.01` (cold-start gradient unblock) |
| moves-left head | ON (`use_moves_left=True`) | categorical over `2·10+1=21` bins; `moves_left_support_size=10`; `moves_left_head_planes=1`, `blocks=0`; **sqrt transform** (`moves_left_use_transform=True`) |
| material head | ON (`use_material_head=True`) | distributional STM material margin, `material_head_support_size=8`; aux loss only (does not touch MCTS Q) |
| consistency (SimSiam) head | ON (`use_consistency_loss=True`) | training-only; proj 512/512, pred 256/512, single-frame target |
| inverse-dynamics head | ON (`use_inverse_dynamics_loss=True`) | training-only; ICM, hidden 96; the validated action-blindness fix |
| norm | LayerNorm everywhere (not BN) | |

Heads that exist on the net but are **training-only / not in the MCTS inference tuple**:
moves-left (queried separately for the search utility), material, projection/predictor (SimSiam),
inverse-dynamics.

---

## 2. Self-play generation — `src/training/self_play.py`

| param | effective (CLI) | default | meaning |
|---|---|---|---|
| engine | GPU-resident (`use_gpu_resident_self_play=True`, `use_gpu_chess=True`, `use_tensor_mcts=True`) | | whole sweep stays on GPU |
| num_self_play_games | **512** | 1024 | games per self-play round |
| num_parallel_games | **512** | 384 | batch width (1 sweep covers all 512) |
| self_play_interval | 660 | | train steps between self-play rounds |
| self_play_warmup_steps | **15000** | 0 | **first 15k steps train on warmstart injection ONLY (no self-play)** |
| max_plies | 750 | | ply cap (cold-start games hit it; warmstarted ~90–115) |
| random_opening_plies | 0 | | no random openings (temperature + Dirichlet only) |
| temperature_init / final | 1.0 / 0.1 | | + `temperature_schedule=[(0.5,0.5),(0.75,0.25)]`, `temperature_drop_step=30` |
| mask_illegal_policy | **True** | False | illegal moves masked in the **root** policy target/loss (`illegal_policy_penalty=1.0`) |

---

## 3. Search (MCTS) — `src/mcts/tensor_mcts.py`

| param | effective | default | meaning |
|---|---|---|---|
| num_simulations | **400** | 200 | sims/move (the main throughput cost × dyn-attention) |
| backend | triton (`tensor_mcts_select_backend="triton"`) → **downgraded to eager/compile** because `root_terminal_draws` + `tb_root_probe` are on | | |
| c_puct | 1.25 | | PUCT exploration |
| sample_k | 50 | | Sampled-MuZero: K distinct actions/node |
| dirichlet_alpha / epsilon | 0.1 / 0.25 | | root noise |
| **root_terminal_draws** | **True** | (preset) | pins repeating + stalemate/insufficient ROOT moves to `draw_score`; `min_repeats=2`, `include_stalemate=True` |
| draw_score | −0.05 | | value assigned to forced-draw moves |
| **moves_left_mcts** | **True** | (preset) | MLH search utility: `sign(-Q)·clip(ml_slope·Δm, ±ml_max_effect)·qscale`, gated `|Q|>ml_threshold` |
| ml_slope / ml_max_effect | **0.02 / 0.3** | 0.005 / 0.1 | **stronger ML utility than default** (endgame-tuned) |
| ml_threshold | 0.3 | | utility only engages in clearly won/lost |
| ml_q_const/linear/square | 0 / 1.6521 / −0.6521 | | smooth |Q| ramp for the utility |
| **tb_root_probe** | **True** | False | **search-time Syzygy DTZ steering** — overwrites root children `value_score` with mover-POV DTZ-shaped TB value (winning→≈+1 minus DTZ penalty); `tb_max_pieces=5`, `tb_dtz_weight=0.05` |
| leaf legal masking | **NOT implemented** | | interior/leaf nodes expand full 4672 action space (structural MuZero limit) |
| interior repetition penalty | **NOT implemented** | | only the root override above |
| subtree reuse | off (`tensor_mcts_subtree_reuse=False`) | | |

---

## 4. The decisive-signal stack (endgame conversion machinery)

This is the heart of what's different. Four independent injections of "ground truth / decisive":

1. **`tb_root_probe`** (search, §3) — DTZ steers the *search* → the corrected visit distribution
   becomes the policy target → the **policy** learns conversion. `tb_gaviota_path=data/gaviota`.
2. **TB value relabel** (`tb_value_weight=1.0`, `tb_value_dtz_shape=0.5`) — at ≤5-man plies the
   value target is replaced by the Syzygy WDL (DTZ-shaped, STM-relative). `tb_value_hard=False`
   (soft). Blended in `replay_buffer.make_target`.
3. **TB moves-left relabel** (`tb_moves_left_weight=1.0`) — at in-TB decisive plies the moves-left
   target = Gaviota `|DTM|` (ground-truth distance-to-mate) instead of policy-rollout length.
4. **Material-margin value shaping** (`material_value_weight=0.5`, anneal frac 0.6 → final 0.0,
   `material_value_scale=5.0`) — KataGo score-margin: blends `tanh(material_margin/scale)` into the
   value target so two won positions are rank-able. **Annealed**: strong early, fades to recover
   the true objective. Plus the **material head** (aux loss, `material_head_loss_weight=0.25`).

Plus **resignation** (`resign_enabled=True`, threshold −0.9, consecutive 5, holdout 0.20) —
truncates+relabels a side that stays below threshold for 5 of its own moves; protects the decisive
label. **(This is the path we just bug-fixed for black-to-move-start games.)**

Deferred TB relabel off the hot path: `tb_relabel_workers=8` (pooled batched relabel pass).

**OFF in this run:** `tb_policy_weight=0.0` (no soft TB *policy* relabel — steering is via the
root-probe value override only), `tb_steer_policy=False`.

---

## 5. Training losses & schedules — `src/training/trainer.py`

| loss term | weight | notes |
|---|---|---|
| policy CE | 1.0 | k=0 + K unroll steps, each ×`1/(K+1)` |
| value CE (WDL) | **1.0 in self-play / 1.0 in warmstart** | `value_loss_weight_selfplay=1.0`, `value_loss_weight_warmstart=1.0` (override the base `value_loss_weight=0.25`) |
| reward CE | (in unroll) | reward support size 1 |
| moves-left CE | 0.25 | `moves_left_loss_weight`; **trainer ALWAYS applies sqrt scalar_transform** regardless of `moves_left_use_transform` (flag only affects search decode) |
| material head | 0.25 → 0.0 | `material_head_loss_weight`, annealable |
| consistency (SimSiam) | 2.0 | `consistency_loss_weight`, single-frame target |
| inverse-dynamics | 1.0 | `inverse_dynamics_loss_weight` |
| num_unroll_steps (K) | 5 | dynamics applied 5× per sample; hidden grads ×0.5 |
| value-target Q-blend | `q_ratio=0.5`, `warmstart_q_ratio=0.5`, **`selfplay_q_ratio=0.0`** | self-referential Q-blend OFF in self-play (anti-collapse) |
| td_steps | −1 | pure Monte-Carlo return (WDL path ignores td_steps anyway) |
| discount | 1.0 | |

Optimizer: Adam, **lr=1e-3**, weight_decay=1e-4, warmup 500 steps, `MultiStepLR` decay ×0.1 at
[0.5, 0.75]·150k. AMP on. Non-finite-loss guard skips the step.

---

## 6. Replay buffer, PER, warmstart anchor — `src/training/replay_buffer.py`

| param | effective (CLI) | default | meaning |
|---|---|---|---|
| replay_buffer_size | 5120 | | ~2.5 self-play rounds resident (sparse policies) |
| PER | on, `per_alpha=0.6`, `per_beta_init=0.4`, `per_epsilon=1e-6` | | |
| decisive_sample_frac | 0.0 | | stratified decisive oversampling OFF (overfit a tiny pool) |
| **Stockfish injection** | path `data/stockfish_injection`, **300 games**, interval **256** | 0/0 | warmstart teacher stream bootstrapped + topped up |
| warmstart_buffer_size | **300** | None | two-pool: 300 warmstart games protected from eviction |
| warmstart_sample_frac | **0.4 → 0.1** over `anneal_frac=0.6` | 0.0 | fraction of each batch drawn from the warmstart pool, annealed down |
| reanalyze | **OFF** (`reanalyze_interval=0`) | 1024 | **no MCTS re-running on stored positions** (throughput; TB relabel supplies fresh targets) |
| save_buffer | True | | `checkpoint_interval=1000`; buffer saved periodically (warmstart excluded) |

---

## 7. Endgame seeding & holdout

| param | effective | meaning |
|---|---|---|
| endgame_seed_frac | **0.5** | half of each self-play round seeded from ≤5-man FENs (GPU-resident path) |
| endgame_seed_archive | `data/endgame_seeds_train.txt` | 74,786 FENs (85% TRAIN split) |
| **holdout** | `data/endgame_seeds_holdout.txt` | **13,198 FENs (15%) — never trained on**, for clean generalization eval (manual, via diag tools) |
| seed monitor | auto `seed/*` TB scalars | conversion_rate, conversion_white/black, mate_rate, draw_rate, balance counts |

---

## 8. What's OFF / notable disabled (so we don't assume)

- **Reanalyze** (`reanalyze_interval=0`).
- **Soft TB policy relabel** (`tb_policy_weight=0`), **TB policy steering** (`tb_steer_policy=False`).
- **Hard WDL value** (`tb_value_hard=False` — soft DTZ-shaped instead).
- **Gumbel MuZero** (`use_gumbel=False`).
- **Leaf legal masking** & **interior repetition penalty** — not implemented (structural MuZero).
- **Target-side repetition penalty** (`repetition_penalty=0.0` — chess_small disables it; only the
  search-side `root_terminal_draws` handles repetitions).
- **Network torch.compile for training** (`compile_network=False`; MCTS net IS compiled,
  `tensor_mcts_compile_net=True`).
- **Subtree reuse** (`tensor_mcts_subtree_reuse=False`).

---

## 9. Data inputs (on disk, gitignored)

| path | what |
|---|---|
| `data/syzygy` (291 files) | 3-4-5-man Syzygy WDL/DTZ tablebases (root probe + value relabel) |
| `data/gaviota` (.gtb.cp4) | Gaviota DTM tablebases (moves-left relabel) |
| `data/stockfish_injection` (bucket_*) | Stockfish warmstart teacher games, bucketed by material |
| `data/endgame_seeds_train.txt` | 74.8k seed FENs (train) |
| `data/endgame_seeds_holdout.txt` | 13.2k seed FENs (holdout) |

---

## 10. Mate-rate diagnostic — `checkpoint_26000`, RAW model (no `tb_root_probe`)

Per-config playout of won ≤5-man test positions through MCTS (200 sims, MLH on), classifying
each game mate / cap(shuffle) / draw(stalemate). **This is the model WITHOUT the TB search crutch.**

```
===== terminal_mask=OFF  MLH=ON =====
  CONVERTED(mate)=0.007  cap=0.245  DRAW(stalemate/rule)=0.748  lost=0.000
    KRvK  CONV 0.00   KQvK CONV 0.00   KPvK CONV 0.00   KPRvK CONV 0.00   KBQvK CONV 0.00
===== terminal_mask=ON  MLH=ON =====
  CONVERTED(mate)=0.013  cap=0.285  DRAW(stalemate/rule)=0.703  lost=0.000
    KBQvK CONV 0.10  KRvK 0.00  KQvK 0.00  KPvK 0.00  KPRvK 0.00
```

**The raw self-play model converts ≈ 0%** (even KQvK, the easiest). Failure is **dominated by
stalemate/draw (0.70–0.75)**; the model drives toward mate (converted games average ~17 plies) but
overwhelmingly fumbles into stalemate.

### The decisive comparison
| | architecture | training targets | KQvK conv | overall conv | draw |
|---|---|---|---|---|---|
| **Supervised attention proxy** (`tb5_endgame`) | rep+dyn attention | **clean tablebase DTM/WDL** | **0.91** | ~0.41 | ~0.48 |
| **Production self-play** (this run, 26k, raw) | SAME rep+dyn attention | self-play V^π + TB relabel | **0.00** | ~0.01 | ~0.72 |

**Same architecture, opposite result.** The attention backbone *demonstrably can* learn the mating
technique (supervised proxy → KQvK 0.91), but in the self-play regime it **does not** — at 26k the
raw model can't convert at all. The seeded-conversion metric in self-play (`seed/conversion ≈ 0.45`,
`mate_rate ≈ 0.02`) is almost entirely the **`tb_root_probe` search crutch + resignation**, NOT the
model's own skill. Strip the crutch and conversion is ~0%, stalemate-dominated.

### Implication
The self-play loop is **not transferring the conversion technique into the network**, even with the
full TB decisive stack. The model leans on `tb_root_probe` at search time and never internalizes the
mate, so the policy/value heads stay at the stalemate-prone baseline. The draw basin in self-play is
therefore **not (only) a value-target problem** — it's that the per-move *policy* signal needed to
learn precise mating (which clean DTM targets provided in the proxy) isn't reaching the net through
self-play. Candidate causes to investigate: the root-probe edits *value_score* (Q) not the prior, so
the policy target is the visit distribution — which may be too soft/imprecise to teach the exact
mating move; `tb_policy_weight=0` (soft TB policy relabel is OFF); reanalyze is OFF; and the
moves-left/material signals shape value, not the policy's move choice.
