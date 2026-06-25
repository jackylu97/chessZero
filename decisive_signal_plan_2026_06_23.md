# Decisive-Signal Plan — escaping the draw basin (2026-06-23)

How AlphaZero, KataGo, and Leela Chess Zero inject the decisive signal that pure
self-play outcomes lack, what that means for our draw-basin root cause
(`V^π ≈ draw`, no within-position value resolution — see
`mechanistic_verdict_2026_06_19.md`), and a prioritized implementation list.

Researched from primary sources (links at the bottom). Written for posterity —
if you're reading this cold, start with the "Core insight" then the
"Prioritized list."

---

## Core insight

Every successful self-play engine **injects decisive signal that the raw game
outcome does not provide.** None of them rely on self-play outcomes alone:

- **AlphaZero** — resignation (truncate decided games, keep the decisive label)
  + a 512-ply cap as a compute backstop.
- **KataGo** — a **score-margin utility** so two winning positions are
  rank-able, plus dense auxiliary targets.
- **Leela Chess Zero** — **Syzygy tablebase rescoring** that relabels endgame
  positions with their ground-truth Win/Draw/Loss value.

Our vetoed "win adjudication" is not a hack — it is the load-bearing ingredient,
present in all three. Our own verdict already named "external decisive
supervision" as the one missing thing; the literature agrees.

---

## What each engine actually does

### AlphaZero (chess/shogi/Go) — [AZ preprint, 1712.01815]
- **Game-length cap:** games over **512 plies** are terminated and scored as a
  **draw**. This is a *compute/memory backstop*, not an outcome shaper — with a
  converting policy + resignation it almost never fires (typical decisive games
  end in 80–160 plies).
- **Resignation:** ~**80%** of self-play games resign once expected score falls
  to **≤5%**; the other **20%** are played to the end purely to **calibrate** the
  resign threshold. Calibration target (AlphaGo Zero method): set the threshold
  so **<5%** of resigned games were actually winnable (bounds false-positive
  resignations).
- **800 simulations** per move during self-play.

**Key clarifications (these tripped us up):**
1. *Resignation does not create decisive games.* Decisive games come from the
   policy-improvement ramp (random start → ~15% of games already end in mate,
   MCTS finds more → better value → more conversions). Resignation (a) ends
   decided games early to save compute, and (b) **preserves the decisive label**
   a weak policy would otherwise fumble into a draw. Every position in a resigned
   game is labeled with the decisive result.
2. *Resignation is an amplifier, not a source.* It triggers off the value head's
   own ≤5% estimate. If the value head says "draw" for everything (our dead
   zone), resignation never fires. **It complements, and must follow, the
   within-position-resolution fixes — it cannot bootstrap a flat value head.**
3. *The 512-ply cap does add draw labels,* but only bites when a winnable game is
   still unresolved at 512 — i.e. only when the policy can't convert. **Do not
   add a ply-cap-as-draw until conversion works**, or it deepens the basin.

### KataGo (Go) — [KataGo, 1902.10565]
Most relevant to our "no within-position resolution" root cause.
- **Score-margin utility:** optimizes `U = U_winloss + c_score · f((x−x0)/b)`
  where `f = (2/π)·arctan`, `x` = final score margin, **`c_score = 0.5` (→0.4
  after ~2 days)**. The arctan bounds the score term so it nudges (≤ ~half the
  win/loss magnitude), never dominates.
  - *Why it matters:* two winning positions (win by 2 vs win by 40) have
    identical win/loss value → pure WDL gives no gradient between them → flat
    value, no conversion pressure. The score term breaks the tie and gives the
    value head a **continuous** regression target → within-position resolution.
- **Auxiliary heads** (tuned so aux gradients are **10–40%** of the main heads):
  - opponent's next move (`w_opp = 0.15`)
  - board ownership per point (`w_o = 1.5/b²`)
  - final-score distribution, pdf + cdf (`w_spdf = w_scdf = 0.02`)
  - These decompose one binary outcome into hundreds of localized targets → dense
    gradient → fast value learning.
- **Playout-cap randomization:** only **p = 0.25** of moves get **full** search
  (**N = 600 → 1000**); the rest get **cheap** search (**n = 100 → 200**, Dirichlet
  noise OFF). **Only full-search moves become policy targets.** Relieves the
  value-vs-policy target tension (cheap moves → more games for value; full moves →
  deep policy targets).
- **No hard resignation.** When the loser is <5% for 5 turns, it **continues with
  reduced visits** and **down-weights** those samples
  (`prob = 0.1 + 0.9·λ`, `λ = winrate/0.05`). Cleaner targets, no mislabeling,
  and it still computes final ownership/score. *This is the defensible middle
  ground if hard adjudication's mislabeling risk is the objection.*
- **Forced playouts + policy-target pruning:** `n_forced(c) = (k·P(c)·ΣN)^(1/2)`,
  `k = 2`; prune forced/explorative visits from the policy target so the stored
  policy is decoupled from MCTS noise dynamics.

### Leela Chess Zero — [Lc0 blog: WDL, tablebase rescoring]
Closest to our domain (chess, WDL value, moves-left head).
- **WDL value head** (v0.21): added so the net can *recognize* draws and cut
  uninformative games short. **We already have this.**
- **Syzygy tablebase rescoring:** self-play runs normally, then the server
  **relabels** positions using endgame tablebases — entering a known won/drawn/
  lost TB position rescores prior moves to the true result; a losing move inside
  a TB position rescores only that short sequence. **Injects ground-truth
  decisive labels exactly where our basin forms (K+P endgames).** This is
  "persistent external value supervision," which our `external_values` path in
  `make_target` already supports.
- **Deblunder rescorer:** corrects value targets around induced (noise) blunders
  so exploration moves don't poison labels.
- **Resign adjudication** in self-play (e.g. score below a threshold for N
  consecutive moves).

---

## Implementation status (2026-06-23)

**Landed (off by default — exact back-compat; verified inert at weight 0 / resign off):**
- **Material/score-margin value utility (Tier 0 #3).** `replay_buffer.make_target`
  blends `(1-w)·outcome_onehot + w·eval_to_wdl(tanh(material_margin_stm/scale))`
  in the chess self-play branch. Margin read STM-relative from observation piece
  planes via `_material_margin_stm`. Config: `material_value_weight` (w, default
  0), `material_value_scale` (default 5.0). CLI: `--material-value-weight`,
  `--material-value-scale`. Verified: drawn K+P-up position target went
  `[0,1,0]` → `[0.22,0.78,0.00]` at w=0.3.
  - **Annealed schedule (2026-06-23):** `material_value_weight` is the INITIAL w;
    decays linearly to `material_value_weight_final` over
    `material_value_anneal_frac · training_steps` (curriculum: strong shaping
    early to escape the basin, fade to recover the true objective — chess
    material is a means, not the end). `get_material_value_weight()` in
    self_play.py; trainer passes the scheduled value per step (targets built at
    sample time → old buffer games relabel with the current w, no stale labels).
    CLI: `--material-value-weight-final`, `--material-value-anneal-frac`. frac=0
    = constant (back-compat).

**Next build (in progress):** distributional **material-margin head** (= the
score-distribution head) predicting current STM material from the LATENT state
(incl. after K dynamics unrolls → regularizes the world model to track material),
categorical over a margin support, auxiliary loss, annealable. Keeps the existing
WDL win/loss head. Model on the existing moves-left head. **Deferred:** opponent
next-move head (least relevant to the basin; inverse-dynamics + consistency
already do representation work).

**Followup C (deferred 2026-06-23) — direct material→Q in MCTS (KataGo score-in-
utility).** As built, the material head is an AUX loss only — it does NOT touch
MCTS Q; material reaches Q only INDIRECTLY via the value-target blend (value head
learns material → its output is Q). To make material shape Q *directly at search
time*, wire `predict_material` into the PUCT selection score in BOTH `mcts.py`
and `tensor_mcts.py`, gated exactly like the existing `moves_left_mcts` utility
(`|Q|>threshold` so it can never veto a genuine value verdict — important so it
can't suppress a sound winning sacrifice). Deferred because: (a) redundant-ish
with the value blend, (b) over-biases toward material grabs without careful
gating, (c) touches both bit-sensitive search engines → requires re-running the
numpy↔tensor equivalence check. Build only if the indirect path proves too slow
in Tier-0 results.
- **Consecutive-move resignation (post-hoc, engine-agnostic).** `self_play._apply_resignation`
  scans stored STM root values; if a side is below `resign_threshold` for
  `resign_consecutive` own-moves, truncate + relabel as a decisive loss. Config:
  `resign_enabled` (default False), `resign_threshold` (-0.9), `resign_consecutive`
  (5). CLI: `--resign-enabled`, `--resign-threshold`, `--resign-consecutive`.
  Label-protection only (not a compute saving); inert until the value head can
  see advantages (pair with the material utility).

**Not yet built:** Tier 1 (#4 tablebase rescoring), Tier 2 (#6 playout-cap
randomization), Tier 3. Tier 0 #1/#2 are run-config only (commands below).

## Prioritized implementation list

Ordered by impact-on-the-draw-basin per unit of effort. Tier 0 is ~free and
validates the theory before building anything.

### Tier 0 — Validate the theory + build the root-cause fix (do first)
1. **Connect4 from scratch** — positive control. Decisive game, no
   insufficient-material draw sink. If the loop ignites here, the code is fine
   and chess's problem is the warm-start dead zone. *(Preset exists; the
   2026-06-18 run died at step 1000.)*
2. **Chess from RANDOM, not Stockfish warm-start** — directly tests the dead-zone
   hypothesis (warm-start skips the decisive-from-blunders ramp). Drop
   `stockfish_injection_*` / warmstart, start cold. Cost ≈ a config change. The
   single biggest divergence from how AlphaZero actually trained.
3. **Material / score-margin term in the value target (KataGo `c_score`).**
   *(promoted to Tier 0 — 2026-06-23.)* Blend a small-weight material-balance
   signal into the value/utility so two won positions are rank-able:
   `U = WDL_value + c·tanh(material_margin / scale)`, small `c` (~0.5 cap, à la
   KataGo), ideally potential-based (see PBRS spec in
   `draw_basin_experiments_2026_06_15.md`). Material is **external/immediate** →
   sidesteps the `V^π = outcome` circularity. **Why it's the root-cause fix:** the
   basin is a degenerate fixed point where value-prediction = outcome = draw
   everywhere → zero gradient → no learning. Material margin makes the target
   *differ* from the flat draw prediction even on drawn-by-shuffle games
   (outcome=draw but margin=+1) → restores a non-zero gradient exactly where the
   basin zeroes it out. *Highest-leverage single change.* Surfaces:
   `replay_buffer._wdl_target_at` / value-target construction; optional aux score
   head in `muzero_net.py`. We already have a moves-left head = distance-to-mate
   half of this.
   - **Attribution caveat (experimental design):** run the Tier-0 controls
     (#1, #2) *clean* (no utility) AND the utility as its **own** arm, so a result
     is attributable to one change. Do NOT enable the utility inside the
     positive-control run, or you can't tell "from-random worked" from "the
     utility worked." One-flag-change discipline (cf. `hp_ablation_plan_2026_06_18.md`).
   - **Connect4 note:** connect4 has no material; its score-margin analogue is
     speed-to-win (moves-left) or threat count. The material form is chess-specific;
     keep connect4 as the clean win/loss control.

### Tier 1 — Inject within-position resolution via external supervision
4. **Syzygy tablebase rescoring of endgames (Leela).** Relabel ≤5-man positions
   (Syzygy 3–5-man ≈ 1 GB) with ground-truth WDL via the existing
   `external_values` path in `make_target`. Strongest option if we stay
   warm-started; attacks the exact K+P positions where the basin forms.

### Tier 2 — Stop the draw sink + feed the bootstrap ramp
5. **Adjudication: resignation + ply-cap-as-draw.** AlphaZero-style hard resign
   (calibrate the threshold on a 10–20% played-out holdout for <5% false
   positives) **or** KataGo's softer reduced-visit continuation + sample
   down-weighting (avoids mislabeling). **Only useful after Tier 1** gives the
   value head something to resign on. Surfaces: `self_play.py` termination +
   `replay_buffer` sample weighting.
6. **Playout-cap randomization (KataGo).** 25% full-search moves (recorded as
   policy targets) / 75% cheap-search (noise off). More games per GPU-hour for the
   value head; resolves the value/policy target tension. Pairs with the
   generation-scaled `2026-06-20` config. Surfaces: `tensor_mcts` / self-play loop.

### Tier 3 — Accelerants (partly present; lower priority)
7. **Moves-left head utility** (we have the head) + KataGo-style auxiliary targets
   — denser gradient, faster value learning.
8. **Forced playouts + policy-target pruning (KataGo)** — decouple the stored
   policy target from MCTS exploration noise.

### Sequencing recommendation
Run **Tier 0 first**. The two controls (#1 connect4, #2 chess-from-random) are
nearly free and tell us whether chess even needs the rest (if from-random chess
ignites, the warm-start was the whole problem). **#3 (material/score-margin
utility)** is now a Tier-0 build because it attacks the literal root cause our
audits identified — the zero-gradient draw fixed point — and KataGo is direct
proof it scales; run it as its **own** arm (see attribution caveat) rather than
folded into a control. **#4 (tablebase rescoring)** is the strongest add if we
stay warm-started. Hold the heavier Tier 2/3 work until Tier 0–1 shows what's
binding.

### Note on the prior veto
We previously vetoed win-adjudication. All three reference engines adjudicate in
some form, and our own mechanistic verdict named "external decisive supervision"
as the one missing thing. The evidence contradicts the veto — worth reconsidering,
ideally via KataGo's *soft* variant (continue-with-reduced-visits + down-weight),
which sidesteps the mislabeling concern that motivated the veto.

---

## Supporting fact: random chess is already ~85% draws
From 29.28B random-move games ([Labelle]): checkmate **15.46%** (W 7.734% / B
7.729%), insufficient material 65.99%, 75-move 12.01%, stalemate 6.54%. So a flat/
weak policy drawing everything is the *default*, even at random init — the question
was never "why do we draw" but "how does AlphaZero escape a 15%-decisive start."
Answer: ~15% decisive is enough gradient at scale + MCTS lookahead manufactures
conversions + adjudication keeps the decisive labels clean + co-evolution ramp.
Our warm-start lands in a dead zone that produces **~99% draws — worse than
random** — because it has unlearned blundering without learning converting.

---

## Sources
- AlphaZero preprint — game-length cap (512 plies → draw), resignation (≤5%,
  80%/20% calibration split), 800 sims: <https://ar5iv.labs.arxiv.org/html/1712.01815>
- AlphaZero in Science (2018): <https://www.science.org/doi/10.1126/science.aar6404>
- KataGo — *Accelerating Self-Play Learning in Go* (1902.10565): score-margin
  utility, auxiliary targets, playout-cap randomization, no-resignation +
  down-weighting, forced playouts / policy-target pruning:
  <https://ar5iv.labs.arxiv.org/html/1902.10565>
- Leela WDL rescale/contempt + training:
  <https://lczero.org/blog/2023/07/the-lc0-v0.30.0-wdl-rescale/contempt-implementation/>
- Leela tablebase rescoring:
  <https://lczero.org/blog/2018/08/tablebase-support-and-leela-weirdness/>
- AlphaGo Zero / AlphaZero resignation calibration (<5% false positives, 10%
  holdout): <https://en.wikipedia.org/wiki/AlphaZero>
- Random chess outcome statistics (29.28B games), Labelle:
  <https://wismuth.com/chess/random-games.html>

---

## Next run (queued 2026-06-24) — A/B for the repetition-draw penalty

**MUST include `--resign-enabled`** (user directive 2026-06-24): the warmstart
value head is calibrated (root↔SF corr +0.93), so resignation now actually fires
and attacks the won-but-shuffled threefolds from the losing side (loser resigns
before the winner can shuffle the won position to a draw + locks the decisive
label). It was wrongly left OFF in `2026_06_23_warmstart_material`.

Arm = warmstart (as before) + the new terminal-aware search + resignation:
```bash
scripts/supervise_train.sh --game chess_small --ckpt-game chess \
  --run-id 2026_06_24_warmstart_repdraw \
  --train-log logs/2026_06_24_warmstart_repdraw.log \
  --device cuda --steps 150000 --eval-interval 2000 --mask-illegal-policy \
  --stockfish-injection-path data/stockfish_injection \
  --stockfish-injection-games 300 --stockfish-injection-interval 256 \
  --self-play-warmup-steps 15000 --warmstart-buffer-size 300 \
  --warmstart-sample-frac 0.4 \
  --material-value-weight 0.5 --material-value-anneal-frac 0.6 \
  --use-material-head --material-head-loss-weight 0.25 \
  --root-terminal-draws \
  --resign-enabled
```
Baseline for the A/B = the current `2026_06_23_warmstart_material` (penalty +
resignation OFF). Watch `self_play/draw_threefold_rate` (penalty should lower the
*played* rate) and `self_play/resignation_rate` (should now be > 0).

---

## IMPLEMENTED 2026-06-25 — Root Syzygy tablebase probing (Tier 1 #4, search-time variant)

Commit `41b6fa8`. This is the search-time form of #4 (tablebase rescoring), chosen
over both the offline endgame seed (compute-heavy) and pure value-only WDL rescoring.
The decisive distinction (from the analysis that motivated it):

- **Value-only rescoring fixes the value target but NOT the policy.** Relabel a won
  KQ-K as +1 and, with a *flat* won value, MCTS still has no gradient within the won
  region → it shuffles, now "confidently winning." The model never learns the moves,
  so it can't convert against a human.
- **Root probing fixes the policy.** The TB verdict steers the *search*; the corrected
  visit distribution becomes the *policy target*, so the policy head learns the
  conversion technique. Once the policy can convert, self-play games reach real mates →
  decisive outcomes enter the buffer → the value learns won≠draw for free → the
  co-evolution loop finally ignites. This is the ignition the warmstart dead zone killed.

### Why root-only (MuZero constraint)
Leela probes WDL at internal search nodes because it searches the **true** tree.
MuZero searches in **latent space** — internal nodes are hidden states with no board,
so they can't be probed. But the root always has a real board, and **self-play visits
every position as a root**, so root-only probing covers a full conversion (each ply of
a KQ-K mate is its own root). Root-only is the only kind we *can* do, and it's enough.

### What is overwritten during MCTS
At each ply, classify the root's legal moves against Syzygy and overwrite the **root
children's `value_score`** (the Q term of PUCT, `scores = prior_score + value_score`)
in `tensor_mcts._select` — the exact slot the repetition penalty (`root_terminal_draws`)
already uses. Per-move TB value (mover POV): winning → ≈+1 minus a small DTZ penalty so
the **shortest-DTZ (progress) move scores highest**; win-throwing → draw_score/−1; NaN
(not in TB / not classifiable) → left untouched (net value used). PUCT then piles visits
on the conversion move → it's both played (argmax visits) and the policy target.

**DTZ, not flat WDL:** flat WDL ties all winning moves → still shuffles. DTZ breaks the
tie with the within-won-region progress gradient — structurally the same as the existing
moves-left (MLH) utility, but a ground-truth version injected at the root.

**Soft bias, not a policy boost:** we edit `value_score` (Q), not the prior, and keep
the override bounded — Lc0 disabled a *direct DTZ policy boost* over KLD-divergence
issues, so we steer via value and let the visit distribution stay smooth.

### Components (all gated off by default → existing runs byte-identical when off)
- `src/games/syzygy_probe.py` — `state_to_board(state, i)` (decode GPU batched state →
  python-chess) + `SyzygyRootProber.root_move_values(state, legal_mask) -> [N, A]`
  (per-move WDL/DTZ classification; GPU piece-count gate so only ≤`tb_max_pieces` games
  hit the CPU probe; FEN cache; 50-move rule → cursed/blessed = draw).
- `src/mcts/tensor_mcts.py` — `run_batch_gpu(root_tb_value=…)` gathers `[N,A]→[N,K]`
  (same as `forced_draw_mask`); `_select` override; forces non-triton backend when on.
- `src/training/self_play.py` — builds `root_tb_value` each ply in the GPU-resident loop.
- config + CLI: `tb_root_probe` / `tb_path` / `tb_max_pieces` / `tb_dtz_weight`.

### GPU-resident property preserved
The 800-sim simulation loop stays 100% on GPU. Per ply, ONE selective CPU excursion:
GPU popcount → for ≤N-piece games only, copy boards to CPU, probe, copy an `[N,A]` value
tensor back. Middlegame plies have zero in-TB games → zero overhead.

### Usage
Tablebases live in `data/syzygy` (gitignored). python-chess ships a usable small set
covering the basin endgames (KQvK/KRvK/KPvK/KBNvK + 4-5-man); copy it there, or download
the full 3-4-5-man (~1 GB). Add to any run:
```bash
... scripts/train.py --game chess_small --run-id 2026_06_25_tb_probe \
  --resume checkpoints/chess/<run>/checkpoint_<step>.pt \
  --tb-root-probe [--tb-path data/syzygy] [--tb-max-pieces 5] [--tb-dtz-weight 0.05] \
  [usual flags]
```
Resume from a checkpoint that already reaches endgames (the probe only fires at ≤N
pieces). **Watch:** conversion of reached ≤N-piece endings (`win_natural_rate` ↑,
`draw_threefold_rate` ↓) and `self_play/games_per_sec` (the per-ply CPU probe is the
throughput risk; FEN cache mitigates).

### Verified
Board round-trip; KQvK gives 24 winning moves with the DTZ gradient (1.00→0.95) and
king-blunders ≤0; end-to-end through the real MCTS the override puts >80% of an untrained
net's visits on a winning move; self-play wiring runs; terminal-draws + 7 MCTS
integration/equivalence tests still pass. Tests in `tests/test_tb_root_probe.py`.

### Open / caveats
- **Generalization** is the real unknown: teaches ≤N-piece conversions; whether the
  policy generalizes to "+3 in a middlegame" (>N pieces, no TB) is untested. If it
  doesn't, escalate to broader TB (6-man) or the seed/value-rescoring as a complement.
- **Throughput** in deep-endgame batches (many simultaneous in-TB games) — monitor; can
  narrow to winners-only DTZ or a smaller `tb_max_pieces`.
- Shares the GPU with the live run — a real TB arm wants `qboot_s800` stopped or run
  alongside; on the A100, copy `data/syzygy` over first.
