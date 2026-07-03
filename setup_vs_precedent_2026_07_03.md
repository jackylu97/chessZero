# Setup vs Precedent — audit of every active mechanism (2026-07-03)

Compares our full training stack against AlphaZero/MuZero (DeepMind), Lc0, and
KataGo, from primary sources (papers, blogs, production configs, source code —
three research passes; all URLs at bottom). Purpose: (1) change ledger of
everything we run, (2) which parts are rigorously validated vs original,
(3) internal conflicts / counterproductive interactions.

Verdict legend: **EXT** = validated externally (published/production elsewhere) ·
**INT** = validated by our own experiments · **PART** = partial/adapted precedent ·
**OFF** = off-script/original, unvalidated beyond our runs.

---

## 1. Change ledger — the full active stack (hybrid v2, 2026-07-03)

### Architecture (chess_hybrid preset, commit 11e3071)
| mechanism | ours | precedent | verdict |
|---|---|---|---|
| MuZero latent search (rep/dyn/pred) | 5.95M inference, 400 sims | DeepMind: 16 blocks × 256 (~40M+), **800 sims**, 1000 TPUs self-play | PART (scale far below reference) |
| Conv-SE stem before attention body | 2 SE blocks, d128, rep+dyn | SE = KataGo/Lc0-era conv nets (large Elo win); conv-stem-before-transformer = CoAtNet/vision | PART (components EXT, combination ours) |
| Attention body (4L, d128, 4 heads) | rep + dyn, matched | Lc0 BT2-4: 15L, 768-1024d, **82-191M** — crossover vs conv happened there | OFF at this scale (INT: endgame proxy only) |
| Smolgen | on, shared final proj | Designed/validated at BT2+ (82M+). Chessformer (6M) skips it, uses Shaw relative position encodings: "good position representation can in large part replace model scale" | **OFF at our scale** (our proxy: EXP A ≈ EXP C — smolgen not the gain source) |
| FFN width 2× embed | d_ff=256 @ d128 | Lc0: chess transformers "don't benefit from large FFN" — BT uses **1.3-1.5×** | EXT-consistent (do NOT widen to 4×; retracts an earlier suggestion) |
| Shared attention prediction body (2L) → policy+value+ML | pred_attn_layers=2 | AZ/Lc0/KataGo: shared trunk, thin heads. Lc0 T78-80: attention policy head **on conv bodies** | PART |
| Conv policy head (per-square × 73) | zero-init proj | AZ-style; Lc0 moved to from→to attention policy head (+270 policy Elo by BT4) | EXT (upgrade path known) |
| WDL value head | 3-way CE | Lc0 since T50 | EXT |
| Moves-left head | categorical support (2K+1), CE | Lc0 v0.25: **scalar + Huber** on plies_left | PART (target semantics same; parameterization ours) |
| Material head (aux, raw latent) | STM material CE | KataGo ownership/score-dist heads: aux decomposition of outcome, ~190 Elo ablation | PART (KataGo analog; ours much weaker signal) |
| SimSiam consistency + inverse-dynamics | on | EfficientZero (Atari); **LightZero board-game defaults: SSL OFF** | PART — see conflict C5 |
| LayerNorm everywhere | on | LightZero actual default is BN (docstring says LN); Dong et al.: LN does NOT prevent rank collapse, skips do | INT (our BN failure was measured) |

### Search (tensor MCTS)
| mechanism | ours | precedent | verdict |
|---|---|---|---|
| PUCT + Dirichlet root noise | α=0.3, ε=0.25-ish | AZ/Lc0/KataGo standard; Lc0 training: α=0.12, ε=0.1 | EXT |
| Moves-left MCTS utility, \|Q\|-gated | slope .005-.02, max .1-.3, thresh .3 | Lc0 **training** self-play: slope .007, max .2, **threshold 0.0**, quadratic scaling | EXT (ours mid-range) |
| root_terminal_draws (stalemate mask at root) | on | no direct precedent; addresses MuZero interior-terminal blindness ("Demystifying MuZero": design gap, no published fix) | OFF (INT: cut stalemates in proxy) |
| tb_root_probe (collect-only; steering retired) | relabel targets only | = Lc0 rescorer philosophy ("fix labels, not games") | EXT |
| No forced playouts / no policy target pruning | — | KataGo: forced playouts k=2 + prune noise-visits from targets ("decouple policy target from exploration noise") | missing EXT — see R4 |
| Fixed 400 sims every move | — | KataGo playout cap randomization (75% cheap @200, 25% full @1000, only full recorded); Lc0 KLDGain early-stop | missing EXT — see R5 |

### Targets & data (the decisive-signal stack)
| mechanism | ours | precedent | verdict |
|---|---|---|---|
| Value target: MC outcome z (td=-1, γ=1) | WDL one-hot | MuZero board games exactly this ("bootstrap to end of game") | EXT (also the V^π≈draw root cause — MuZero Unplugged's TD-5-to-target-net is the published alternative) |
| TB VALUE relabel @1.0 (Syzygy WDL) | in-TB plies | **Lc0 rescorer**: rewrites result_q/d from Syzygy, propagates BACKWARD through earlier plies | EXT — but see C2b (we don't back-propagate pre-TB; the FILL covers that instead) |
| TB MOVES-LEFT relabel (Gaviota DTM) | in-TB plies | Lc0 rescorer ApplyGaviotaCorrections (prod since T70) | EXT |
| TB POLICY relabel (tb_policy_weight) | built, OFF | Lc0 dtz_policy_boost: tried, **rejected** (KLD interaction we don't have) | PART (correctly off; viable if needed) |
| **TB anchor** (20k TB-optimal demo games, cycling 64/256) | on | Closest: KataGo startPoses/hintPoses (external positions, 4× visits, more full searches; "mild benefits"); DQfD (demos kept in replay with fixed priority bonus); AlphaGo SL bootstrap. **No engine injects full TB demo GAMES** | **OFF** (INT: conv diag floor→ceiling in 14k steps) |
| **TB rollout fill** (truncate at decisive TB ply + TB finish + true z) | on | Lc0: resign-adjudication z + rescorer backward z-propagation + deblunder (z := best_q after blunders) = same *goal*; nobody extends games with oracle play and trains on the tail | **OFF** (INT: works; the demo-splice is the novel part) |
| Endgame seeding (30% of games from TB FENs) | on, train/holdout split | KataGo startPoses/forked openings (validated, mild); Lc0: **never** (startpos + books only) | PART |
| Stockfish warmstart + injection anchor | 15k warmup + 300-game pool refresh | AlphaGo SL bootstrap (pre-Zero); AZ/Lc0/KataGo tabula rasa | PART (INT: prevents value collapse — run-comparison A/B) |
| Material value TARGET blend (annealed 0.5→0) | pre-TB plies | **KataGo score utility is SEARCH-side only; value label stays pure z** | **OFF + now conflicts with fill** — C2 |
| Resignation as z (normal games) | thresh -0.9×5, 20% holdout | **Lc0 does exactly this** (resign 2%, playthrough 20%, adjudicated result IS z); KataGo never resigns (soft-resign: reduced visits + 0.1× weight) | EXT (Lc0-style; KataGo's is stricter) |
| Seed resign exemption (v2, 5eca27e) | on | closer to KataGo play-to-completion | PART |
| **PER on chess** (α=0.6) | on | MuZero: PER **Atari only, board games uniform**; KataGo uniform window; Lc0 uniform + SKIP=32 + value-focus | **OFF, 3-way consensus against** — C1 |
| Reanalyze | OFF | MuZero Unplugged: central at scale; LightZero board defaults: 0 | consistent with small-scale practice — but see C4 |
| Zeroing-aware DTZ move ranking | playout + policy targets | standard TB-play lore (min-DTZ trap); found via our tests | INT |

---

## 2. Conflicts & counterproductive interactions

**C1. PER vs the anchor (highest priority).** Three independent production systems
sample uniformly for board games; we run PER keyed on value-TD error. Anchor/fill
games have trivially-fittable value targets → their priority decays → the policy
demonstrations get starved exactly as they're learned least. DQfD's published fix:
demonstrations get a FIXED priority bonus (ε_demo=1.0 vs ε_agent=0.001) so
sampling never abandons them. Options: (a) uniform sampling (precedent-consistent),
(b) DQfD-style floor priority for anchor/filled games, (c) Lc0-style value-focus
acceptance sampling. Flat policy_loss in the attn run was the early symptom.

**C2. Material target blend now dilutes the fill's z.** KataGo keeps score shaping
in SEARCH utility; the value label stays pure outcome. Ours blends material into
the TARGET at pre-TB plies — which, post-fill, are exactly the plies whose z we
just made TRUE. At w=0.5 early, a won-and-filled game's middlegame target is half
"material says maybe"; the blend anneals to 0 by 90k but fights the fill until
then. It was a draw-basin stopgap; the fill supersedes it. Retire it (or move to
search utility, the KataGo placement).
**C2b (gap, same family):** Lc0's rescorer propagates the TB verdict BACKWARD to
all pre-TB plies of the game; our per-ply relabel doesn't — for UNFILLED games
(TB-drawn entries, fill-skipped games) pre-TB plies keep the played z. The fill
covers decisive contradictions only. Minor; worth knowing.

**C3. Dirichlet noise pollutes our policy targets.** KataGo forced-playouts +
target pruning exists precisely to scrub exploration noise from targets; Lc0's
low α=0.12 limits it. We inject α=0.3-style noise and store raw visit fractions.
Our own α-sweep showed the noise does little steering good; it still lands in
every target. Adopting KataGo's pruning is validated, self-contained, and
composes with everything else.

**C4. Reanalyze would DESTROY anchor/fill targets if naively enabled.** Reanalyze
overwrites policies/root_values in place with the current (weaker-in-endgame)
net's search. Warmstart games are excluded; anchor/filled games are NOT (they
look like self-play). Guard needed before reanalyze is ever turned on: exempt
games with tb_filled or TB-authored policies.

**C5. SSL consistency + attention body = dimensional-collapse pressure.**
LightZero ships SSL OFF for board games; Tang et al. 2023 show latent
self-prediction tends toward collapse (stop-grad+predictor is the counterweight);
our own measurement: hybrid eff_rank 121 vs conv 220, crosspos_cos 0.87 vs 0.70
(r2 intact so far — geometry warning, not failure). No published VICReg+MuZero
pairing exists (novel if we add the variance/covariance terms). Watch
eff_rank+r2 together; act only if both fall.

**C6. Smolgen at 128d is an unvalidated ornament.** Designed at 82-191M; our own
proxy showed no clear smolgen gain (EXP A no-smolgen ≈ EXP C smolgen); the 6M
Chessformer uses Shaw relative position encodings instead and argues position
representation substitutes for scale. Candidate simplification/swap for the next
architecture rung: RPE instead of (or alongside) smolgen.

**C7. Redundant decisive-value channels.** Post-fill, the value target at in-TB
plies is set by (in precedence): TB relabel @1.0 ← material blend ← outcome z —
three writers where one (relabel) wins anyway; pre-TB plies get fill-z + material
dilution (C2). The stack works but has more moving parts than functions. After
C2 retirement: relabel owns in-TB, fill owns pre-TB z, outcome owns the rest —
one writer per ply class.

**C8. Retraction: don't widen FFN to 4×.** I earlier suggested 4× FFN as a rank
counterweight; Lc0's chess-specific finding is FFN 1.3-1.5× embed ("don't seem
to benefit from large FFN sizes"). Our 2× is already right-sized.

---

## 3. What's rigorously validated vs original (summary)

**Validated externally + running:** WDL head · MLH + gated search utility (Lc0
training values bracket ours) · TB value/DTM rescoring (Lc0 rescorer, production
since T70) · resignation-as-z with 20% playthrough (Lc0's exact numbers) ·
conv-SE blocks (KataGo/Lc0 era) · STM-relative encoding · LayerNorm-not-BN for
recurrent unroll (ours measured; literature consistent) · pure-z MC value target
(MuZero board games).

**Validated internally (our experiments):** attention rep+dyn for endgame
geometry (proxy 4%→41%, KQvK 0.91) · warmstart anchor prevents value collapse
(A/B corr 0.89 vs 0.34) · anchor+fill transfer technique to conv at its
supervised ceiling in 14k steps (raw diag 0.058, first nonzero KQvK) · seed
resign exemption (all-cost-no-benefit analysis) · zeroing-aware DTZ ranking
(trap reproduced + fixed under test).

**Original / off-script (watch these hardest):** TB anchor demo-game injection
(nearest: hintPoses + DQfD) · rollout fill demonstration splice (nearest: Lc0
adjudication + deblunder, but nobody extends games) · endgame seeding at 30%
(KataGo startposes is a milder cousin) · PER on chess (against 3-way consensus)
· material TARGET blend (KataGo does it search-side) · smolgen at ≤6M ·
hybrid stem+attention at 6M · root_terminal_draws · shared pred body feeding
the ML head.

---

## 4. Ranked recommendations (next-run levers, cheapest first)

R1. **Fix PER×anchor** (C1): uniform sampling for board games, or DQfD floor
    priority on anchor/filled games. One-line-ish; precedent-backed both ways.
R2. **Retire the material target blend** (C2): fill+relabel own decisive signal
    now. (Keep the material HEAD — it's the KataGo-aux analog, and world-model
    regularization is its real job.)
R3. **Reanalyze guard** (C4): exempt TB-authored targets before it's ever re-enabled.
R4. **KataGo policy-target pruning + forced playouts** (C3): validated, scrubs
    Dirichlet noise from targets, independent of everything else.
R5. **Playout cap randomization** (KataGo p=0.25 full/0.75 cheap): ~2-3× more
    games per GPU-hour for the value head at equal policy-target quality —
    directly relieves our biggest bottleneck (self-play wall-clock).
R6. **Attention policy head from→to** (Lc0 T78-80 validated it on CONV bodies —
    we don't need the full transformer to get the +policy-Elo).
R7. **Shaw RPE vs smolgen** at our scale (C6) — proxy-ladder rung.
R8. Value/diff-focus acceptance sampling (Lc0) as the modern non-PER
    prioritization if R1's uniform feels too flat.
R9. **Gumbel retry** once the value head has sibling resolution (guarantees are
    conditional on q̂ signal — the draw-basin failure was the precondition, not
    the method). Enables the 25M model at viable wall-clock.

## 5. Sources
Lc0: transformer-progress blog (BT1-4, smolgen dims, FFN finding) · project-history wiki
(rescorer/MLH/WDL/960-book timeline) · rescorer.cc (WDL/DTM/policy relabel, deblunder) ·
selfplay/game.cc + training.lczero.org/training_runs (resign-adjudication params, KLDGain,
MLH training values) · lczero-training (uniform sampling, SKIP=32, value focus, q_ratio removal).
KataGo: arXiv:1902.10565 (score utility, playout cap randomization, forced playouts/pruning,
soft resign, uniform window, ownership ablation) · KataGoMethods.md · selfplay8b20.cfg ·
play.cpp (hintPoses 4× visits) · README/TrainingHistory (external data stance).
DeepMind/derivatives: arXiv:1911.08265 (MuZero appendix: 800 sims, uniform board sampling,
absorbing terminals) · arXiv:2104.06294 (Reanalyse/TD-5) · arXiv:2111.00210 (EfficientZero) ·
LightZero repo+paper (board defaults) · arXiv:2103.03404 (rank collapse; skips not LN) ·
arXiv:2411.04580 (beyond-terminal states) · arXiv:1704.03732 (DQfD) · AlphaGo Nature 2016 ·
arXiv:2409.12272 (Chessformer, RPE at 6M) · arXiv:2212.03319 / 2401.08898 (self-predictive collapse).
