#!/bin/bash
# 2026_07_05_hybrid_bundle_v2: bundle + REWARD-PRECISION GUARD (commit 7d46a3a) — the FROZEN next-run bundle on the validated base
# (hybrid arch + Gumbel + uniform + anchor + fill + seed exemption — identical
# to 2026_07_04_hybrid_gumbel, which is the matched-steps CONTROL). Deltas, all
# gated flags built + equivariance-tested + smoked together (commits a572188,
# 6667302; strategy doc §10):
#   --merged-seed-batch          one sweep, one straggler tail (~15-30% wall)
#   --opening-mix-mean-plies 6   ε-mixture openings (policy-sampled @T1.5 +
#                                15% uniform floor), ZERO-target opening plies
#   --tb-value-hard (+shape 0)   TB certainty = one-hot (fixes ±0.88/±1 inversion)
#   --tb-policy-weight 0.5→0.2   DAgger: expert action-SETS at learner-visited
#                                in-TB states (the corridor/dither fix)
#   --symmetry-augment           D4 random transform per pawnless sample window
#   --seed-curriculum            DTM reverse curriculum (easy 8 → 100 over 50%)
# Sims stay 400 (freeze discipline; sims A/B deferred to big-GPU prep).
# SUCCESS METRIC: raw conversion at 30k measured under TRAINING CONDITIONS
# (diag with USE_GUMBEL=1 SIMS=400) vs the control's 0.15; secondary: eval
# band vs control's 50-60, seed mate_rate vs ~8%, SF-agreement ≤1.05.
set -uo pipefail
cd /workspace/chessZero
RUN=2026_07_05_hybrid_bundle_v2
tmux kill-session -t selfplay 2>/dev/null
tmux new-session -d -s selfplay "PYTHONPATH=. .venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_hybrid --run-id $RUN --device cuda \
  --resume checkpoints/chess/2026_07_03_hybrid_anchor_fill/checkpoint_15000.pt \
  --use-gumbel --gumbel-m 16 --per-alpha 0 \
  --resign-exempt-seeded \
  --merged-seed-batch \
  --opening-mix-mean-plies 6 \
  --symmetry-augment \
  --seed-curriculum \
  --tb-value-hard \
  --tb-policy-weight 0.5 --tb-policy-weight-final 0.2 --tb-policy-anneal-frac 0.6 \
  --steps 150000 --eval-interval 2000 --mask-illegal-policy \
  --stockfish-injection-path data/stockfish_injection \
  --stockfish-injection-games 300 --stockfish-injection-interval 256 \
  --self-play-warmup-steps 15000 --warmstart-buffer-size 300 \
  --warmstart-sample-frac 0.4 --warmstart-sample-frac-final 0.1 --warmstart-anneal-frac 0.6 \
  --material-value-weight 0.5 --material-value-anneal-frac 0.6 \
  --use-material-head --material-head-loss-weight 0.25 \
  --root-terminal-draws --root-terminal-draws-min-repeats 2 \
  --resign-enabled --resign-holdout-frac 0.20 \
  --num-simulations 400 --num-self-play-games 512 --num-parallel-games 512 \
  --reanalyze-interval 0 \
  --tb-root-probe --tb-path data/syzygy --tb-gaviota-path data/gaviota \
  --tb-relabel-workers 8 \
  --tb-value-weight 1.0 --tb-value-dtz-shape 0.0 --tb-moves-left-weight 1.0 \
  --endgame-seed-frac 0.30 --endgame-seed-archive data/endgame_seeds_train.txt \
  --tb-anchor-path data/tb_anchor --tb-anchor-games 64 --tb-anchor-interval 256 \
  --tb-rollout-fill \
  2>&1 | tee /workspace/chessZero/selfplay_hybrid_bundle_v2.log"
echo "launched tmux 'selfplay' (run $RUN); attach: tmux attach -t selfplay"
