#!/bin/bash
# 2026_07_05_hybrid_v3: FRESH-START clean baseline (user decision 2026-07-05).
# Base = the validated hybrid + Gumbel + uniform + anchor + fill + exemption
# stack, with the bundle levers MINUS symmetry augmentation, PLUS a properly
# provisioned reward head.
#
# Post-mortem context (strategy doc §11-13): the v1/v2 regression root cause
# was the D4 augmentation parity bug (audit Finding 1, fixed in a0fd5af but
# symmetry is OFF here anyway per user direction). The dynamics-reward head is
# the search's in-tree mate detector and THE conversion-critical component
# (muted-reward diag: 0.20 -> 0.048 on identical weights) — historically a
# 1-plane afterthought, now widened to 8 planes to match value/moves-left.
#
# FRESH START (no --resume): the new reward head trains through the full
# 15k-step Stockfish warmstart phase (decisive games = dense true mate-reward
# examples from step 0). Overnight run.
#
# Deltas vs the control (2026_07_04_hybrid_gumbel):
#   --reward-head-planes 8      provision the conversion engine
#   --merged-seed-batch         one sweep, one straggler tail
#   --opening-mix-mean-plies 6  ε-mixture openings, ZERO-target opening plies
#   --tb-value-hard             TB certainty = one-hot (exonerated by probe §12)
#   --tb-policy-weight 0.5→0.2  DAgger expert action-sets at learner states
#   --seed-curriculum           DTM reverse curriculum
#   (NO symmetry augmentation; reward guard inert without it)
# SUCCESS METRIC: raw conversion at matched selfplay steps (30k here = 15k
# warmstart + 15k selfplay, same structure as control's 30k) measured under
# TRAINING CONDITIONS: diag with USE_GUMBEL=1 SIMS=400 REWARD_PLANES=8.
# Control = 0.152. Also: reward-precision probe (false-fire <= 0.12 and
# maturing), eval band 50-60, seed mate_rate ~8%, SF-agreement <= 1.05.
set -uo pipefail
cd /workspace/chessZero
RUN=2026_07_05_hybrid_v3
tmux kill-session -t selfplay 2>/dev/null
tmux new-session -d -s selfplay "PYTHONPATH=. .venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_hybrid --run-id $RUN --device cuda \
  --reward-head-planes 8 \
  --use-gumbel --gumbel-m 16 --per-alpha 0 \
  --resign-exempt-seeded \
  --merged-seed-batch \
  --opening-mix-mean-plies 6 \
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
  2>&1 | tee /workspace/chessZero/selfplay_hybrid_v3.log"
echo "launched tmux 'selfplay' (run $RUN, FRESH START); attach: tmux attach -t selfplay"
