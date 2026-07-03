#!/bin/bash
# 2026_07_03_conv_anchor_fill: the anchor + rollout-fill bundle on the CONV
# architecture — a single-variable A/B against the REAL baseline (seed30,
# win-vs-random 0.68 @26k). Matched-step evals showed BOTH attention production
# runs sink to 0.22-0.44 in the 18-26k window while every conv-era run climbs to
# 0.6+ (even the on-policy ones without steering/anchor) — a general-play
# regression in the attention config, independent of the new signal channels.
# So: attention goes back to the supervised proxy ladder; the new mechanisms get
# tested on the architecture with a proven-healthy production loop.
#
# Recipe = _launch_seed30.sh EXACTLY (same resume ckpt, soft TB value at
# dtz-shape 0.0, default ML-utility params, seed frac 0.30, --no-attention to
# override the preset) PLUS only:
#   + --tb-anchor-*      (TB demonstration games, 64/256 steps, cycling)
#   + --tb-rollout-fill  (win adjudication by demonstration)
#   + --tb-relabel-workers 8 (deferred relabel — placement-only, target-identical)
#   + endgame_seeds_train.txt instead of endgame_seeds.txt (holdout hygiene —
#     the old archive contains the 15% holdout FENs used for clean eval)
#   (+ the resign color fix, in code since 9615723 — bugfix, not a variable)
# Success = beat seed30's matched-step eval curve AND raw diag off the 0.00 floor.
set -uo pipefail
cd /workspace/chessZero
RUN=2026_07_03_conv_anchor_fill
tmux kill-session -t selfplay 2>/dev/null
tmux new-session -d -s selfplay "PYTHONPATH=. .venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_small --run-id $RUN --device cuda --no-attention \
  --resume checkpoints/chess/2026_06_23_warmstart_material/checkpoint_15000.pt \
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
  2>&1 | tee /workspace/chessZero/selfplay_conv_anchor_fill.log"
echo "launched tmux 'selfplay' (run $RUN); attach: tmux attach -t selfplay"
