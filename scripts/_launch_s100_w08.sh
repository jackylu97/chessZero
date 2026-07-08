#!/usr/bin/env bash
# 2026_06_27_s100_w08: PRIOR-lever test — tb_policy_weight 0.8 CONSTANT, resume s100 ckpt_30000 (value already recalibrated margin 0.85), seed_frac=1.0. Does the stronger policy relabel finally move the prior off the liquidating move + lift conversion?
set -euo pipefail
cd /workspace/chessZero
RUN=2026_06_27_s100_w08
exec .venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_small --run-id "$RUN" --device cuda \
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
  --tb-value-weight 1.0 --tb-value-dtz-shape 0.5 --tb-moves-left-weight 1.0 \
  --tb-policy-weight 0.8 --tb-policy-weight-final 0.0 --tb-policy-anneal-frac 0.0 \
  --endgame-seed-frac 1.0 --endgame-seed-archive data/endgame_seeds.txt \
  --resume checkpoints/chess/2026_06_27_polrelabel_s100/checkpoint_30000.pt
