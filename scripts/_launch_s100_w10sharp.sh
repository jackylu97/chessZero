#!/usr/bin/env bash
# 2026_06_27_s100_w10sharp: SHARP/CLEAN policy target test — tb_policy_weight 1.0 (zero the throw-favoring visit blend) + tb_policy_temp 0.15 (sharper toward DTZ-best win move). Resume w08 ckpt_32000 (value calibrated). Tests: does a HARD policy signal move the emitted prior off the throw, or is it a policy-head capacity wall? Watch the emitted prior at liquidation positions + seed/mate_rate.
set -euo pipefail
cd /workspace/chessZero
RUN=2026_06_27_s100_w10sharp
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
  --tb-policy-weight 1.0 --tb-policy-weight-final 0.0 --tb-policy-anneal-frac 0.0 --tb-policy-temp 0.15 \
  --endgame-seed-frac 1.0 --endgame-seed-archive data/endgame_seeds.txt \
  --resume checkpoints/chess/2026_06_27_s100_w08/checkpoint_32000.pt
