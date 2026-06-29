#!/bin/bash
# TB-probe arm: repdraw config + root Syzygy probing, 400 sims / 512 games / 256 parallel
# (256x400 = 102400 tree-units = qboot's 128x800, proven to fit 32GB). Resumed from 15k.
cd /workspace/chessZero
.venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_small \
  --run-id 2026_06_25_tb_probe \
  --device cuda \
  --steps 150000 \
  --eval-interval 2000 \
  --mask-illegal-policy \
  --stockfish-injection-path data/stockfish_injection \
  --stockfish-injection-games 300 \
  --stockfish-injection-interval 256 \
  --self-play-warmup-steps 15000 \
  --warmstart-buffer-size 300 \
  --warmstart-sample-frac 0.4 \
  --warmstart-sample-frac-final 0.1 \
  --warmstart-anneal-frac 0.6 \
  --material-value-weight 0.5 \
  --material-value-anneal-frac 0.6 \
  --use-material-head \
  --material-head-loss-weight 0.25 \
  --root-terminal-draws \
  --root-terminal-draws-min-repeats 2 \
  --resign-enabled \
  --resign-holdout-frac 0.20 \
  --num-simulations 400 \
  --num-self-play-games 512 \
  --num-parallel-games 256 \
  --reanalyze-interval 0 \
  --tb-root-probe \
  --tb-dtz-weight 1.0 \
  --resume checkpoints/chess/2026_06_23_warmstart_material/checkpoint_15000.pt \
  2>&1 | tee logs/2026_06_25_tb_probe.log
