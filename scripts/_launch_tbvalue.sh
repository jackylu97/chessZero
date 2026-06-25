#!/bin/bash
# TB-VALUE arm: 400 sims / 512 games / 512 parallel (all games in one self-play batch;
# functionally equivalent to 256-parallel for training, just one GPU pass instead of two)
# but ADDS DTZ value-target relabeling (tb_value_weight=1.0, tb_value_dtz_shape=0.5)
# on top of the search-side probe. Starts FRESH from the SAME 15k warmstart anchor
# tb_probe used (NOT a continuation of tb_probe's 61k) — so it's a clean parallel A/B
# from a common origin: self-play runs from scratch with relabeling on from the first
# self-play game. self-play-warmup-steps=15000 == the resume step, so self-play begins
# immediately. selfplay_q_ratio stays 0 (config default) so the TB value isn't washed.
cd /workspace/chessZero

RESUME=checkpoints/chess/2026_06_23_warmstart_material/checkpoint_15000.pt
if [ ! -f "$RESUME" ]; then
  echo "ERROR: warmstart anchor $RESUME not found"; exit 1
fi
echo "Starting tb_value FRESH from the warmstart anchor (same start as tb_probe): $RESUME"

.venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_small \
  --run-id 2026_06_25_tb_value \
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
  --num-parallel-games 512 \
  --reanalyze-interval 0 \
  --tb-root-probe \
  --tb-dtz-weight 1.0 \
  --tb-value-weight 1.0 \
  --tb-value-dtz-shape 0.5 \
  --resume "$RESUME" \
  2>&1 | tee logs/2026_06_25_tb_value.log
